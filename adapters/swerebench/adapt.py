"""SWE-rebench-openhands-trajectories -> OpenInference adapter (local parquet, iter_batches).

Mapping is identical to v2 (1 trajectory -> trace + AGENT root + LLM/TOOL children,
tenant_id=repo). The only change is the data path: read directly from the dataset's
trajectories.parquet (downloaded from HF) using pyarrow's row-group streaming.
This bypasses the `datasets` library which OOMs at metadata-load time on this VPS.

Run (after `curl -L -o /tmp/swerebench-trajectories.parquet ...` is complete):
    uv run --with pyarrow python adapt.py
"""
from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

OUT_DIR = Path(__file__).parent / "output"
OUT_PARQUET = OUT_DIR / "spans.parquet"
SOURCE_PARQUET = Path("/tmp/swerebench-trajectories.parquet")
SAMPLE_N = 5000              # trajectories to ingest
ROW_BATCH_SIZE = 100         # input rows per pyarrow iter_batches call
FLUSH_TRAJECTORIES = 200     # output flush cadence


def _safe_json(s):
    if not isinstance(s, str):
        return s
    try:
        return json.loads(s)
    except Exception:
        return None


def _coerce_trajectory(val):
    """Trajectory may be a list[dict], a JSON string, or a numpy/pyarrow list[struct]."""
    if val is None:
        return None
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        return _safe_json(val)
    # pyarrow scalar list -> python list
    if hasattr(val, 'as_py'):
        return val.as_py()
    return None


def _extract_repo(row: dict) -> str:
    for key in ("repo", "repository", "instance_id", "task"):
        v = row.get(key)
        if v:
            return str(v)
    iid = row.get("instance_id") or ""
    m = re.match(r"([^_]+)__([^-]+)", str(iid))
    if m:
        return f"{m.group(1)}/{m.group(2)}"
    return "unknown"


def _walk_trajectory_to_spans(traj, trace_id, agent_span_id, tenant_id, t0, model_default="unknown"):
    """OpenHands chat-message trajectory:
       - role=assistant -> LLM span (parent=AGENT). If tool_calls present, emit
         TOOL spans (parent=LLM) for each.
       - role=tool -> attach the tool result content's length to the prior TOOL
         span's duration (best-fit synthesis; no wall clock in source).
       - role=system/user -> no span.
    """
    llm_spans = []
    tool_spans = []
    cur = t0
    last_tool_span_by_id: dict[str, dict] = {}  # tool_call_id -> tool span dict

    if not isinstance(traj, list):
        return llm_spans, tool_spans, cur

    for step in traj:
        if not isinstance(step, dict):
            continue
        role = step.get("role") or ""
        content = step.get("content") or ""

        if role == "assistant":
            dur_ms = max(200, min(5000, 300 + len(str(content)) // 2))
            llm_end = cur + timedelta(milliseconds=dur_ms)
            in_tok  = max(0, len(str(content)) // 4)
            out_tok = max(0, len(str(content)) // 4)
            llm_id = uuid.uuid4().hex
            attrs = {"llm": {"model_name": model_default,
                             "input_tokens": in_tok, "output_tokens": out_tok}}
            llm_spans.append({
                "trace_id": trace_id,
                "span_id": llm_id,
                "parent_id": agent_span_id,
                "name": f"llm.completion.{model_default}",
                "start_time": cur,
                "end_time": llm_end,
                "status_code": "OK",
                "status_message": None,
                "span_kind": "LLM",
                "llm_token_count_prompt": int(in_tok),
                "llm_token_count_completion": int(out_tok),
                "attributes": json.dumps(attrs),
                "tenant_id": tenant_id,
            })
            tcur = llm_end

            # Tool calls dispatched by this assistant turn
            for tc in (step.get("tool_calls") or []):
                if not isinstance(tc, dict):
                    continue
                fn = (tc.get("function") or {})
                tool_name = fn.get("name") or "tool"
                tool_args = fn.get("arguments") or ""
                tc_id = tc.get("id") or uuid.uuid4().hex
                t_dur_ms = max(50, min(2000, 100 + len(str(tool_args)) // 4))
                t_end = tcur + timedelta(milliseconds=t_dur_ms)
                ts = {
                    "trace_id": trace_id,
                    "span_id": uuid.uuid4().hex,
                    "parent_id": llm_id,
                    "name": f"tool.{tool_name}",
                    "start_time": tcur,
                    "end_time": t_end,
                    "status_code": "OK",
                    "status_message": None,
                    "span_kind": "TOOL",
                    "llm_token_count_prompt": None,
                    "llm_token_count_completion": None,
                    "attributes": json.dumps({"tool": {"name": tool_name, "tool_call_id": tc_id}}),
                    "tenant_id": tenant_id,
                }
                tool_spans.append(ts)
                last_tool_span_by_id[tc_id] = ts
                tcur = t_end
            cur = tcur

        elif role == "tool":
            # Extend the matching tool span's duration by content length (best-fit)
            tc_id = step.get("tool_call_id") or ""
            ts = last_tool_span_by_id.get(tc_id)
            if ts is not None:
                extra_ms = max(0, min(2000, len(str(content)) // 8))
                ts["end_time"] = ts["end_time"] + timedelta(milliseconds=extra_ms)
                cur = max(cur, ts["end_time"])

    return llm_spans, tool_spans, cur


def _build_schema():
    import pyarrow as pa
    return pa.schema([
        pa.field("trace_id",                   pa.string(), nullable=False),
        pa.field("span_id",                    pa.string(), nullable=False),
        pa.field("parent_id",                  pa.string(), nullable=True),
        pa.field("name",                       pa.string(), nullable=False),
        pa.field("start_time",                 pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("end_time",                   pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("status_code",                pa.string(), nullable=False),
        pa.field("status_message",             pa.string(), nullable=True),
        pa.field("span_kind",                  pa.string(), nullable=False),
        pa.field("llm_token_count_prompt",     pa.int64(),  nullable=True),
        pa.field("llm_token_count_completion", pa.int64(),  nullable=True),
        pa.field("attributes",                 pa.string(), nullable=True),
        pa.field("tenant_id",                  pa.string(), nullable=False),
    ])


def _flush(rows, writer, schema):
    import pyarrow as pa
    if not rows:
        return 0
    cols = {f.name: [r[f.name] for r in rows] for f in schema}
    table = pa.table(cols, schema=schema)
    writer.write_table(table)
    return table.num_rows


def adapt():
    import pyarrow.parquet as pq

    if not SOURCE_PARQUET.exists():
        raise SystemExit(f"missing {SOURCE_PARQUET}; download trajectories.parquet first")

    schema = _build_schema()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"opening {SOURCE_PARQUET} ({SOURCE_PARQUET.stat().st_size:,} bytes)", flush=True)
    src = pq.ParquetFile(str(SOURCE_PARQUET))
    print(f"row groups: {src.num_row_groups}, total rows: {src.metadata.num_rows}", flush=True)
    print(f"columns: {[f.name for f in src.schema_arrow]}", flush=True)

    n_traj = 0
    n_skipped = 0
    n_spans_total = 0
    tenant_counts: dict[str, int] = {}
    kind_counts: dict[str, int] = {"AGENT": 0, "LLM": 0, "TOOL": 0}
    batch_rows: list[dict] = []
    t_anchor = datetime.now(timezone.utc)

    with pq.ParquetWriter(str(OUT_PARQUET), schema, compression="snappy") as writer:
        for batch in src.iter_batches(batch_size=ROW_BATCH_SIZE):
            if n_traj >= SAMPLE_N:
                break
            for row_dict in batch.to_pylist():
                if n_traj >= SAMPLE_N:
                    break
                n_traj += 1

                traj_raw = (row_dict.get("trajectory") or row_dict.get("steps") or
                            row_dict.get("messages")    or row_dict.get("history") or
                            row_dict.get("trajectory_json"))
                traj = _coerce_trajectory(traj_raw)
                if not traj:
                    n_skipped += 1
                    continue

                tenant_id = _extract_repo(row_dict)
                model = row_dict.get("model") or row_dict.get("llm_model") or "unknown"
                t0 = t_anchor - timedelta(seconds=n_traj)

                trace_id = uuid.uuid4().hex
                agent_span_id = uuid.uuid4().hex

                llm_spans, tool_spans, t_end = _walk_trajectory_to_spans(
                    traj, trace_id, agent_span_id, tenant_id, t0, model_default=model,
                )
                if not llm_spans:
                    n_skipped += 1
                    continue

                batch_rows.append({
                    "trace_id": trace_id,
                    "span_id": agent_span_id,
                    "parent_id": None,
                    "name": "swerebench.trajectory",
                    "start_time": t0,
                    "end_time": t_end,
                    "status_code": "OK",
                    "status_message": None,
                    "span_kind": "AGENT",
                    "llm_token_count_prompt": None,
                    "llm_token_count_completion": None,
                    "attributes": json.dumps({"agent": {
                        "trajectory_steps": len(traj),
                        "model": str(model),
                    }}),
                    "tenant_id": tenant_id,
                })
                batch_rows.extend(llm_spans)
                batch_rows.extend(tool_spans)

                tenant_counts[tenant_id] = tenant_counts.get(tenant_id, 0) + 1 + len(llm_spans) + len(tool_spans)
                kind_counts["AGENT"] += 1
                kind_counts["LLM"]   += len(llm_spans)
                kind_counts["TOOL"]  += len(tool_spans)

                if n_traj % FLUSH_TRAJECTORIES == 0:
                    n_spans_total += _flush(batch_rows, writer, schema)
                    batch_rows.clear()
                    print(f"  ... {n_traj} trajectories -> {n_spans_total} spans (skipped {n_skipped})",
                          flush=True)

        if batch_rows:
            n_spans_total += _flush(batch_rows, writer, schema)
            batch_rows.clear()

    print(f"adapted {n_traj} trajectories -> {n_spans_total} spans (skipped {n_skipped})", flush=True)
    print(f"wrote {OUT_PARQUET} ({OUT_PARQUET.stat().st_size:,} bytes)", flush=True)

    print("\ntenant cardinality (top 20):", flush=True)
    for tenant, n in sorted(tenant_counts.items(), key=lambda kv: -kv[1])[:20]:
        print(f"  {tenant}: {n} spans", flush=True)
    print("\nspan_kind distribution:", flush=True)
    for kind, n in sorted(kind_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {kind}: {n}", flush=True)


if __name__ == "__main__":
    adapt()
