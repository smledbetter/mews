"""WildChat-1M -> OpenInference adapter (sample-scale).

Mapping:
- 1 conversation -> 1 trace
- conversation -> 1 AGENT span (root, span_kind=AGENT)
- each assistant turn -> 1 LLM span (parent_id = AGENT span_id, span_kind=LLM)
- tenant_id = country (or "unknown")

Latency: WildChat doesn't carry per-turn wall-clock, so we synthesize ~content-length-proportional
durations as a best-fit placeholder (200-3000ms per LLM turn, scaled by output length).
This gives the locked reference query (latency_by_model_under_agent) something
non-degenerate to aggregate. Documented as synthesis disclosure in README.md.

Run:
    uv run --with datasets --with pandas --with duckdb python adapt.py
"""
from __future__ import annotations

import json
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

OUT_DIR = Path(__file__).parent / "output"
OUT_PARQUET = OUT_DIR / "spans.parquet"
SAMPLE_N = 10000  # conversations to ingest


def _ts_to_dt(ts):
    if ts is None:
        return datetime.now(timezone.utc)
    if isinstance(ts, datetime):
        return ts.replace(tzinfo=timezone.utc) if ts.tzinfo is None else ts
    if isinstance(ts, str):
        try:
            return datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except Exception:
            return datetime.now(timezone.utc)
    if isinstance(ts, (int, float)):
        return datetime.fromtimestamp(ts, tz=timezone.utc)
    return datetime.now(timezone.utc)


def adapt():
    from datasets import load_dataset
    print(f"loading WildChat-1M (streaming, first {SAMPLE_N} convos)...", flush=True)
    ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)

    spans = []
    n_convos = 0
    n_skipped = 0
    for row in ds:
        if n_convos >= SAMPLE_N:
            break
        n_convos += 1

        msgs = row.get("conversation") or []
        if not msgs:
            n_skipped += 1
            continue

        country = row.get("country") or "unknown"
        ts_root = _ts_to_dt(row.get("timestamp"))
        model = row.get("model") or "unknown"

        trace_id = uuid.uuid4().hex
        agent_span_id = uuid.uuid4().hex

        cur = ts_root
        agent_end = ts_root
        llm_spans = []
        for msg in msgs:
            role = (msg or {}).get("role")
            if role != "assistant":
                continue
            content = (msg or {}).get("content") or ""
            # Synthetic latency: 200ms baseline + 1ms per char output, capped 3000ms.
            dur_ms = max(200, min(3000, 200 + len(content) // 1))
            llm_start = cur
            llm_end = cur + timedelta(milliseconds=dur_ms)

            attrs = {
                "llm": {
                    "model_name": model,
                    "input_tokens": len(content) // 4,
                    "output_tokens": len(content) // 4,
                },
                "input": {"value": f"input-{uuid.uuid4().hex[:8]}"},
                "output": {"value": f"output-{uuid.uuid4().hex[:8]}"},
            }
            llm_spans.append({
                "trace_id": trace_id,
                "span_id": uuid.uuid4().hex,
                "parent_id": agent_span_id,
                "name": f"llm.completion.{model}",
                "start_time": llm_start,
                "end_time": llm_end,
                "status_code": "OK",
                "status_message": None,
                "span_kind": "LLM",
                "llm_token_count_prompt": attrs["llm"]["input_tokens"],
                "llm_token_count_completion": attrs["llm"]["output_tokens"],
                "attributes": json.dumps(attrs),
                "tenant_id": country,
            })
            cur = llm_end
            agent_end = llm_end

        if not llm_spans:
            n_skipped += 1
            continue

        spans.append({
            "trace_id": trace_id,
            "span_id": agent_span_id,
            "parent_id": None,
            "name": "wildchat.conversation",
            "start_time": ts_root,
            "end_time": agent_end,
            "status_code": "OK",
            "status_message": None,
            "span_kind": "AGENT",
            "llm_token_count_prompt": None,
            "llm_token_count_completion": None,
            "attributes": json.dumps({"convo": {"turns": len(msgs), "model": model}}),
            "tenant_id": country,
        })
        spans.extend(llm_spans)

        if n_convos % 1000 == 0:
            print(f"  ... {n_convos} convos -> {len(spans)} spans (skipped {n_skipped})", flush=True)

    print(f"adapted {n_convos} convos -> {len(spans)} spans (skipped {n_skipped})", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    import pandas as pd
    df = pd.DataFrame(spans)

    import duckdb
    con = duckdb.connect(":memory:")
    con.register("_df", df)
    con.execute(f"COPY _df TO '{OUT_PARQUET}' (FORMAT PARQUET)")
    print(f"wrote {OUT_PARQUET} ({OUT_PARQUET.stat().st_size:,} bytes)", flush=True)

    # Tenant cardinality summary
    print("\ntenant cardinality (top 20):", flush=True)
    counts = df.groupby("tenant_id").size().sort_values(ascending=False).head(20)
    for tenant, n in counts.items():
        print(f"  {tenant}: {n} spans", flush=True)

    # Span-kind distribution
    print("\nspan_kind distribution:", flush=True)
    for kind, n in df.groupby("span_kind").size().sort_values(ascending=False).items():
        print(f"  {kind}: {n}", flush=True)


if __name__ == "__main__":
    adapt()
