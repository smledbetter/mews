"""HELM (Stanford CRFM) -> OpenInference adapter.

Mapping (Option 2):
- 1 scenario_state.json file = 1 (scenario, model) run
- Each request_state in the file -> 1 trace, with:
    AGENT root span (parent_id=NULL, span_kind=AGENT)
    LLM child span (parent_id=AGENT, span_kind=LLM)
- tenant_id = scenario name (e.g. "legalbench", "gsm", "mmlu")

Why HELM: chat logs (WildChat) and agent trajectories (SWE-rebench) are end-user
shapes; HELM is structured offline-eval data, the third Phoenix-user shape we
care about. Bonus: result.request_time is REAL wall-clock latency, not synthetic.

Data source:
- Public GCS bucket: gs://crfm-helm-public/lite/benchmark_output/runs/v1.13.0/
- Accessible via plain HTTPS (https://storage.googleapis.com/...) without auth
- ~112 run dirs * ~250KB scenario_state.json each = ~30MB total

Run:
    uv run --with pyarrow --with requests python adapt.py
"""
from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

OUT_DIR = Path(__file__).parent / "output"
OUT_PARQUET = OUT_DIR / "spans.parquet"
GCS_BUCKET = "crfm-helm-public"
GCS_PREFIX = "lite/benchmark_output/runs/v1.13.0/"
BATCH_REQUESTS = 500     # spans flushed to parquet per write_table call


def _gcs_list(bucket: str, prefix: str) -> list[str]:
    """List all run directory prefixes under prefix/. Public GCS, no auth."""
    runs = []
    page_token = None
    while True:
        url = (f"https://storage.googleapis.com/storage/v1/b/{bucket}/o"
               f"?prefix={urllib.parse.quote(prefix, safe='')}"
               f"&delimiter=/&maxResults=1000")
        if page_token:
            url += f"&pageToken={urllib.parse.quote(page_token)}"
        with urllib.request.urlopen(url) as r:
            d = json.load(r)
        runs.extend(d.get('prefixes', []))
        page_token = d.get('nextPageToken')
        if not page_token:
            break
    return runs


def _gcs_get_json(bucket: str, path: str) -> dict | None:
    """Fetch a JSON object from public GCS by HTTPS. Returns None on 404."""
    url = f"https://storage.googleapis.com/{bucket}/{path}"
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise


def _parse_run_name(run_dir: str) -> tuple[str, str]:
    """Extract (scenario, model) from a run dir like
    'lite/benchmark_output/runs/v1.13.0/legalbench:subset=abercrombie,model=amazon_nova-lite-v1:0,stop=none/'
    """
    name = run_dir.rstrip('/').split('/')[-1]
    m = re.match(r'([^:,]+)', name)
    scenario = m.group(1) if m else 'unknown'
    mm = re.search(r'model=([^,/]+)', name)
    model = mm.group(1) if mm else 'unknown'
    return scenario, model


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


def _request_state_to_spans(rs: dict, scenario: str, model: str) -> list[dict]:
    """Map one HELM request_state to AGENT + LLM spans."""
    request_time = rs.get('result', {}).get('request_time')
    if request_time is None:
        return []
    request_datetime = rs.get('result', {}).get('request_datetime')
    if request_datetime is None:
        return []

    t0 = datetime.fromtimestamp(request_datetime, tz=timezone.utc)
    t1 = t0 + timedelta(seconds=float(request_time))

    instance = rs.get('instance', {}) or {}
    request  = rs.get('request', {})  or {}
    result   = rs.get('result', {})   or {}

    instance_id = instance.get('id', uuid.uuid4().hex[:8])
    success = result.get('success', True)
    status = "OK" if success else "ERROR"

    # Token estimation (HELM doesn't carry token counts in scenario_state.json)
    prompt = request.get('prompt') or ''
    completion = ''
    completions = result.get('completions') or []
    if completions and isinstance(completions[0], dict):
        completion = completions[0].get('text') or ''
    in_tokens = max(0, len(prompt) // 4)
    out_tokens = max(0, len(completion) // 4)

    trace_id = uuid.uuid4().hex
    agent_id = uuid.uuid4().hex
    llm_id   = uuid.uuid4().hex

    agent_span = {
        "trace_id": trace_id,
        "span_id": agent_id,
        "parent_id": None,
        "name": f"helm.{scenario}.{instance_id}",
        "start_time": t0,
        "end_time": t1,
        "status_code": status,
        "status_message": None,
        "span_kind": "AGENT",
        "llm_token_count_prompt": None,
        "llm_token_count_completion": None,
        "attributes": json.dumps({
            "agent": {"scenario": scenario, "instance_id": instance_id, "split": instance.get('split')},
        }),
        "tenant_id": scenario,
    }

    llm_span = {
        "trace_id": trace_id,
        "span_id": llm_id,
        "parent_id": agent_id,
        "name": f"llm.completion.{model}",
        "start_time": t0,
        "end_time": t1,
        "status_code": status,
        "status_message": None,
        "span_kind": "LLM",
        "llm_token_count_prompt": int(in_tokens),
        "llm_token_count_completion": int(out_tokens),
        "attributes": json.dumps({
            "llm": {
                "model_name": model,
                "input_tokens":  in_tokens,
                "output_tokens": out_tokens,
                "temperature":   request.get('temperature'),
                "max_tokens":    request.get('max_tokens'),
            },
        }),
        "tenant_id": scenario,
    }
    return [agent_span, llm_span]


def adapt():
    import pyarrow.parquet as pq

    schema = _build_schema()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"listing GCS: gs://{GCS_BUCKET}/{GCS_PREFIX}", flush=True)
    run_dirs = _gcs_list(GCS_BUCKET, GCS_PREFIX)
    print(f"found {len(run_dirs)} run dirs", flush=True)

    n_runs_ok = 0
    n_runs_skipped = 0
    n_spans_total = 0
    tenant_counts: dict[str, int] = {}
    kind_counts: dict[str, int] = {"AGENT": 0, "LLM": 0}
    batch_rows: list[dict] = []

    with pq.ParquetWriter(str(OUT_PARQUET), schema, compression="snappy") as writer:
        for i, run_dir in enumerate(run_dirs):
            scenario, model = _parse_run_name(run_dir)
            scenario_state_path = run_dir + "scenario_state.json"
            try:
                doc = _gcs_get_json(GCS_BUCKET, scenario_state_path)
            except Exception as e:
                print(f"  [{i+1}/{len(run_dirs)}] {scenario}/{model}: HTTP fail ({e})", flush=True)
                n_runs_skipped += 1
                continue
            if not doc or not doc.get('request_states'):
                print(f"  [{i+1}/{len(run_dirs)}] {scenario}/{model}: empty", flush=True)
                n_runs_skipped += 1
                continue

            run_model = doc.get('adapter_spec', {}).get('model') or model
            n_in_run = 0
            for rs in doc['request_states']:
                spans = _request_state_to_spans(rs, scenario, run_model)
                if not spans:
                    continue
                batch_rows.extend(spans)
                n_in_run += len(spans)
                tenant_counts[scenario] = tenant_counts.get(scenario, 0) + len(spans)
                for s in spans:
                    kind_counts[s["span_kind"]] = kind_counts.get(s["span_kind"], 0) + 1

                if len(batch_rows) >= BATCH_REQUESTS:
                    n_spans_total += _flush(batch_rows, writer, schema)
                    batch_rows.clear()

            n_runs_ok += 1
            print(f"  [{i+1}/{len(run_dirs)}] {scenario}/{run_model}: {n_in_run} spans  "
                  f"(running total: {n_spans_total + len(batch_rows)})", flush=True)

        # Flush tail
        if batch_rows:
            n_spans_total += _flush(batch_rows, writer, schema)
            batch_rows.clear()

    print(f"\nadapted {n_runs_ok} runs ({n_runs_skipped} skipped) -> {n_spans_total} spans", flush=True)
    print(f"wrote {OUT_PARQUET} ({OUT_PARQUET.stat().st_size:,} bytes)", flush=True)

    print("\ntenant cardinality (by scenario):", flush=True)
    for tenant, n in sorted(tenant_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {tenant}: {n} spans", flush=True)
    print("\nspan_kind distribution:", flush=True)
    for kind, n in sorted(kind_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {kind}: {n}", flush=True)


if __name__ == "__main__":
    adapt()
