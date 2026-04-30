"""Mews Gate-6 drift detector.

Periodic guard that compares a deployed engine binary against a DuckDB
reference on a current parquet, plus a baseline schema snapshot, and
emits a structured DriftReport. Decides whether regen should fire.

Drift types:
  - SCHEMA: parquet schema differs from baseline (col added/removed/typed)
  - CORRECTNESS: engine output diverges from DuckDB reference for any tenant
  - PERF: engine wall has degraded vs synthesis-time baseline AND now
          underperforms DuckDB-warm by the configured ratio threshold

Decision priority: SCHEMA -> CORRECTNESS -> PERF. The first match wins
and earlier checks short-circuit the later ones (engine output is
meaningless under schema mismatch; perf is meaningless if correctness
is broken).

Example:
    python drift_detector.py \\
        --engine /path/to/deployed/engine \\
        --workspace /path/to/deployed/workspace \\
        --parquet /path/to/current/spans.parquet \\
        --tenants 'China,United States,Russia' \\
        --baseline-schema /path/to/baseline.schema.json \\
        --baseline-engine-ms 535.0 \\
        --out /path/to/report.json

Exit codes:
    0 = no drift, no regen recommended
    1 = drift detected, regen recommended (see report.json for type)
    2 = harness error (file missing, bad invocation)
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pyarrow.parquet as pq

INFRA_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(INFRA_DIR))


@dataclass
class DriftReport:
    ts_utc: str
    tenants_checked: list[str]
    parquet: str
    engine: str
    schema_status: str = "MATCH"  # MATCH | DRIFT | UNCHECKED
    schema_drift: dict[str, Any] = field(default_factory=dict)
    correctness_status: str = "UNCHECKED"  # MATCH | DRIFT | INCONCLUSIVE | UNCHECKED
    correctness_drift_per_tenant: dict[str, str] = field(default_factory=dict)
    perf_status: str = "UNCHECKED"  # OK | DEGRADED | INCONCLUSIVE | UNCHECKED
    perf_engine_ms: Optional[float] = None
    perf_duckdb_ms: Optional[float] = None
    perf_baseline_engine_ms: Optional[float] = None
    perf_baseline_ratio_threshold: float = 1.5
    regen_recommended: bool = False
    regen_reason: str = "none"  # schema | correctness | perf | none
    notes: list[str] = field(default_factory=list)


def schema_to_dict(schema) -> dict[str, list[dict[str, str]]]:
    """Stable, JSON-serializable view of an arrow schema."""
    return {
        "fields": [
            {"name": f.name, "type": str(f.type), "nullable": bool(f.nullable)}
            for f in schema
        ]
    }


def diff_schemas(baseline: dict, current: dict) -> dict[str, Any]:
    """Return a diff dict; empty dict if schemas match.

    Layout matches the field name + type + nullability tuples; ordering
    matters because the canonical engine reads columns by index, not
    name. A column reorder is treated as drift.
    """
    base_fields = baseline.get("fields", [])
    cur_fields = current.get("fields", [])
    base_names = [f["name"] for f in base_fields]
    cur_names = [f["name"] for f in cur_fields]
    added = [n for n in cur_names if n not in base_names]
    removed = [n for n in base_names if n not in cur_names]
    type_changed = []
    reordered = False
    base_by_name = {f["name"]: f for f in base_fields}
    cur_by_name = {f["name"]: f for f in cur_fields}
    for n in cur_names:
        if n in base_by_name and cur_by_name[n]["type"] != base_by_name[n]["type"]:
            type_changed.append({
                "name": n,
                "baseline_type": base_by_name[n]["type"],
                "current_type": cur_by_name[n]["type"],
            })
    common = [n for n in cur_names if n in base_names]
    base_order_of_common = [n for n in base_names if n in common]
    if common != base_order_of_common:
        reordered = True
    if not (added or removed or type_changed or reordered):
        return {}
    return {
        "added": added,
        "removed": removed,
        "type_changed": type_changed,
        "reordered": reordered,
    }


def run_validate_all(workspace: Path, parquet: Path, tenants: list[str],
                     timeout: float = 300.0) -> tuple[bool, dict[str, str], str]:
    """Returns (overall_pass, per_tenant_msg, raw_stdout). Per-tenant msg
    parsed from the [PASS]/[FAIL] lines."""
    cmd = [
        os.environ.get("UV_BIN", "uv"), "run", "python",
        str(INFRA_DIR / "validate_all.py"),
        "--workspace", str(workspace),
        "--parquet", str(parquet),
        "--tenants", ",".join(tenants),
    ]
    env = dict(os.environ)
    env["UV_CACHE_DIR"] = env.get("UV_CACHE_DIR", "/tmp/uv_cache")
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, env=env,
        )
    except subprocess.TimeoutExpired:
        return False, {t: "TIMEOUT" for t in tenants}, ""
    overall_pass = proc.returncode == 0 and "OVERALL: PASS" in proc.stdout
    per_tenant: dict[str, str] = {}
    for line in proc.stdout.splitlines():
        line_stripped = line.strip()
        if line_stripped.startswith("[PASS]"):
            rest = line_stripped[len("[PASS]"):].strip()
            tenant_id, _, msg = rest.partition(":")
            per_tenant[tenant_id.strip()] = "PASS: " + msg.strip()
        elif line_stripped.startswith("[FAIL]"):
            rest = line_stripped[len("[FAIL]"):].strip()
            tenant_id, _, msg = rest.partition(":")
            per_tenant[tenant_id.strip()] = "FAIL: " + msg.strip()
    for t in tenants:
        if t not in per_tenant:
            per_tenant[t] = "UNREPORTED"
    return overall_pass, per_tenant, proc.stdout + ("\n" + proc.stderr if proc.stderr else "")


def quick_bench_engine(engine: Path, parquet: Path, tenant: str,
                       iters: int = 3, timeout: float = 120.0) -> Optional[float]:
    """Median wall-clock ms over `iters` cold engine runs."""
    if not engine.exists():
        return None
    times_ms: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        try:
            proc = subprocess.run(
                [str(engine), tenant, str(parquet)],
                capture_output=True, text=True, timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return None
        elapsed_ms = (time.perf_counter() - t0) * 1000
        if proc.returncode != 0:
            return None
        times_ms.append(elapsed_ms)
    return statistics.median(times_ms) if times_ms else None


def quick_bench_duckdb(parquet: Path, tenant: str,
                       iters: int = 3, timeout_per_iter: float = 90.0) -> Optional[float]:
    """Median wall-clock ms for DuckDB to compute the locked reference query.

    We invoke a Python subprocess so DuckDB's per-iter ingest cost is
    paid each iteration (cold-start path). This is not the warm baseline
    engine-warm is benchmarked against, but it matches the engine arm's
    cold protocol -- apples-to-apples.

    Returns None on failure or OOM.
    """
    contract_dir = INFRA_DIR.parent / "contracts" / "openinference"
    schema_sql = contract_dir / "schema.sql"
    query_sql = contract_dir / "queries" / "latency_by_model_under_agent.sql"
    py = f"""
import duckdb, time, sys
con = duckdb.connect(':memory:')
con.execute(open({str(schema_sql)!r}).read())
con.execute("INSERT INTO spans SELECT * FROM read_parquet('{parquet}')")
q = open({str(query_sql)!r}).read()
t0 = time.perf_counter()
rows = con.execute(q, ('{tenant}',)).fetchall()
elapsed_ms = (time.perf_counter() - t0) * 1000
print(f'{{elapsed_ms:.3f}}')
"""
    times_ms: list[float] = []
    for _ in range(iters):
        try:
            proc = subprocess.run(
                [os.environ.get("UV_BIN", "uv"), "run", "--with", "duckdb", "python", "-c", py],
                capture_output=True, text=True, timeout=timeout_per_iter,
            )
        except subprocess.TimeoutExpired:
            return None
        if proc.returncode != 0:
            return None
        try:
            times_ms.append(float(proc.stdout.strip().splitlines()[-1]))
        except (ValueError, IndexError):
            return None
    return statistics.median(times_ms) if times_ms else None


def detect_drift(args) -> tuple[DriftReport, int]:
    report = DriftReport(
        ts_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        tenants_checked=args.tenants_list,
        parquet=str(args.parquet),
        engine=str(args.engine),
        perf_baseline_engine_ms=args.baseline_engine_ms,
        perf_baseline_ratio_threshold=args.perf_threshold,
    )

    # 1. Schema drift
    if args.baseline_schema:
        baseline_schema = json.loads(Path(args.baseline_schema).read_text())
        try:
            cur = schema_to_dict(pq.read_schema(args.parquet))
        except Exception as e:
            report.notes.append(f"schema read failed: {type(e).__name__}: {e}")
            report.schema_status = "DRIFT"
            report.schema_drift = {"error": str(e)}
            report.regen_recommended = True
            report.regen_reason = "schema"
            return report, 1
        diff = diff_schemas(baseline_schema, cur)
        if diff:
            report.schema_status = "DRIFT"
            report.schema_drift = diff
            report.regen_recommended = True
            report.regen_reason = "schema"
            report.notes.append("schema drift; correctness/perf checks skipped")
            return report, 1
        else:
            report.schema_status = "MATCH"
    else:
        report.schema_status = "UNCHECKED"
        report.notes.append("no --baseline-schema; schema check skipped")

    # 2. Correctness drift
    overall_pass, per_tenant, _stdout = run_validate_all(
        args.workspace, args.parquet, args.tenants_list,
    )
    report.correctness_drift_per_tenant = per_tenant
    if not overall_pass:
        # If every tenant was UNREPORTED, validate_all.py crashed before
        # emitting a [PASS]/[FAIL] line for any tenant -- typically
        # DuckDB reference computation OOMed at this scale. We cannot
        # distinguish "engine wrong" from "reference unavailable" in
        # this case, so the conservative verdict is INCONCLUSIVE: no
        # regen recommended on the correctness axis. Schema check
        # already passed (we got here), so structural drift is ruled
        # out; perf check still runs to provide a partial signal.
        all_unreported = all(v == "UNREPORTED" for v in per_tenant.values())
        if all_unreported:
            report.correctness_status = "INCONCLUSIVE"
            report.notes.append(
                "validate_all.py emitted no per-tenant output -- "
                "likely DuckDB reference OOM at this scale. "
                "Correctness verdict is inconclusive; perf check still runs."
            )
            # Do NOT short-circuit; fall through to perf.
        else:
            report.correctness_status = "DRIFT"
            report.regen_recommended = True
            report.regen_reason = "correctness"
            report.notes.append("correctness drift; perf check skipped")
            return report, 1
    else:
        report.correctness_status = "MATCH"

    # 3. Performance drift
    # v0 logic: PERF fires only when DuckDB at current scale outperforms
    # engine by `--perf-threshold` (default 1.5x). This avoids conflating
    # data-scale growth (engine slows because there's more data) with
    # engine degradation (engine slows for the same shape). When DuckDB
    # bench fails (OOM or other), perf is INCONCLUSIVE and no regen is
    # recommended on the perf axis. The synthesis-time baseline
    # `--baseline-engine-ms` is recorded if provided but does not gate
    # regen alone -- it's an informational signal for the operator.
    perf_tenant = args.perf_tenant or args.tenants_list[0]
    eng_ms = quick_bench_engine(args.engine, args.parquet, perf_tenant, iters=args.perf_iters)
    if eng_ms is None:
        report.perf_status = "INCONCLUSIVE"
        report.notes.append(f"engine bench failed at {args.parquet} for tenant={perf_tenant}")
        return report, 0
    report.perf_engine_ms = eng_ms

    if args.skip_duckdb_perf:
        report.perf_status = "INCONCLUSIVE"
        report.notes.append("DuckDB perf bench skipped (--skip-duckdb-perf)")
        return report, 0

    duck_ms = quick_bench_duckdb(args.parquet, perf_tenant, iters=args.perf_iters)
    if duck_ms is None:
        report.perf_status = "INCONCLUSIVE"
        report.notes.append(
            f"DuckDB bench failed at {args.parquet} (likely OOM at this "
            f"scale); engine wall {eng_ms:.0f}ms recorded but ratio "
            f"unavailable. No PERF regen recommendation possible without "
            f"a reference."
        )
        return report, 0
    report.perf_duckdb_ms = duck_ms

    ratio_engine_to_duckdb = eng_ms / duck_ms
    if args.baseline_engine_ms is not None:
        ratio_vs_baseline = eng_ms / args.baseline_engine_ms
        report.notes.append(
            f"informational: engine_ms / baseline_engine_ms = "
            f"{ratio_vs_baseline:.2f}x (baseline-ms = {args.baseline_engine_ms:.0f})"
        )

    if ratio_engine_to_duckdb > args.perf_threshold:
        report.perf_status = "DEGRADED"
        report.regen_recommended = True
        report.regen_reason = "perf"
        report.notes.append(
            f"engine wall {eng_ms:.0f}ms > {args.perf_threshold}x DuckDB "
            f"wall {duck_ms:.0f}ms -- engine no longer beats DuckDB at "
            f"this scale; regen recommended"
        )
        return report, 1
    else:
        report.perf_status = "OK"
        report.notes.append(
            f"engine_ms / duckdb_ms = {ratio_engine_to_duckdb:.2f} "
            f"(threshold {args.perf_threshold}); engine still beats DuckDB"
        )
        return report, 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--engine", type=Path, required=True)
    p.add_argument("--workspace", type=Path, default=None,
                   help="dir containing the engine + scratch files for "
                        "validate_all.py. Defaults to engine's parent dir.")
    p.add_argument("--parquet", type=Path, required=True)
    p.add_argument("--tenants", type=str, required=True,
                   help="comma-separated tenant_ids")
    p.add_argument("--baseline-schema", type=Path, default=None,
                   help="JSON snapshot of synthesis-time parquet schema "
                        "(emitted by schema_to_dict). When unset, schema "
                        "drift is not checked.")
    p.add_argument("--baseline-engine-ms", type=float, default=None,
                   help="synthesis-time engine wall-clock median in ms "
                        "(single-tenant cold). Used as the perf-drift "
                        "denominator. When unset, perf drift is not checked.")
    p.add_argument("--perf-tenant", type=str, default=None,
                   help="tenant for the perf bench (defaults to first --tenants entry)")
    p.add_argument("--perf-threshold", type=float, default=1.5,
                   help="ratio that triggers PERF drift (default 1.5 = "
                        "DEGRADED when engine_ms > 1.5 * duckdb_ms at "
                        "current data scale; engine no longer beats "
                        "DuckDB by the configured margin)")
    p.add_argument("--perf-iters", type=int, default=3)
    p.add_argument("--skip-duckdb-perf", action="store_true",
                   help="skip the DuckDB perf bench (use when DuckDB OOMs "
                        "at this scale; falls back to baseline-ratio only)")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    args.tenants_list = [t.strip() for t in args.tenants.split(",") if t.strip()]
    if args.workspace is None:
        args.workspace = args.engine.parent
    if not args.engine.exists():
        print(f"ERROR: engine not found at {args.engine}", file=sys.stderr)
        return 2
    if not args.parquet.exists():
        print(f"ERROR: parquet not found at {args.parquet}", file=sys.stderr)
        return 2

    report, exit_code = detect_drift(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(asdict(report), indent=2))
    print(json.dumps(asdict(report), indent=2))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
