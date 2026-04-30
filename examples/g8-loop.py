#!/usr/bin/env python3
"""Mews Gate-8 sustained autonomous loop orchestrator.

Iterates infra.loop_lib.run_one_cycle over 3 distinct drift inducers:
  cycle 1: correctness drift  -- silent run-3 engine + wildchat-x10
  cycle 2: cardinality drift  -- china-10x inducer parquet
  cycle 3: schema drift       -- extra-col inducer parquet (baseline updates after regen)

Plus a no-drift baseline check after each cycle to verify the detector
doesn't false-positive between cycles, plus a final baseline at the end.

Retry policy: at most 1 retry per cycle when post-regen still flags drift
(re-seeded from canonical, NOT from the failed regen output). On retry
failure: KILL.

Hard kill at $200 cumulative regen spend.

Usage: python loop.py --run-dir <experiments/gate-8/run-N> [--skip-baseline-checks]
"""
from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from infra.loop_lib import (  # noqa: E402
    ROOT,
    CycleResult,
    initialize_deployed,
    log_line,
    run_one_cycle,
    write_loop_manifest,
)

CANON_ENGINE = ROOT / "experiments/gate-4/stage3-run2-opus/output/engine"
CANON_CPP = ROOT / "experiments/gate-4/stage3-run2-opus/output/engine.cpp"
SILENT_ENGINE = ROOT / "experiments/gate-4/stage3-run3-sonnet-profiler/output/engine"
SILENT_CPP = ROOT / "experiments/gate-4/stage3-run3-sonnet-profiler/output/engine.cpp"

PARQ_X1 = ROOT / "experiments/gate-4/cliff-probe/parquet/wildchat-x1.parquet"
PARQ_X10 = ROOT / "experiments/gate-4/cliff-probe/parquet/wildchat-x10.parquet"
PARQ_X100 = ROOT / "experiments/gate-4/cliff-probe/parquet/wildchat-x100.parquet"

G6_INDUCERS = ROOT / "experiments/gate-6/inducers"
INDUCER_BASELINE_SCHEMA = G6_INDUCERS / "baseline-schema.json"
INDUCER_CHINA_10X = G6_INDUCERS / "wildchat-x1-china-10x.parquet"
INDUCER_EXTRA_X1 = G6_INDUCERS / "wildchat-x1-extra-col.parquet"

G8_INDUCERS = ROOT / "experiments/gate-8/run-1/inducers"
INDUCER_EXTRA_X10 = G8_INDUCERS / "wildchat-x10-extra-col.parquet"
INDUCER_EXTRA_X100 = G8_INDUCERS / "wildchat-x100-extra-col.parquet"

KILL_THRESHOLD_USD = 200.0


def cumulative_cost(results: list[CycleResult]) -> float:
    return sum((r.regen_cost_usd or 0.0) for r in results)


def kill_on_budget(results: list[CycleResult], log_file: Path) -> bool:
    c = cumulative_cost(results)
    if c > KILL_THRESHOLD_USD:
        log_line(f"!!! KILL: cumulative regen cost ${c:.2f} > ${KILL_THRESHOLD_USD:.2f}", log_file)
        return True
    return False


def run_cycle_with_retry(
    *,
    cycle_id: str,
    run_dir: Path,
    log_file: Path,
    results: list[CycleResult],
    cycle_kwargs: dict,
) -> CycleResult:
    """Execute a regen cycle. Retry once from canonical seed if cycle_closed=False."""
    primary_dir = run_dir / cycle_id
    r = run_one_cycle(cycle_id=cycle_id, cycle_dir=primary_dir, log_file=log_file, **cycle_kwargs)
    results.append(r)

    if r.cycle_closed:
        return r

    # Per gate-8-plan.md convergence policy: retry once on any non-closure
    # (artifact-validation failure OR post-regen drift), reseeded from canonical.
    if r.regen_artifact_ok:
        retry_reason = "post-regen still flagged drift"
    else:
        retry_reason = "Stage-3 produced non-validating artifact"
    log_line(f">>> RETRY {cycle_id}: {retry_reason}; reseeding from canonical", log_file)
    retry_id = f"{cycle_id}-retry"
    retry_kwargs = dict(cycle_kwargs)
    retry_kwargs["regen_seed_engine"] = CANON_ENGINE
    retry_kwargs["regen_seed_cpp"] = CANON_CPP
    r2 = run_one_cycle(
        cycle_id=retry_id, cycle_dir=run_dir / retry_id, log_file=log_file, **retry_kwargs,
    )
    results.append(r2)
    if not r2.cycle_closed:
        log_line(f"!!! KILL {cycle_id}: retry also failed to close", log_file)
    return r2


def baseline_check(
    *,
    label: str,
    run_dir: Path,
    log_file: Path,
    results: list[CycleResult],
    deployed: Path,
    trigger_parquet: Path,
    baseline_schema: Path,
    parquet_x1: Path,
    parquet_x10: Path,
    bench_parquet: Path,
) -> CycleResult:
    """No-drift check: same trigger parquet as last cycle, expect MATCH idle."""
    log_line("-" * 78, log_file)
    log_line(f">>> BASELINE CHECK ({label}): no inducer change, expect MATCH", log_file)
    log_line("-" * 78, log_file)
    r = run_one_cycle(
        cycle_id=label,
        cycle_dir=run_dir / label,
        log_file=log_file,
        deployed_ws=deployed,
        regen_seed_engine=CANON_ENGINE,
        regen_seed_cpp=CANON_CPP,
        trigger_parquet=trigger_parquet,
        baseline_schema=baseline_schema,
        parquet_x1=parquet_x1,
        parquet_x10=parquet_x10,
        bench_parquet=bench_parquet,
        expected_pre_drift=False,
    )
    results.append(r)
    if not r.cycle_closed:
        log_line(f"!!! BASELINE FP at {label}: detector flagged drift when none injected", log_file)
    return r


def emit_manifest(run_dir: Path, results: list[CycleResult], status: str, args_dict: dict) -> None:
    write_loop_manifest(
        run_dir / "manifest.json",
        cycles=results,
        extra={
            "orchestrator": "gate-8-v1",
            "status": status,
            "kill_threshold_usd": KILL_THRESHOLD_USD,
            "args": args_dict,
        },
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument(
        "--skip-baseline-checks", action="store_true",
        help="Skip inter-cycle and final baseline checks (still kills on regen-drift).",
    )
    args = p.parse_args()

    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    log_file = run_dir / "run.log"
    log_file.touch()

    deployed = run_dir / "deployed"
    baseline_schema = run_dir / "baseline-schema.json"
    shutil.copy(INDUCER_BASELINE_SCHEMA, baseline_schema)
    log_line(f"baseline_schema initialized: {baseline_schema}", log_file)

    initialize_deployed(deployed, SILENT_ENGINE, SILENT_CPP)
    log_line("deployed/ initialized with silent run-3 engine (cycle-1 starting state)", log_file)

    results: list[CycleResult] = []
    args_dict = {"run_dir": str(run_dir), "skip_baseline_checks": args.skip_baseline_checks}

    # ============================================================
    # CYCLE 1 — Correctness drift (silent engine + x10)
    # ============================================================
    log_line("=" * 78, log_file)
    log_line(">>> CYCLE 1: CORRECTNESS DRIFT (silent run-3 engine + wildchat-x10)", log_file)
    log_line("=" * 78, log_file)
    r1 = run_cycle_with_retry(
        cycle_id="cycle-1-correctness",
        run_dir=run_dir, log_file=log_file, results=results,
        cycle_kwargs=dict(
            deployed_ws=deployed,
            regen_seed_engine=CANON_ENGINE, regen_seed_cpp=CANON_CPP,
            trigger_parquet=PARQ_X10,
            baseline_schema=baseline_schema,
            parquet_x1=PARQ_X1, parquet_x10=PARQ_X10, bench_parquet=PARQ_X100,
            expected_pre_drift=True,
        ),
    )
    if not r1.cycle_closed:
        emit_manifest(run_dir, results, "killed_at_cycle_1", args_dict); return 4
    if kill_on_budget(results, log_file):
        emit_manifest(run_dir, results, "killed_budget_after_cycle_1", args_dict); return 5

    if not args.skip_baseline_checks:
        baseline_check(
            label="baseline-1", run_dir=run_dir, log_file=log_file, results=results,
            deployed=deployed, trigger_parquet=PARQ_X10, baseline_schema=baseline_schema,
            parquet_x1=PARQ_X1, parquet_x10=PARQ_X10, bench_parquet=PARQ_X100,
        )

    # ============================================================
    # CYCLE 2 — Cardinality drift (china-10x inducer)
    # ============================================================
    log_line("=" * 78, log_file)
    log_line(">>> CYCLE 2: CARDINALITY DRIFT (wildchat-x1-china-10x.parquet)", log_file)
    log_line("=" * 78, log_file)
    r2 = run_cycle_with_retry(
        cycle_id="cycle-2-cardinality",
        run_dir=run_dir, log_file=log_file, results=results,
        cycle_kwargs=dict(
            deployed_ws=deployed,
            regen_seed_engine=CANON_ENGINE, regen_seed_cpp=CANON_CPP,
            trigger_parquet=INDUCER_CHINA_10X,
            baseline_schema=baseline_schema,
            # parquet_x1 = inducer (so Stage-3 validate_all checks against the new shape)
            parquet_x1=INDUCER_CHINA_10X, parquet_x10=PARQ_X10, bench_parquet=PARQ_X100,
            expected_pre_drift=True,
        ),
    )
    if not r2.cycle_closed:
        emit_manifest(run_dir, results, "killed_at_cycle_2", args_dict); return 4
    if kill_on_budget(results, log_file):
        emit_manifest(run_dir, results, "killed_budget_after_cycle_2", args_dict); return 5

    if not args.skip_baseline_checks:
        baseline_check(
            label="baseline-2", run_dir=run_dir, log_file=log_file, results=results,
            deployed=deployed, trigger_parquet=INDUCER_CHINA_10X, baseline_schema=baseline_schema,
            parquet_x1=INDUCER_CHINA_10X, parquet_x10=PARQ_X10, bench_parquet=PARQ_X100,
        )

    # ============================================================
    # CYCLE 3 — Schema drift (extra-col inducer; baseline updates after regen)
    # ============================================================
    log_line("=" * 78, log_file)
    log_line(">>> CYCLE 3: SCHEMA DRIFT (wildchat-x1-extra-col.parquet)", log_file)
    log_line("=" * 78, log_file)
    r3 = run_cycle_with_retry(
        cycle_id="cycle-3-schema",
        run_dir=run_dir, log_file=log_file, results=results,
        cycle_kwargs=dict(
            deployed_ws=deployed,
            regen_seed_engine=CANON_ENGINE, regen_seed_cpp=CANON_CPP,
            trigger_parquet=INDUCER_EXTRA_X1,
            baseline_schema=baseline_schema,
            parquet_x1=INDUCER_EXTRA_X1,
            parquet_x10=INDUCER_EXTRA_X10,
            bench_parquet=INDUCER_EXTRA_X100,
            expected_pre_drift=True,
            update_baseline_after_regen=True,
        ),
    )
    if not r3.cycle_closed:
        emit_manifest(run_dir, results, "killed_at_cycle_3", args_dict); return 4
    if kill_on_budget(results, log_file):
        emit_manifest(run_dir, results, "killed_budget_after_cycle_3", args_dict); return 5

    # Final baseline check on the new (extra-col) shape.
    final_label = "baseline-3-final"
    baseline_check(
        label=final_label, run_dir=run_dir, log_file=log_file, results=results,
        deployed=deployed, trigger_parquet=INDUCER_EXTRA_X1, baseline_schema=baseline_schema,
        parquet_x1=INDUCER_EXTRA_X1, parquet_x10=INDUCER_EXTRA_X10, bench_parquet=INDUCER_EXTRA_X100,
    )

    # ============================================================
    # Verdict
    # ============================================================
    regen_cycles_closed = sum(1 for r in results if r.cycle_id.startswith("cycle-") and r.cycle_closed)
    baselines_idle = sum(
        1 for r in results
        if r.cycle_id.startswith("baseline-") and r.cycle_closed and not r.pre_regen_recommended
    )
    final_cost = cumulative_cost(results)

    log_line("=" * 78, log_file)
    log_line(f"G8 summary: {regen_cycles_closed} regen cycles closed, "
             f"{baselines_idle} baselines idle, ${final_cost:.2f} total regen spend",
             log_file)
    log_line("=" * 78, log_file)

    pass_ok = regen_cycles_closed >= 3 and final_cost < KILL_THRESHOLD_USD
    status = "passed" if pass_ok else "incomplete"
    emit_manifest(run_dir, results, status, args_dict)

    if pass_ok:
        print(f"\nG8 PASS: {regen_cycles_closed} cycles, {baselines_idle} baselines idle, ${final_cost:.2f}")
        return 0
    print(f"\nG8 INCOMPLETE: {regen_cycles_closed} cycles closed, ${final_cost:.2f}")
    return 4


if __name__ == "__main__":
    sys.exit(main())
