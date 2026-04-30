#!/usr/bin/env python3
"""Mews Gate-7 first autonomous regen loop (v2 — uses infra/loop_lib).

Single-cycle orchestrator. Initializes deployed/ with the silent run-3
engine, runs one detect -> regen -> deploy -> re-detect cycle from the
canonical seed, and writes a manifest.

For G8 (sustained loop), iterate `run_one_cycle` over multiple drift
inducers using the same loop_lib module.

Usage:
    python loop.py --run-dir <experiments/gate-7/run-N>
                   [--smoke-no-drift]   # use post-regen engine in deployed/, expect no-drift idle path
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from infra.loop_lib import (  # noqa: E402
    ROOT,
    CycleResult,
    initialize_deployed,
    run_one_cycle,
    write_loop_manifest,
)

# Canonical artifact references.
CANON_ENGINE = ROOT / "experiments/gate-4/stage3-run2-opus/output/engine"
CANON_CPP = ROOT / "experiments/gate-4/stage3-run2-opus/output/engine.cpp"
SILENT_ENGINE = ROOT / "experiments/gate-4/stage3-run3-sonnet-profiler/output/engine"
SILENT_CPP = ROOT / "experiments/gate-4/stage3-run3-sonnet-profiler/output/engine.cpp"

PARQ_X1 = ROOT / "experiments/gate-4/cliff-probe/parquet/wildchat-x1.parquet"
PARQ_X10 = ROOT / "experiments/gate-4/cliff-probe/parquet/wildchat-x10.parquet"
PARQ_X100 = ROOT / "experiments/gate-4/cliff-probe/parquet/wildchat-x100.parquet"
BASELINE_SCHEMA = ROOT / "experiments/gate-6/inducers/baseline-schema.json"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--smoke-no-drift", action="store_true",
                   help="Skip the silent-engine seeding; assume deployed/ is "
                        "already populated with a known-good engine. Cycle "
                        "should pre-detect MATCH and exit cleanly with "
                        "cycle_closed=True (idle path).")
    args = p.parse_args()
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    log_file = run_dir / "run.log"
    log_file.touch()

    deployed = run_dir / "deployed"
    cycle_dir = run_dir / "cycle-1"

    if args.smoke_no_drift:
        # Smoke path: deployed/ already has a known-good engine; we expect
        # no drift; the cycle should idle out without firing regen.
        if not (deployed / "engine").exists():
            print(f"--smoke-no-drift requires {deployed}/engine to already exist", file=sys.stderr)
            return 2
        result = run_one_cycle(
            cycle_id="cycle-1-smoke-nodrift",
            deployed_ws=deployed,
            regen_seed_engine=CANON_ENGINE,
            regen_seed_cpp=CANON_CPP,
            trigger_parquet=PARQ_X10,
            baseline_schema=BASELINE_SCHEMA,
            cycle_dir=cycle_dir,
            log_file=log_file,
            parquet_x1=PARQ_X1,
            parquet_x10=PARQ_X10,
            bench_parquet=PARQ_X100,
            expected_pre_drift=False,
        )
    else:
        initialize_deployed(deployed, SILENT_ENGINE, SILENT_CPP)
        result = run_one_cycle(
            cycle_id="cycle-1",
            deployed_ws=deployed,
            regen_seed_engine=CANON_ENGINE,
            regen_seed_cpp=CANON_CPP,
            trigger_parquet=PARQ_X10,
            baseline_schema=BASELINE_SCHEMA,
            cycle_dir=cycle_dir,
            log_file=log_file,
            parquet_x1=PARQ_X1,
            parquet_x10=PARQ_X10,
            bench_parquet=PARQ_X100,
            expected_pre_drift=True,
        )

    write_loop_manifest(
        run_dir / "manifest.json",
        cycles=[result],
        extra={"orchestrator": "gate-7-v2", "smoke_no_drift": args.smoke_no_drift},
    )

    if result.cycle_closed:
        print(f"\nCYCLE CLOSED: {result.cycle_id}")
        return 0
    print(f"\nCYCLE NOT CLOSED: abort_reason={result.abort_reason}")
    return 4


if __name__ == "__main__":
    sys.exit(main())
