"""Mews multi-tenant validator (Gate 1+ generalized).

Wraps validate.py to run the engine binary against N tenants and report
combined pass/fail. The engine is invoked once per tenant_id; its
result1.csv is captured per tenant and diffed against DuckDB reference.

Used by the optim loop's exit condition: speedup is only reported as
"done" once all selected tenants pass diff.

Usage:
    # Gate 1 default (synthetic fixture, three named tenants)
    python validate_all.py --workspace .

    # Gate 3+ (real-data parquet, named tenants)
    python validate_all.py --workspace . \\
        --parquet ~/projects/mews/data/adapters/wildchat/output/spans.parquet \\
        --tenants 'China,United States,Russia'
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

# Allow importing validate.py whether run from infra/ or elsewhere.
INFRA_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(INFRA_DIR))
import validate  # type: ignore  # noqa: E402

DEFAULT_TENANTS = ["tenant-01", "tenant-02", "tenant-03"]
DEFAULT_PARQUET = (
    Path.home()
    / "projects/mews/artifacts/openinference_parquet/sf1/spans.parquet"
)


def _safe_filename(s: str) -> str:
    """Sanitize tenant name for use in saved CSV filenames (handles spaces, slashes)."""
    return "".join(c if c.isalnum() or c in ".-_" else "_" for c in s)


def run_one(workspace: Path, tenant_id: str, parquet: Path, rel_tol: float):
    engine = workspace / "engine"
    if not engine.exists():
        return False, f"{tenant_id}: engine binary not at {engine}"
    # Pass parquet to engine so the silent-x1 catch is sound: an engine that
    # ignores argv[2] reads its hardcoded default and produces row counts at
    # the wrong scale; an engine that respects argv[2] reads the requested
    # parquet and matches the DuckDB reference at that scale. Without this
    # pass-through, every "correct" engine fails at x10/x100/etc. because
    # its hardcoded default doesn't match --parquet (false positive).
    proc = subprocess.run(
        [str(engine), tenant_id, str(parquet)],
        cwd=workspace,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if proc.returncode != 0:
        return False, f"{tenant_id}: engine exit={proc.returncode} stderr={proc.stderr[:200]}"
    csv = workspace / "result1.csv"
    if not csv.exists():
        return False, f"{tenant_id}: no result1.csv produced"
    saved = workspace / f"result1.{_safe_filename(tenant_id)}.csv"
    shutil.copy(csv, saved)
    res = validate.diff(parquet, tenant_id, saved, rel_tol=rel_tol)
    return res.ok, f"{tenant_id}: {res.summary}" + (
        "\n  " + "\n  ".join(res.detail_lines) if res.detail_lines else ""
    )


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace", type=Path, required=True,
                   help="dir containing the engine binary")
    p.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    p.add_argument("--tenants", type=str, default=",".join(DEFAULT_TENANTS),
                   help="comma-separated tenant_id values to validate against. "
                        "Default: tenant-01,tenant-02,tenant-03 (Gate-1 synthetic fixture)")
    p.add_argument("--rel-tol", type=float, default=1e-3)
    args = p.parse_args(argv)
    ws = args.workspace.resolve()

    tenants = [t.strip() for t in args.tenants.split(",") if t.strip()]
    if not tenants:
        print("FAIL: no tenants specified", flush=True)
        return 2

    print(f"workspace: {ws}")
    print(f"parquet:   {args.parquet}")
    print(f"tenants:   {tenants}")
    print(f"rel_tol:   {args.rel_tol}")
    print("---")

    all_ok = True
    for t in tenants:
        ok, msg = run_one(ws, t, args.parquet, args.rel_tol)
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {msg}")
        if not ok:
            all_ok = False
    print("---")
    n = len(tenants)
    print("OVERALL: " + (f"PASS (all {n} tenants match DuckDB reference)" if all_ok else "FAIL"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
