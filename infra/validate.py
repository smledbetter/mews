"""Mews Gate-1 differential validator.

Runs the locked OpenInference reference query against (a) DuckDB on the
synthetic parquet (the "reference"), and (b) a synthesized engine binary
producing a CSV result, and reports whether they match within a numeric
tolerance.

Used by runner.py between synthesis stages and as the final Gate-1 PASS check.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import duckdb

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_DIR = PROJECT_ROOT / "contracts" / "openinference"
SCHEMA_SQL = CONTRACT_DIR / "schema.sql"
REFERENCE_QUERY_SQL = CONTRACT_DIR / "queries" / "latency_by_model_under_agent.sql"
DEFAULT_PARQUET = PROJECT_ROOT / "artifacts" / "openinference_parquet" / "sf1" / "spans.parquet"


@dataclass
class ValidationResult:
    ok: bool
    n_rows_ref: int
    n_rows_eng: int
    summary: str
    detail_lines: list[str]


def _duckdb_reference(parquet_path: Path, tenant_id: str) -> list[tuple]:
    con = duckdb.connect(":memory:")
    con.execute(SCHEMA_SQL.read_text())
    con.execute(f"INSERT INTO spans SELECT * FROM read_parquet('{parquet_path}')")
    rows = con.execute(REFERENCE_QUERY_SQL.read_text(), (tenant_id,)).fetchall()
    return rows


def _read_engine_csv(csv_path: Path) -> list[tuple]:
    with csv_path.open() as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [tuple(r) for r in reader]
    return rows


def _coerce_row(r: tuple, column_types: list[str]) -> tuple:
    out = []
    for v, t in zip(r, column_types):
        if v is None or (isinstance(v, str) and v == ""):
            out.append(None)
        elif t in ("int", "bigint"):
            out.append(int(v))
        elif t in ("float", "double"):
            out.append(float(v))
        else:
            out.append(str(v))
    return tuple(out)


def diff(
    parquet_path: Path,
    tenant_id: str,
    engine_csv_path: Path,
    rel_tol: float = 1e-3,
) -> ValidationResult:
    """Compute reference vs engine result diff. Both should have columns
    [model_name (str), n (int), p50_ms (float), p95_ms (float)].
    """
    ref = _duckdb_reference(parquet_path, tenant_id)
    if not engine_csv_path.exists():
        return ValidationResult(
            ok=False, n_rows_ref=len(ref), n_rows_eng=0,
            summary=f"engine output not found at {engine_csv_path}",
            detail_lines=[],
        )

    eng = _read_engine_csv(engine_csv_path)
    types = ["str", "int", "float", "float"]
    eng_typed = [_coerce_row(r, types) for r in eng]

    ref_by_model = {r[0]: r for r in ref}
    eng_by_model = {r[0]: r for r in eng_typed}

    detail: list[str] = []

    only_ref = set(ref_by_model) - set(eng_by_model)
    only_eng = set(eng_by_model) - set(ref_by_model)
    if only_ref:
        detail.append(f"models only in reference: {sorted(only_ref)}")
    if only_eng:
        detail.append(f"models only in engine: {sorted(only_eng)}")

    mismatches = 0
    for model in sorted(set(ref_by_model) & set(eng_by_model)):
        r = ref_by_model[model]
        e = eng_by_model[model]
        # column 1: count, exact
        if int(r[1]) != int(e[1]):
            detail.append(f"  {model}: n ref={r[1]} eng={e[1]}")
            mismatches += 1
            continue
        # columns 2,3: p50_ms, p95_ms; relative tolerance
        for idx, label in [(2, "p50_ms"), (3, "p95_ms")]:
            rv, ev = float(r[idx]), float(e[idx])
            denom = max(abs(rv), 1e-9)
            if abs(rv - ev) / denom > rel_tol:
                detail.append(f"  {model}: {label} ref={rv:.4f} eng={ev:.4f} (rel diff {abs(rv-ev)/denom:.4e})")
                mismatches += 1

    ok = (not only_ref) and (not only_eng) and (mismatches == 0)
    summary = (
        f"OK: {len(ref)} rows match within rel_tol={rel_tol}"
        if ok
        else f"DIFF: {len(only_ref)} only-ref, {len(only_eng)} only-eng, {mismatches} value mismatches"
    )
    return ValidationResult(
        ok=ok, n_rows_ref=len(ref), n_rows_eng=len(eng_typed),
        summary=summary, detail_lines=detail,
    )


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    p.add_argument("--tenant-id", type=str, required=True)
    p.add_argument("--engine-csv", type=Path, required=True)
    p.add_argument("--rel-tol", type=float, default=1e-3)
    args = p.parse_args(argv)

    res = diff(args.parquet, args.tenant_id, args.engine_csv, rel_tol=args.rel_tol)
    print(f"reference rows: {res.n_rows_ref}")
    print(f"engine rows:    {res.n_rows_eng}")
    print(res.summary)
    for line in res.detail_lines:
        print(line)
    return 0 if res.ok else 1


if __name__ == "__main__":
    sys.exit(main())
