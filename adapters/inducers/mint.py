#!/usr/bin/env python3
"""Mint schema- and cardinality-perturbed parquets for drift-detector tests.

Produces the four inducer parquets the autonomous-loop integration tests
and the G8 sustained-loop experiment use as drift events:

  * extra-col       - extra string column injected at index 0
  * rename-tenant   - `tenant_id` column renamed to `tenant`
  * start-int64     - `start_time` cast from timestamp[us, UTC] to int64
                      (strips the TIMESTAMP logical type)
  * china-10x       - `China` tenant rows duplicated 10x (cardinality bump,
                      correctness preserved against DuckDB)

Plus a `baseline-schema.json` snapshot of the source schema, used by the
drift detector as the "synthesis time" reference.

Usage:
    python mint.py --src SPANS.parquet --out-dir OUT_DIR
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


def _schema_to_dict(schema: pa.Schema) -> dict:
    return {
        "fields": [
            {"name": f.name, "type": str(f.type), "nullable": bool(f.nullable)}
            for f in schema
        ]
    }


def _mint_extra_col(src: pa.Table, out: Path) -> None:
    extra = pa.array(["inducer-marker"] * src.num_rows, type=pa.string())
    table = src.add_column(0, "inducer_extra_col", extra)
    pq.write_table(table, out)
    print(f"extra-col written: {out}")


def _mint_rename_tenant(src: pa.Table, out: Path) -> None:
    new_names = ["tenant" if n == "tenant_id" else n for n in src.column_names]
    table = src.rename_columns(new_names)
    pq.write_table(table, out)
    print(f"rename-tenant written: {out}")


def _mint_start_int64(src: pa.Table, out: Path) -> None:
    st_int64 = pc.cast(src.column("start_time"), pa.int64())
    fields, arrays = [], []
    for i, f in enumerate(src.schema):
        if f.name == "start_time":
            fields.append(pa.field("start_time", pa.int64(), nullable=f.nullable))
            arrays.append(st_int64)
        else:
            fields.append(f)
            arrays.append(src.column(i))
    table = pa.Table.from_arrays(arrays, schema=pa.schema(fields))
    pq.write_table(table, out)
    print(f"start-int64 written: {out}")


def _mint_china_10x(src: pa.Table, out: Path) -> None:
    if "tenant_id" not in src.column_names:
        print("china-10x skipped: source has no tenant_id column", file=sys.stderr)
        return
    mask = pc.equal(src.column("tenant_id"), "China")
    china = src.filter(mask)
    if china.num_rows == 0:
        print("china-10x skipped: no China rows in source", file=sys.stderr)
        return
    non_china = src.filter(pc.invert(mask))
    extra_china = pa.concat_tables([china] * 9)
    table = pa.concat_tables([non_china, china, extra_china])
    pq.write_table(table, out)
    print(
        f"china-10x written: {out} "
        f"(rows {table.num_rows:,}; china {china.num_rows:,} -> {china.num_rows * 10:,})"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path,
                    help="Source spans.parquet from one of the adapters")
    ap.add_argument("--out-dir", required=True, type=Path,
                    help="Directory to write inducer parquets + baseline-schema.json")
    ap.add_argument("--prefix", default="spans",
                    help="Filename prefix for the inducer parquets (default: spans)")
    args = ap.parse_args()

    if not args.src.exists():
        print(f"source not found: {args.src}", file=sys.stderr)
        return 2

    args.out_dir.mkdir(parents=True, exist_ok=True)
    src = pq.read_table(args.src)
    print(f"source: {args.src}  rows={src.num_rows:,}")

    _mint_extra_col(src, args.out_dir / f"{args.prefix}-extra-col.parquet")
    _mint_rename_tenant(src, args.out_dir / f"{args.prefix}-rename-tenant.parquet")
    _mint_start_int64(src, args.out_dir / f"{args.prefix}-start-int64.parquet")
    _mint_china_10x(src, args.out_dir / f"{args.prefix}-china-10x.parquet")

    baseline = args.out_dir / "baseline-schema.json"
    baseline.write_text(json.dumps(_schema_to_dict(src.schema), indent=2))
    print(f"baseline schema written: {baseline}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
