#!/usr/bin/env python3
"""Replicate a parquet N times with unique IDs per copy.

Used to mint the `xN` benchmark parquets (`x10`, `x100`, `x500`) the
autonomous-loop benches and Stage-3 differential validation run against.

Why unique IDs per copy: the locked OpenInference reference query has a
self-join `s.parent_id = p.span_id AND s.tenant_id = p.tenant_id` filtered
to `s.span_kind='LLM' AND p.span_kind='AGENT'`. If `span_id` and
`parent_id` are duplicated across replicas, the join cross-products across
copies and produces a Cartesian explosion of intermediate rows that is
not representative of real data growth — the cliff a bench would find
would be a join-blowup cliff, not a data-volume cliff.

To produce a realistic linear scaling: append `_k{idx}` suffix to
`span_id`, `parent_id` (where non-null), and `trace_id` per copy. This
preserves within-copy join structure (copy K's LLMs match copy K's
AGENTs) and keeps cross-copy span_id collisions impossible. `tenant_id`
is NOT suffixed — the per-tenant query path is what we want to scale.

Usage:
    python replicate.py --in INPUT.parquet --out OUTPUT.parquet --factor N

Strategy: stream-read input via ParquetFile.iter_batches; for each batch,
emit N modified copies via ParquetWriter. RSS-bounded.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


_ID_COLS = ("span_id", "parent_id", "trace_id")


def _suffix_ids(batch: pa.RecordBatch, suffix: str) -> pa.RecordBatch:
    """Return a new RecordBatch with span_id/parent_id/trace_id suffixed.

    NULL values in parent_id (root spans) stay NULL.
    """
    arrays = []
    names = batch.schema.names
    for name in names:
        col = batch.column(batch.schema.get_field_index(name))
        if name in _ID_COLS:
            suffix_scalar = pa.array([suffix] * len(col), type=col.type)
            new_col = pc.binary_join_element_wise(col, suffix_scalar, "")
            arrays.append(new_col)
        else:
            arrays.append(col)
    return pa.RecordBatch.from_arrays(arrays, names=names)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, type=Path)
    ap.add_argument("--out", dest="outp", required=True, type=Path)
    ap.add_argument("--factor", type=int, required=True,
                    help="Number of times each row is duplicated (with unique IDs)")
    ap.add_argument("--batch-size", type=int, default=8192)
    args = ap.parse_args()

    if args.factor < 1:
        print(f"factor must be >= 1, got {args.factor}", file=sys.stderr)
        return 2

    pf = pq.ParquetFile(args.inp)
    schema = pf.schema_arrow
    args.outp.parent.mkdir(parents=True, exist_ok=True)

    writer = pq.ParquetWriter(args.outp, schema, compression="snappy")
    t0 = time.time()
    total_rows = 0
    try:
        for batch in pf.iter_batches(batch_size=args.batch_size):
            for k in range(args.factor):
                if args.factor == 1:
                    out_batch = batch
                else:
                    out_batch = _suffix_ids(batch, f"_k{k}")
                writer.write_batch(out_batch)
                total_rows += out_batch.num_rows
    finally:
        writer.close()

    elapsed = time.time() - t0
    out_bytes = args.outp.stat().st_size
    print(
        f"wrote {total_rows:,} rows to {args.outp} "
        f"({out_bytes/1e6:.1f} MB) in {elapsed:.1f}s"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
