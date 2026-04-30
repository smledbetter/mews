"""Long-lived DuckDB sidecar for bench_warm.py.

Holds a single duckdb.connect(':memory:') with parquet ingested once.
Reads tenant_id lines on stdin, runs the locked reference query, writes
"<wall_ms> <peak_rss_kb> <rows>\\n" on stdout per query.

Emits a single READY line after parquet ingest completes.

Stdin EOF -> graceful shutdown.

Usage:
    python duckdb_sidecar.py <schema.sql> <reference.sql> <parquet>
"""
from __future__ import annotations

import pathlib
import resource
import sys
import time

import duckdb


def main():
    if len(sys.argv) != 4:
        print(f"usage: {sys.argv[0]} <schema.sql> <reference.sql> <parquet>",
              file=sys.stderr)
        return 2
    schema = pathlib.Path(sys.argv[1]).read_text()
    ref = pathlib.Path(sys.argv[2]).read_text()
    parquet = sys.argv[3]

    con = duckdb.connect(":memory:")
    con.execute(schema)
    con.execute(f"INSERT INTO spans SELECT * FROM read_parquet('{parquet}')")
    print("READY", flush=True)

    for line in sys.stdin:
        tenant = line.strip()
        if not tenant:
            continue
        t0 = time.perf_counter_ns()
        rows = con.execute(ref, (tenant,)).fetchall()
        wall_ms = (time.perf_counter_ns() - t0) / 1e6
        # ru_maxrss is high-water-mark since process start (kB on Linux).
        # In a long-lived sidecar this is monotonic; the headline RSS is
        # essentially "DuckDB peak across all queries handled so far,"
        # which is what we want for capacity-planning purposes.
        peak_rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        print(f"{wall_ms:.3f} {peak_rss_kb} {len(rows)}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
