"""Mews engine-warm correctness smoke (Gate 4 / step 5).

Drives the dual-mode engine binary in --server mode, sends N tenant
queries via stdin, reads CSV result blocks (terminated by a literal
"END\\n" line) on stdout, and diffs each against the DuckDB reference
using validate.py's existing diff() — same per-row tolerance as cold
mode validation.

Existence of this script is the warm-mode correctness gate; lockin's
validate_all.py only exercises cold mode.

Usage:
    python smoke_warm.py \\
        --workspace .../engine-warm/output \\
        --parquet .../wildchat-x100.parquet \\
        --tenants 'China,United States,Russia' \\
        --n-queries 10

Exit 0 = all N queries match. Exit 1 = at least one mismatch. Exit 2 =
protocol failure (engine didn't follow stdin/stdout contract).
"""
from __future__ import annotations

import argparse
import csv
import io
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

INFRA_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(INFRA_DIR))
import validate  # type: ignore  # noqa: E402

END_SENTINEL = "END"
STARTUP_TIMEOUT_S = 600.0
QUERY_TIMEOUT_S = 600.0


def _read_csv_block_until_end(stream, deadline: float) -> list[str]:
    """Read lines from `stream` until a line == 'END' is seen.
    Returns the list of CSV lines (header + body) excluding the END line.
    Raises RuntimeError on EOF or deadline.
    """
    lines: list[str] = []
    while True:
        if time.perf_counter() > deadline:
            raise RuntimeError(f"timeout waiting for END sentinel after {len(lines)} lines")
        line = stream.readline()
        if not line:
            raise RuntimeError(
                f"engine stdout closed before END sentinel (got {len(lines)} lines)"
            )
        stripped = line.rstrip("\n").rstrip("\r")
        if stripped == END_SENTINEL:
            return lines
        lines.append(stripped)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace", type=Path, required=True)
    p.add_argument("--parquet", type=Path, required=True)
    p.add_argument("--tenants", type=str, required=True,
                   help="comma-separated tenant_ids; queries cycle through them")
    p.add_argument("--n-queries", type=int, default=10)
    p.add_argument("--rel-tol", type=float, default=1e-3)
    p.add_argument("--startup-timeout-s", type=float, default=STARTUP_TIMEOUT_S)
    p.add_argument("--query-timeout-s", type=float, default=QUERY_TIMEOUT_S)
    args = p.parse_args(argv)

    ws = args.workspace.resolve()
    engine = ws / "engine"
    if not engine.exists():
        print(f"FAIL: no engine binary at {engine}", flush=True)
        return 2

    tenants = [t.strip() for t in args.tenants.split(",") if t.strip()]
    if not tenants:
        print("FAIL: no tenants specified", flush=True)
        return 2

    print(f"workspace: {ws}")
    print(f"parquet:   {args.parquet}")
    print(f"tenants:   {tenants} (cycled)")
    print(f"queries:   {args.n_queries}")
    print(f"rel_tol:   {args.rel_tol}")
    print("---", flush=True)

    proc = subprocess.Popen(
        [str(engine), "--server", str(args.parquet)],
        cwd=ws,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    pass_count = 0
    fail_count = 0
    try:
        startup_deadline = time.perf_counter() + args.startup_timeout_s
        # The protocol does NOT require a READY line — the engine simply
        # starts answering queries. We send the first tenant and read up
        # to its END; any startup ingest cost is folded into that first
        # response wall.

        tmp_csv_dir = Path(tempfile.mkdtemp(prefix="mews-smoke-warm-"))
        try:
            for i in range(args.n_queries):
                tenant = tenants[i % len(tenants)]
                print(f"[query {i+1}/{args.n_queries}] tenant={tenant!r}", flush=True)

                # Write tenant on stdin.
                if proc.poll() is not None:
                    err = proc.stderr.read() if proc.stderr else ""
                    print(f"FAIL: engine exited before query {i+1}: {err[:500]}",
                          flush=True)
                    return 2
                proc.stdin.write(f"{tenant}\n")
                proc.stdin.flush()

                # Read result block.
                deadline = (
                    startup_deadline if i == 0
                    else time.perf_counter() + args.query_timeout_s
                )
                t0 = time.perf_counter()
                try:
                    csv_lines = _read_csv_block_until_end(proc.stdout, deadline)
                except RuntimeError as e:
                    err = proc.stderr.read() if proc.stderr else ""
                    print(f"FAIL: protocol error on query {i+1}: {e}\n"
                          f"  engine stderr: {err[:500]}", flush=True)
                    return 2
                wall_ms = (time.perf_counter() - t0) * 1000

                # Materialize as CSV file for validate.diff().
                csv_text = "\n".join(csv_lines) + "\n"
                csv_path = tmp_csv_dir / f"q{i+1}.{validate.__name__}.csv"
                # validate.diff expects header + body. Sanity check:
                if not csv_lines:
                    print(f"  WARN: empty CSV block (no header) — treating as 0 rows",
                          flush=True)
                csv_path.write_text(csv_text)

                # Diff vs DuckDB reference for THIS tenant on THIS parquet.
                res = validate.diff(args.parquet, tenant, csv_path,
                                    rel_tol=args.rel_tol)
                if res.ok:
                    pass_count += 1
                    print(f"  PASS: {res.summary} ({wall_ms:.0f} ms)", flush=True)
                else:
                    fail_count += 1
                    print(f"  FAIL: {res.summary} ({wall_ms:.0f} ms)", flush=True)
                    for line in res.detail_lines[:6]:
                        print(f"    {line}", flush=True)
        finally:
            shutil.rmtree(tmp_csv_dir, ignore_errors=True)
    finally:
        try:
            if proc.poll() is None:
                proc.stdin.close()
                proc.wait(timeout=10)
        except Exception:
            proc.kill()

    print("---", flush=True)
    overall_ok = (fail_count == 0 and pass_count == args.n_queries)
    print(f"OVERALL: {'PASS' if overall_ok else 'FAIL'} "
          f"({pass_count}/{args.n_queries})", flush=True)
    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
