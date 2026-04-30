"""Mews warm-vs-cold and warm-vs-warm paired bench.

Compares engine vs DuckDB on the locked reference query, paired
iteration-by-iteration. Reports median speedup + 95% bootstrap CI.

The engine arm has two modes:
  - cold (default, unchanged): fresh ./engine subprocess each iter,
    timed via /usr/bin/time. This is the original "engine-cold vs
    DuckDB-warm" comparison cell — relevant to process-per-request
    deployment shapes.
  - warm (new in step 6 of gate-4-to-gate-5-runbook): a single
    long-lived ./engine --server <parquet> sidecar holds parquet
    once at startup; queries arrive on stdin, CSV blocks return
    on stdout terminated by a single "END" line. Both arms now
    amortize parquet ingest — pure query comparison.

Methodology:
  - Cold engine arm: spawn fresh ./engine subprocess per iter, time
    wall + RSS via /usr/bin/time -f "%e %M".
  - Warm engine arm: a long-lived ./engine --server sidecar
    handles N+warmup queries; bench measures wall externally
    (perf_counter around the stdin-write/stdout-read pair) and
    samples VmRSS from /proc/<pid>/status after each query.
  - DuckDB arm: a single long-lived python sidecar holds
    duckdb.connect(':memory:') with parquet ingested once. Driver
    writes tenant on stdin, reads "wall_ms peak_rss_kb rows" on
    stdout. READY emitted post-ingest so timed iters skip ingest.
  - Paired: each iteration alternates engine then duckdb. Per-pair
    speedup = duckdb_ms / engine_ms absorbs host jitter into the
    within-pair ratio.
  - Warmup: first --warmup iters discarded (default 3 for warm mode
    to amortize engine-warm's ingest; default 1 for cold mode).
  - Bootstrap CI: 10,000 resamples of per-pair speedups, percentile.

Usage:
    # cold-vs-warm (original)
    python bench_warm.py --workspace .../engine-output \\
        --parquet .../wildchat-x100.parquet --tenant China --iterations 15

    # warm-vs-warm (step 6)
    python bench_warm.py --workspace .../engine-warm/output \\
        --parquet .../wildchat-x100.parquet --tenant China --iterations 15 \\
        --engine-arm warm --warmup 3
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

INFRA_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = INFRA_DIR.parent
SCHEMA_SQL = PROJECT_ROOT / "contracts" / "openinference" / "schema.sql"
REFERENCE_SQL = PROJECT_ROOT / "contracts" / "openinference" / "queries" / "latency_by_model_under_agent.sql"
SIDECAR_SCRIPT = INFRA_DIR / "duckdb_sidecar.py"


def _time_engine(cmd: list[str], cwd: Path, timeout: float) -> tuple[float, float]:
    """Wall-clock + peak RSS via /usr/bin/time for engine subprocess."""
    wrapped = ["/usr/bin/time", "-f", "%e %M", "--"] + cmd
    proc = subprocess.run(wrapped, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(f"engine cmd {cmd} exit={proc.returncode}: {proc.stderr[:200]}")
    last_line = proc.stderr.strip().splitlines()[-1]
    parts = last_line.split()
    return float(parts[0]), float(parts[1])


def _bootstrap_ci(samples: list[float], stat=statistics.median, n: int = 10_000,
                  alpha: float = 0.05, seed: int = 42) -> tuple[float, float]:
    rng = random.Random(seed)
    boots = []
    k = len(samples)
    for _ in range(n):
        resample = [samples[rng.randrange(k)] for _ in range(k)]
        boots.append(stat(resample))
    boots.sort()
    lo = boots[int(n * alpha / 2)]
    hi = boots[int(n * (1 - alpha / 2))]
    return lo, hi


def _read_proc_rss_kb(pid: int) -> int:
    """Read VmRSS from /proc/<pid>/status. Returns 0 if unavailable."""
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
    except (FileNotFoundError, ProcessLookupError):
        return 0
    return 0


class EngineWarmSidecar:
    """Long-lived engine binary in --server mode.

    Stdin protocol: <tenant_id>\\n
    Stdout protocol: CSV header + body lines, terminated by single line "END".

    The engine does NOT emit a READY line — its first query bundles the
    parquet ingest cost, so the bench's --warmup must be ≥1 (default 3
    for warm mode) to amortize ingest before timing begins.

    Wall timing is captured EXTERNALLY (perf_counter around stdin write
    + stdout drain). VmRSS is sampled from /proc/<pid>/status after
    each query — represents engine RSS at-time-of-query.
    """

    def __init__(self, engine_path: Path, parquet: Path, cwd: Path):
        if not engine_path.exists():
            raise RuntimeError(f"engine binary not at {engine_path}")
        self.proc = subprocess.Popen(
            [str(engine_path), "--server", str(parquet)],
            cwd=cwd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        # No READY emit; first query absorbs ingest. Caller must warmup.

    def query(self, tenant: str, timeout: float = 600.0) -> tuple[float, float, int]:
        """Returns (wall_ms, vm_rss_kb_after_query, rows). Timing is external."""
        if self.proc.poll() is not None:
            err = self.proc.stderr.read() if self.proc.stderr else ""
            raise RuntimeError(f"engine-warm sidecar exited: {err[:1000]}")
        t0 = time.perf_counter()
        self.proc.stdin.write(f"{tenant}\n")
        self.proc.stdin.flush()
        deadline = t0 + timeout
        rows = 0  # body row count (excl header + END)
        seen_header = False
        while True:
            if time.perf_counter() > deadline:
                raise RuntimeError(f"engine-warm query timeout ({timeout}s)")
            line = self.proc.stdout.readline()
            if not line:
                err = self.proc.stderr.read() if self.proc.stderr else ""
                raise RuntimeError(f"engine-warm EOF before END: {err[:1000]}")
            stripped = line.rstrip("\n").rstrip("\r")
            if stripped == "END":
                wall_ms = (time.perf_counter() - t0) * 1000
                rss_kb = _read_proc_rss_kb(self.proc.pid)
                return wall_ms, float(rss_kb), rows
            if not seen_header:
                seen_header = True  # discard header
            else:
                rows += 1

    def close(self):
        try:
            if self.proc.poll() is None:
                self.proc.stdin.close()
                self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()


class WarmDuckDBSidecar:
    """Long-lived python process holding a duckdb connection.
    Stdin protocol: <tenant_id>\\n
    Stdout protocol: <wall_ms> <peak_rss_kb> <rows>\\n
    A single READY line is emitted after parquet ingest.
    """

    def __init__(self, parquet: Path, python: str, schema_sql: Path,
                 reference_sql: Path, startup_timeout: float = 600.0):
        self.proc = subprocess.Popen(
            [python, str(SIDECAR_SCRIPT), str(schema_sql),
             str(reference_sql), str(parquet)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # line-buffered
        )
        # Block until sidecar prints READY (parquet ingest can be long at scale)
        deadline = time.perf_counter() + startup_timeout
        while True:
            if time.perf_counter() > deadline:
                self.close()
                raise RuntimeError("sidecar did not emit READY within timeout")
            line = self.proc.stdout.readline()
            if not line:
                err = self.proc.stderr.read() if self.proc.stderr else ""
                raise RuntimeError(f"sidecar exited before READY: {err[:1000]}")
            if line.strip() == "READY":
                break

    def query(self, tenant: str, timeout: float = 600.0) -> tuple[float, float, int]:
        """Returns (wall_ms, peak_rss_kb, rows)."""
        if self.proc.poll() is not None:
            err = self.proc.stderr.read() if self.proc.stderr else ""
            raise RuntimeError(f"sidecar already exited: {err[:1000]}")
        self.proc.stdin.write(f"{tenant}\n")
        self.proc.stdin.flush()
        # Block on the response line; rely on parent timeout via select if needed.
        # For simplicity we use a thread-less readline + a wall-time check loop.
        deadline = time.perf_counter() + timeout
        while True:
            if time.perf_counter() > deadline:
                raise RuntimeError(f"sidecar query timeout ({timeout}s)")
            line = self.proc.stdout.readline()
            if not line:
                err = self.proc.stderr.read() if self.proc.stderr else ""
                raise RuntimeError(f"sidecar EOF: {err[:1000]}")
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 3:
                raise RuntimeError(f"sidecar bad response: {line!r}")
            return float(parts[0]), float(parts[1]), int(parts[2])

    def close(self):
        try:
            if self.proc.poll() is None:
                self.proc.stdin.close()
                self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace", type=Path, required=True)
    p.add_argument("--parquet", type=Path, required=True)
    p.add_argument("--tenant", type=str, required=True)
    p.add_argument("--iterations", type=int, default=15,
                   help="timed paired iterations (after --warmup discards)")
    p.add_argument("--warmup", type=int, default=None,
                   help="warmup iters to discard before timing (default: 1 for cold engine, 3 for warm engine — warm needs >=1 to amortize ingest)")
    p.add_argument("--engine-arm", choices=["cold", "warm"], default="cold",
                   help="cold: spawn fresh engine subprocess each iter (process-per-request shape). "
                        "warm: long-lived ./engine --server sidecar (production shape).")
    p.add_argument("--out-json", type=Path, default=None)
    p.add_argument("--python", type=str, default=sys.executable)
    p.add_argument("--engine-timeout-s", type=float, default=120.0)
    p.add_argument("--duckdb-timeout-s", type=float, default=600.0)
    p.add_argument("--engine-takes-parquet", action="store_true",
                   help="cold mode only: append --parquet to engine_cmd as second positional arg")
    args = p.parse_args(argv)

    if args.warmup is None:
        args.warmup = 3 if args.engine_arm == "warm" else 1

    ws = args.workspace.resolve()
    engine = ws / "engine"
    if not engine.exists():
        print(f"FAIL: no engine binary at {engine}", flush=True)
        return 2

    print(f"workspace: {ws}", flush=True)
    print(f"parquet:   {args.parquet}", flush=True)
    print(f"tenant:    {args.tenant}", flush=True)
    print(f"engine arm: {args.engine_arm}", flush=True)
    print(f"warmup:    {args.warmup} iters (discarded)", flush=True)
    print(f"timed:     {args.iterations} iters", flush=True)
    if args.engine_arm == "cold":
        print(f"mode:      cold engine subprocess vs warm DuckDB sidecar", flush=True)
    else:
        print(f"mode:      warm engine sidecar vs warm DuckDB sidecar (pure-query comparison)", flush=True)
    print("---", flush=True)

    duckdb_sidecar = WarmDuckDBSidecar(
        parquet=args.parquet,
        python=args.python,
        schema_sql=SCHEMA_SQL,
        reference_sql=REFERENCE_SQL,
        startup_timeout=args.duckdb_timeout_s,
    )
    print(f"  [duckdb sidecar ready] parquet ingested once", flush=True)

    engine_warm_sidecar = None
    engine_cold_cmd = None
    if args.engine_arm == "warm":
        engine_warm_sidecar = EngineWarmSidecar(
            engine_path=engine, parquet=args.parquet, cwd=ws,
        )
        print(f"  [engine sidecar started] --server mode; first query absorbs ingest", flush=True)
    else:
        engine_cold_cmd = [str(engine), args.tenant]
        if args.engine_takes_parquet:
            engine_cold_cmd.append(str(args.parquet))

    e_times: list[float] = []
    d_times: list[float] = []
    e_rss_kb: list[float] = []
    d_rss_kb: list[float] = []
    pair_speedups: list[float] = []

    total_iters = args.warmup + args.iterations
    try:
        for i in range(total_iters):
            if args.engine_arm == "warm":
                e_ms, e_rss, _rows = engine_warm_sidecar.query(
                    args.tenant, timeout=args.engine_timeout_s,
                )
                e_s = e_ms / 1000.0
            else:
                e_s, e_rss = _time_engine(
                    engine_cold_cmd, cwd=ws, timeout=args.engine_timeout_s,
                )
            d_ms, d_rss, _rows = duckdb_sidecar.query(
                args.tenant, timeout=args.duckdb_timeout_s,
            )
            d_s = d_ms / 1000.0
            is_warmup = i < args.warmup
            if is_warmup:
                print(f"  iter {i:2d} [warmup discarded]: "
                      f"engine={e_s*1000:.1f}ms/{e_rss/1024:.0f}MB  "
                      f"duckdb-warm={d_ms:.1f}ms/{d_rss/1024:.0f}MB", flush=True)
                continue
            e_times.append(e_s)
            d_times.append(d_s)
            e_rss_kb.append(e_rss)
            d_rss_kb.append(d_rss)
            pair_speedups.append(d_s / e_s)
            print(f"  iter {i:2d}: "
                  f"engine={e_s*1000:.1f}ms/{e_rss/1024:.0f}MB  "
                  f"duckdb-warm={d_ms:.1f}ms/{d_rss/1024:.0f}MB  "
                  f"speedup={d_s/e_s:.2f}x", flush=True)
    finally:
        duckdb_sidecar.close()
        if engine_warm_sidecar is not None:
            engine_warm_sidecar.close()

    e_med = statistics.median(e_times)
    d_med = statistics.median(d_times)
    e_rss_med = statistics.median(e_rss_kb)
    d_rss_med = statistics.median(d_rss_kb)
    sp_med = statistics.median(pair_speedups)
    sp_lo, sp_hi = _bootstrap_ci(pair_speedups, statistics.median)

    engine_label = "engine warm" if args.engine_arm == "warm" else "engine cold"
    print("---", flush=True)
    print(f"{engine_label}:   {e_med*1000:.1f} ms  rss={e_rss_med/1024:.0f}MB  (n={len(e_times)})", flush=True)
    print(f"duckdb warm:    {d_med*1000:.1f} ms  rss={d_rss_med/1024:.0f}MB  (n={len(d_times)})", flush=True)
    print(f"pair speedup:   median={sp_med:.2f}x  range=[{min(pair_speedups):.2f}x, {max(pair_speedups):.2f}x]", flush=True)
    print(f"  95% bootstrap CI on median speedup: [{sp_lo:.2f}x, {sp_hi:.2f}x]", flush=True)

    report = {
        "mode": "warm-vs-warm" if args.engine_arm == "warm" else "warm-vs-cold",
        "engine_arm": args.engine_arm,
        "workspace": str(ws),
        "parquet": str(args.parquet),
        "tenant": args.tenant,
        "warmup": args.warmup,
        "iterations_kept": len(e_times),
        "engine_ms_median": e_med * 1000,
        "duckdb_ms_median": d_med * 1000,
        "engine_rss_mb_median": e_rss_med / 1024,
        "duckdb_rss_mb_median": d_rss_med / 1024,
        "engine_ms_per_iter": [t * 1000 for t in e_times],
        "duckdb_ms_per_iter": [t * 1000 for t in d_times],
        "engine_rss_mb_per_iter": [r / 1024 for r in e_rss_kb],
        "duckdb_rss_mb_per_iter": [r / 1024 for r in d_rss_kb],
        "pair_speedup_median": sp_med,
        "pair_speedup_min": min(pair_speedups),
        "pair_speedup_max": max(pair_speedups),
        "pair_speedup_ci95_lo": sp_lo,
        "pair_speedup_ci95_hi": sp_hi,
    }
    if args.out_json:
        args.out_json.write_text(json.dumps(report, indent=2))
        print(f"wrote {args.out_json}", flush=True)
    else:
        print("---", flush=True)
        print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
