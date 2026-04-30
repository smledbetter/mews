"""Mews paired-runs benchmark with bootstrap CIs.

Compares engine binary cold-start vs DuckDB cold-start on the locked reference
query, paired iteration-by-iteration to absorb host jitter into the within-pair
difference. Reports median speedup, p99 speedup, and 95% bootstrap CIs over
the per-pair speedup distribution.

Methodology:
  - Cold-vs-cold: each measurement is a fresh subprocess (no warmed caches,
    no shared interpreter). This is what Gate-1's 26× claim is anchored on.
  - Paired: iterations alternate (engine_i, duckdb_i) so any host load spike
    affects both arms similarly — speedup is the within-pair ratio.
  - Bootstrap: 10,000 resamples of per-pair speedups, percentile CI.
  - Warmup: first iteration discarded (page cache prefetch, libraries linked).

Usage:
    python bench.py \\
        --workspace ~/projects/mews/experiments/gate-1/run-4/output \\
        --parquet ~/projects/mews/artifacts/openinference_parquet/sf1/spans.parquet \\
        --tenant tenant-01 \\
        --iterations 25
"""
from __future__ import annotations

import argparse
import json
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


DUCKDB_RUNNER = """\
import sys, duckdb, pathlib
schema = pathlib.Path(sys.argv[1]).read_text()
ref    = pathlib.Path(sys.argv[2]).read_text()
parquet = sys.argv[3]
tenant  = sys.argv[4]
con = duckdb.connect(':memory:')
con.execute(schema)
con.execute(f"INSERT INTO spans SELECT * FROM read_parquet('{parquet}')")
rows = con.execute(ref, (tenant,)).fetchall()
print(f'rows={len(rows)}')
"""


def _time_one(cmd: list[str], cwd: Path, timeout: float = 60.0) -> float:
    """Wall-clock seconds for one fresh subprocess. Raises on non-zero exit."""
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    elapsed = time.perf_counter() - t0
    if proc.returncode != 0:
        raise RuntimeError(f"cmd {cmd} exit={proc.returncode}: {proc.stderr[:200]}")
    return elapsed


def _time_one_with_rss(cmd: list[str], cwd: Path, timeout: float) -> tuple[float, float]:
    """Wall-clock + peak RSS via /usr/bin/time. Returns (wall_s, rss_kb)."""
    wrapped = ["/usr/bin/time", "-f", "%e %M", "--"] + cmd
    proc = subprocess.run(wrapped, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(f"cmd {cmd} exit={proc.returncode}: {proc.stderr[:200]}")
    last_line = proc.stderr.strip().splitlines()[-1]
    parts = last_line.split()
    return float(parts[0]), float(parts[1])


def _bootstrap_ci(samples: list[float], stat=statistics.median, n: int = 10_000,
                  alpha: float = 0.05, seed: int = 42) -> tuple[float, float]:
    """Percentile-bootstrap CI."""
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


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace", type=Path, required=True,
                   help="dir containing engine binary; results written here")
    p.add_argument("--parquet", type=Path, required=True,
                   help="span parquet (engine reads this; DuckDB ingests this)")
    p.add_argument("--tenant", type=str, required=True)
    p.add_argument("--iterations", type=int, default=25,
                   help="paired iterations (first is warmup)")
    p.add_argument("--out-json", type=Path, default=None,
                   help="optional path to write a structured report")
    p.add_argument("--python", type=str, default=sys.executable,
                   help="Python used for the DuckDB runner subprocess")
    p.add_argument("--timeout-s", type=float, default=60.0,
                   help="per-iteration subprocess timeout (default 60s)")
    p.add_argument("--engine-takes-parquet", action="store_true",
                   help="if set, append --parquet to engine_cmd as second positional arg")
    args = p.parse_args(argv)

    ws = args.workspace.resolve()
    engine = ws / "engine"
    if not engine.exists():
        print(f"FAIL: no engine binary at {engine}", flush=True)
        return 2

    runner_script = ws / "_bench_duckdb_runner.py"
    runner_script.write_text(DUCKDB_RUNNER)

    engine_cmd = [str(engine), args.tenant]
    if args.engine_takes_parquet:
        engine_cmd.append(str(args.parquet))
    duckdb_cmd = [args.python, str(runner_script), str(SCHEMA_SQL), str(REFERENCE_SQL),
                  str(args.parquet), args.tenant]

    print(f"workspace: {ws}", flush=True)
    print(f"parquet:   {args.parquet}", flush=True)
    print(f"tenant:    {args.tenant}", flush=True)
    print(f"iters:     {args.iterations} (first discarded as warmup)", flush=True)
    print("---", flush=True)

    e_times: list[float] = []
    d_times: list[float] = []
    e_rss_kb: list[float] = []
    d_rss_kb: list[float] = []
    pair_speedups: list[float] = []

    for i in range(args.iterations):
        e, e_rss = _time_one_with_rss(engine_cmd, cwd=ws, timeout=args.timeout_s)
        d, d_rss = _time_one_with_rss(duckdb_cmd, cwd=ws, timeout=args.timeout_s)
        if i == 0:
            print(f"  iter {i:2d} (warmup discarded): "
                  f"engine={e*1000:.1f}ms/{e_rss/1024:.0f}MB  "
                  f"duckdb={d*1000:.1f}ms/{d_rss/1024:.0f}MB", flush=True)
            continue
        e_times.append(e)
        d_times.append(d)
        e_rss_kb.append(e_rss)
        d_rss_kb.append(d_rss)
        pair_speedups.append(d / e)
        print(f"  iter {i:2d}: "
              f"engine={e*1000:.1f}ms/{e_rss/1024:.0f}MB  "
              f"duckdb={d*1000:.1f}ms/{d_rss/1024:.0f}MB  "
              f"speedup={d/e:.2f}x", flush=True)

    e_med = statistics.median(e_times)
    d_med = statistics.median(d_times)
    e_rss_med = statistics.median(e_rss_kb)
    d_rss_med = statistics.median(d_rss_kb)
    e_rss_max = max(e_rss_kb)
    d_rss_max = max(d_rss_kb)
    sp_med = statistics.median(pair_speedups)
    sp_min = min(pair_speedups)
    sp_max = max(pair_speedups)
    sp_p99 = sorted(pair_speedups)[int(len(pair_speedups) * 0.99)] if len(pair_speedups) >= 100 else sp_max

    sp_lo, sp_hi = _bootstrap_ci(pair_speedups, statistics.median)

    print("---", flush=True)
    print(f"engine median:  {e_med*1000:.1f} ms  rss_median={e_rss_med/1024:.0f}MB rss_max={e_rss_max/1024:.0f}MB  (n={len(e_times)})", flush=True)
    print(f"duckdb median:  {d_med*1000:.1f} ms  rss_median={d_rss_med/1024:.0f}MB rss_max={d_rss_max/1024:.0f}MB  (n={len(d_times)})", flush=True)
    print(f"pair speedup:   median={sp_med:.2f}x  range=[{sp_min:.2f}x, {sp_max:.2f}x]", flush=True)
    print(f"  95% bootstrap CI on median speedup: [{sp_lo:.2f}x, {sp_hi:.2f}x]", flush=True)
    print(f"  ratio of medians (less defensible): {d_med/e_med:.2f}x", flush=True)
    print(f"rss ratio (duckdb/engine): {d_rss_med/e_rss_med:.2f}x at median", flush=True)

    report = {
        "workspace": str(ws),
        "parquet": str(args.parquet),
        "tenant": args.tenant,
        "iterations_kept": len(e_times),
        "engine_ms_median": e_med * 1000,
        "duckdb_ms_median": d_med * 1000,
        "engine_rss_mb_median": e_rss_med / 1024,
        "duckdb_rss_mb_median": d_rss_med / 1024,
        "engine_rss_mb_max": e_rss_max / 1024,
        "duckdb_rss_mb_max": d_rss_max / 1024,
        "engine_ms_per_iter": [t * 1000 for t in e_times],
        "duckdb_ms_per_iter": [t * 1000 for t in d_times],
        "engine_rss_mb_per_iter": [r / 1024 for r in e_rss_kb],
        "duckdb_rss_mb_per_iter": [r / 1024 for r in d_rss_kb],
        "pair_speedup_median": sp_med,
        "pair_speedup_min": sp_min,
        "pair_speedup_max": sp_max,
        "pair_speedup_p99": sp_p99,
        "pair_speedup_ci95_lo": sp_lo,
        "pair_speedup_ci95_hi": sp_hi,
    }
    if args.out_json:
        args.out_json.write_text(json.dumps(report, indent=2))
        print(f"wrote {args.out_json}", flush=True)
    else:
        print("---", flush=True)
        print(json.dumps(report, indent=2), flush=True)

    runner_script.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
