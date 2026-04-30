"""Mews autonomous-loop orchestration library.

Reusable components for the autonomous-regen loop. G7 (first close) uses
this through a thin wrapper script; G8 (sustained loop) iterates
`run_one_cycle` across multiple drift inducers.

Key abstractions:
    initialize_deployed(deployed_ws, initial_engine, initial_cpp) ->
        one-time setup of the deployment workspace
    run_one_cycle(cycle_id, ...) -> CycleResult
        single detect -> regen -> deploy -> re-detect cycle

Exit-code semantics for runner.py (post-2026-04-30):
    0 = agent yielded successfully + post-validate PASS
    1 = exception or MaxTurnsExceeded with validate FAIL
    2 = agent yielded successfully but post-validate FAIL
    3 = MaxTurnsExceeded but artifact validates clean (substantive success)

Exit codes 0 and 3 both mean "artifact is OK." The orchestrator treats
them as equivalent for deploy purposes; the cycle manifest records the
distinction for analysis.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# --- Paths (project-relative via env var, with __file__-derived fallback) ---

ROOT = Path(os.environ.get("MEWS_ROOT", str(Path(__file__).resolve().parent.parent)))
INFRA = ROOT / "infra"
DETECT = INFRA / "drift_detector.py"
RUNNER = INFRA / "runner.py"

UV = os.environ.get("UV_BIN", "uv")
DEFAULT_ENV = {**os.environ, "UV_CACHE_DIR": os.environ.get("UV_CACHE_DIR", "/tmp/uv_cache")}

# Cost-watch parsing: cached_litellm:Cost: $X.YYYY (per-call).
COST_RE = re.compile(r"cached_litellm[^:]*:Cost:\s+\$([0-9]+\.[0-9]+)")


# --- Data ---

@dataclass
class CycleResult:
    cycle_id: str
    started_at: str
    finished_at: str
    wall_s: float

    pre_report: Optional[dict] = None
    pre_regen_recommended: bool = False
    pre_regen_reason: str = "none"

    regen_exit_code: Optional[int] = None
    regen_wall_s: Optional[float] = None
    regen_cost_usd: Optional[float] = None
    regen_artifact_ok: Optional[bool] = None
    regen_artifact_msg: Optional[str] = None

    deploy_done: bool = False

    post_report: Optional[dict] = None
    post_regen_recommended: Optional[bool] = None

    cycle_closed: bool = False
    abort_reason: Optional[str] = None


# --- Helpers ---

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log_line(msg: str, log_file: Path) -> None:
    line = f"[{now_iso()}] {msg}\n"
    sys.stdout.write(line)
    sys.stdout.flush()
    with log_file.open("a") as f:
        f.write(line)


def parse_stage3_cost(stdout_text: str) -> float:
    """Sum all `cached_litellm:Cost: $X` lines in stage-3 stdout."""
    return round(sum(float(m) for m in COST_RE.findall(stdout_text)), 4)


def cleanup_validate_artifacts(workspace: Path) -> None:
    """Remove result1*.csv left behind by validate_all.py runs.

    validate_all.py writes result1.csv (raw engine output) plus per-tenant
    saved copies (result1.<safe_tenant>.csv) into the workspace it ran
    against. For sustained-loop operation these accumulate; clean them
    after each validate cycle.
    """
    for p in workspace.glob("result1*.csv"):
        try:
            p.unlink()
        except OSError:
            pass


def initialize_deployed(deployed_ws: Path, initial_engine: Path, initial_cpp: Path) -> None:
    """One-time setup: stage the initial engine + cpp into the deployment workspace.

    Idempotent: if deployed_ws already has an engine, this overwrites with
    the supplied initial state. The G8 sustained loop should call this
    once at the start; later cycles mutate deployed_ws in place via
    hot-deploy.
    """
    deployed_ws.mkdir(parents=True, exist_ok=True)
    shutil.copy(initial_engine, deployed_ws / "engine")
    (deployed_ws / "engine").chmod(0o755)
    shutil.copy(initial_cpp, deployed_ws / "engine.cpp")


def initialize_regen_seed(regen_ws: Path, seed_engine: Path, seed_cpp: Path) -> None:
    """Set up a Stage-3 workspace from a known-good seed engine.

    Wipes regen_ws first so prior cycle state (lockin snapshots,
    .synthesis_schema.json, etc.) doesn't leak into the new run.
    """
    if regen_ws.exists():
        shutil.rmtree(regen_ws)
    regen_ws.mkdir(parents=True, exist_ok=True)
    shutil.copy(seed_engine, regen_ws / "engine")
    (regen_ws / "engine").chmod(0o755)
    shutil.copy(seed_cpp, regen_ws / "engine.cpp")


def run_drift_detector(
    label: str,
    engine: Path,
    workspace: Path,
    parquet: Path,
    baseline_schema: Path,
    perf_tenant: str,
    out: Path,
    log_file: Path,
    tenants: str = "China,United States,Russia",
    skip_duckdb_perf: bool = False,
    perf_iters: int = 3,
    timeout: int = 900,
    env: Optional[dict] = None,
) -> tuple[int, dict[str, Any]]:
    """Run drift_detector.py and return (exit_code, report_dict).

    After the call, cleans validate_all.py result1*.csv artifacts from
    the workspace so subsequent cycles start clean.
    """
    log_line(f"--- {label}: drift detector ---", log_file)
    cmd = [
        UV, "run", "--with", "pyarrow", "--with", "duckdb", "python", str(DETECT),
        "--engine", str(engine),
        "--workspace", str(workspace),
        "--parquet", str(parquet),
        "--tenants", tenants,
        "--baseline-schema", str(baseline_schema),
        "--perf-tenant", perf_tenant,
        "--perf-iters", str(perf_iters),
        "--out", str(out),
    ]
    if skip_duckdb_perf:
        cmd.append("--skip-duckdb-perf")
    log_line(f"  cmd: {' '.join(cmd)}", log_file)
    proc = subprocess.run(
        cmd, cwd=ROOT / "upstream/BespokeOLAP",
        capture_output=True, text=True, timeout=timeout,
        env=env or DEFAULT_ENV,
    )
    log_line(f"  exit={proc.returncode}", log_file)
    if proc.stdout:
        log_line(f"  stdout (last 600ch):\n{proc.stdout[-600:]}", log_file)
    if proc.stderr:
        # stderr can contain the full uv warning so trim aggressively.
        log_line(f"  stderr (last 400ch):\n{proc.stderr[-400:]}", log_file)
    cleanup_validate_artifacts(workspace)
    if not out.exists():
        raise RuntimeError(f"{label}: drift detector produced no report at {out}")
    return proc.returncode, json.loads(out.read_text())


def run_stage3(
    regen_ws: Path,
    log_dir: Path,
    parquet_x1: Path,
    parquet_x10: Path,
    bench_parquet: Path,
    bench_tenant: str,
    validation_tenants: str,
    model: str,
    max_turns: int,
    log_file: Path,
    stdout_log: Path,
    stage3_prompt: Optional[str] = None,
    arm_lockin_pre: bool = True,
    timeout: int = 3600,
    env: Optional[dict] = None,
) -> tuple[int, float]:
    """Launch a Stage-3 synthesis. Captures stdout to stdout_log.

    Returns (exit_code, cost_usd). Exit codes follow the runner.py
    semantics (0/1/2/3); both 0 and 3 indicate an artifact that
    post-stage-validate accepted.
    """
    log_line("--- regen: Stage-3 synthesis ---", log_file)
    log_line(f"  regen_ws={regen_ws}", log_file)
    log_line(f"  log_dir={log_dir}", log_file)
    cmd = [
        UV, "run", "python", str(RUNNER),
        "--stage", "3",
        "--tenant", "China",
        "--workspace", str(regen_ws),
        "--log-dir", str(log_dir),
        "--max-turns", str(max_turns),
        "--model", model,
        "--parquet", str(parquet_x1),
        "--parquet-x10", str(parquet_x10),
        "--bench-parquet", str(bench_parquet),
        "--bench-tenant", bench_tenant,
        "--validation-tenants", validation_tenants,
    ]
    if stage3_prompt:
        cmd += ["--stage3-prompt", stage3_prompt]
    if arm_lockin_pre:
        cmd.append("--arm-lockin-pre")
    log_line(f"  cmd: {' '.join(cmd)}", log_file)

    # Capture stdout so we can parse cost. Tee to the per-cycle stdout_log.
    with stdout_log.open("w") as out:
        proc = subprocess.run(
            cmd, cwd=ROOT / "upstream/BespokeOLAP",
            stdout=out, stderr=subprocess.STDOUT, text=True,
            timeout=timeout, env=env or DEFAULT_ENV,
        )

    cost = parse_stage3_cost(stdout_log.read_text())
    log_line(f"  Stage-3 exit={proc.returncode}  cost=${cost:.2f}", log_file)
    return proc.returncode, cost


def stage3_artifact_validates(
    regen_ws: Path,
    parquet_x1: Path,
    parquet_x10: Path,
    log_file: Path,
    tenants: str = "China,United States,Russia",
    env: Optional[dict] = None,
) -> tuple[bool, str]:
    """Substantive-correctness check used as a fallback for exit codes 1/2.

    With the new exit-3 semantic in runner.py, exit 0 and exit 3 already
    imply substantive correctness (runner ran post_stage_validate before
    returning either). This helper is the orchestrator-side fallback for
    the case where runner returned 1 or 2 but the artifact may still be
    correct (e.g., a runner bug that mis-classifies). Cheap to run.
    """
    log_line("--- post-Stage-3 substantive validation ---", log_file)
    for parq, label in [(parquet_x1, "x1"), (parquet_x10, "x10")]:
        cmd = [
            UV, "run", "python", str(INFRA / "validate_all.py"),
            "--workspace", str(regen_ws),
            "--parquet", str(parq),
            "--tenants", tenants,
        ]
        proc = subprocess.run(
            cmd, cwd=ROOT / "upstream/BespokeOLAP",
            capture_output=True, text=True, timeout=300,
            env=env or DEFAULT_ENV,
        )
        ok = proc.returncode == 0 and "OVERALL: PASS" in proc.stdout
        log_line(f"  {label}: exit={proc.returncode} {'PASS' if ok else 'FAIL'}", log_file)
        if not ok:
            cleanup_validate_artifacts(regen_ws)
            return False, f"validate_all.py at {label} did not PASS"
    cleanup_validate_artifacts(regen_ws)
    return True, "validate_all.py PASS at x1+x10"


def hot_deploy(regen_ws: Path, deployed_ws: Path, log_file: Path) -> None:
    log_line("--- deploy: copy regen engine -> deployed/ ---", log_file)
    shutil.copy(regen_ws / "engine", deployed_ws / "engine")
    (deployed_ws / "engine").chmod(0o755)
    shutil.copy(regen_ws / "engine.cpp", deployed_ws / "engine.cpp")


def run_one_cycle(
    cycle_id: str,
    deployed_ws: Path,
    regen_seed_engine: Path,
    regen_seed_cpp: Path,
    trigger_parquet: Path,
    baseline_schema: Path,
    cycle_dir: Path,
    log_file: Path,
    parquet_x1: Path,
    parquet_x10: Path,
    bench_parquet: Path,
    bench_tenant: str = "China",
    validation_tenants: str = "China,United States,Russia",
    model: str = "litellm/anthropic/claude-opus-4-6",
    max_turns: int = 75,
    stage3_prompt: Optional[str] = None,
    perf_tenant: str = "China",
    skip_duckdb_perf: bool = False,
    expected_pre_drift: bool = True,
    update_baseline_after_regen: bool = False,
) -> CycleResult:
    """Run one detect -> regen -> deploy -> re-detect cycle.

    Mutates deployed_ws in place: replaces engine + engine.cpp on
    successful regen + deploy.

    Returns a CycleResult with phase wall times, regen cost, both drift
    reports, and a final cycle_closed verdict.
    """
    cycle_dir.mkdir(parents=True, exist_ok=True)
    regen_ws = cycle_dir / "regen-ws"
    regen_log_dir = cycle_dir / "regen-log"
    pre_report_path = cycle_dir / "pre-regen.report.json"
    post_report_path = cycle_dir / "post-regen.report.json"
    stdout_log = cycle_dir / "regen-stdout.log"

    started_at = now_iso()
    t_start = time.time()
    result = CycleResult(cycle_id=cycle_id, started_at=started_at, finished_at="", wall_s=0.0)

    log_line("=" * 78, log_file)
    log_line(f"CYCLE {cycle_id} START", log_file)
    log_line(f"  deployed_ws={deployed_ws}", log_file)
    log_line(f"  regen_seed_engine={regen_seed_engine}", log_file)
    log_line(f"  trigger_parquet={trigger_parquet}", log_file)
    log_line("=" * 78, log_file)

    # PHASE 1: pre-regen detect
    _exit, pre = run_drift_detector(
        f"{cycle_id} pre-regen",
        deployed_ws / "engine", deployed_ws,
        trigger_parquet, baseline_schema, perf_tenant,
        pre_report_path, log_file,
        tenants=validation_tenants,
        skip_duckdb_perf=skip_duckdb_perf,
    )
    result.pre_report = pre
    result.pre_regen_recommended = bool(pre.get("regen_recommended"))
    result.pre_regen_reason = pre.get("regen_reason", "none")
    log_line(
        f"  pre-regen verdict: schema={pre['schema_status']} "
        f"correctness={pre['correctness_status']} "
        f"regen={result.pre_regen_recommended}/{result.pre_regen_reason}",
        log_file,
    )

    if not result.pre_regen_recommended:
        if expected_pre_drift:
            result.abort_reason = "pre-regen verdict was not regen-recommended"
            log_line(f"ABORT: expected pre-regen drift but got NONE", log_file)
        else:
            result.cycle_closed = True
            log_line("cycle: no drift detected; nothing to regen (idle)", log_file)
        result.finished_at = now_iso()
        result.wall_s = round(time.time() - t_start, 2)
        return result

    # PHASE 2: Stage-3 regen
    initialize_regen_seed(regen_ws, regen_seed_engine, regen_seed_cpp)
    regen_log_dir.mkdir(parents=True, exist_ok=True)
    t_regen = time.time()
    regen_exit, regen_cost = run_stage3(
        regen_ws, regen_log_dir,
        parquet_x1, parquet_x10, bench_parquet, bench_tenant, validation_tenants,
        model, max_turns, log_file, stdout_log,
        stage3_prompt=stage3_prompt, arm_lockin_pre=True,
    )
    result.regen_exit_code = regen_exit
    result.regen_wall_s = round(time.time() - t_regen, 2)
    result.regen_cost_usd = regen_cost

    # Exit codes 0 + 3 mean runner already ran post_stage_validate and it
    # PASSed. For 1 + 2 we re-run substantive check as a fallback.
    if regen_exit in (0, 3):
        result.regen_artifact_ok = True
        result.regen_artifact_msg = (
            "runner.py exit 3: substantive success (MaxTurnsExceeded but artifact validates clean)"
            if regen_exit == 3 else "runner.py exit 0: clean yield"
        )
    else:
        ok, msg = stage3_artifact_validates(regen_ws, parquet_x1, parquet_x10, log_file, validation_tenants)
        result.regen_artifact_ok = ok
        result.regen_artifact_msg = msg

    log_line(f"  artifact_ok={result.regen_artifact_ok} ({result.regen_artifact_msg})", log_file)

    if not result.regen_artifact_ok:
        result.abort_reason = "regen artifact failed substantive validation"
        log_line(f"KILL CYCLE: {result.abort_reason}", log_file)
        result.finished_at = now_iso()
        result.wall_s = round(time.time() - t_start, 2)
        return result

    # Optional: refresh the baseline-schema reference from the regen
    # workspace before post-detect runs. Used when the inducer legitimately
    # changes the parquet shape (e.g., G8 cycle 3 extra-col): without this
    # the post-detect would re-fire schema drift on the inducer parquet.
    if update_baseline_after_regen:
        new_schema = regen_ws / ".synthesis_schema.json"
        if new_schema.exists():
            shutil.copy(new_schema, baseline_schema)
            log_line(f"baseline_schema refreshed from {new_schema} -> {baseline_schema}", log_file)
        else:
            log_line(f"WARN: update_baseline_after_regen=True but {new_schema} missing", log_file)

    # PHASE 3: hot-deploy
    hot_deploy(regen_ws, deployed_ws, log_file)
    result.deploy_done = True

    # PHASE 4: post-regen detect
    _exit, post = run_drift_detector(
        f"{cycle_id} post-regen",
        deployed_ws / "engine", deployed_ws,
        trigger_parquet, baseline_schema, perf_tenant,
        post_report_path, log_file,
        tenants=validation_tenants,
        skip_duckdb_perf=skip_duckdb_perf,
    )
    result.post_report = post
    result.post_regen_recommended = bool(post.get("regen_recommended"))
    log_line(
        f"  post-regen verdict: schema={post['schema_status']} "
        f"correctness={post['correctness_status']} "
        f"regen={result.post_regen_recommended}/{post.get('regen_reason','none')}",
        log_file,
    )

    result.cycle_closed = (
        post["schema_status"] == "MATCH"
        and post["correctness_status"] == "MATCH"
        and not result.post_regen_recommended
    )
    result.finished_at = now_iso()
    result.wall_s = round(time.time() - t_start, 2)

    log_line("=" * 78, log_file)
    log_line(
        f"CYCLE {cycle_id} END: closed={result.cycle_closed} "
        f"wall={result.wall_s:.0f}s cost=${result.regen_cost_usd:.2f}",
        log_file,
    )
    log_line("=" * 78, log_file)
    return result


def write_loop_manifest(manifest_path: Path, cycles: list[CycleResult], extra: Optional[dict] = None) -> None:
    """Write a multi-cycle manifest aggregating cycle results."""
    manifest = {
        "cycles": [asdict(c) for c in cycles],
        "total_cycles": len(cycles),
        "cycles_closed": sum(1 for c in cycles if c.cycle_closed),
        "total_cost_usd": round(sum((c.regen_cost_usd or 0.0) for c in cycles), 4),
        "total_wall_s": round(sum(c.wall_s for c in cycles), 2),
    }
    if extra:
        manifest.update(extra)
    manifest_path.write_text(json.dumps(manifest, indent=2))
