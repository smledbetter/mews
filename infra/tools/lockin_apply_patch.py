"""Tool-level lock-in for apply_patch.

Wraps BespokeOLAP's `make_litellm_apply_patch_tool` so that, once the
workspace has reached a multi-tenant-correct state (signaled by the
`.lockin_armed` marker file written by runner._post_stage_validate),
every subsequent apply_patch on engine.cpp is gated:

  1. Snapshot engine.cpp.
  2. Apply the patch normally.
  3. Recompile engine.
  4. Run validate_all.py at x1 (existing tenants list).
  5. If MEWS_LOCKIN_PARQUET_X10 is set, run validate_all.py at x10 too.
     Catches the silent-x1 pattern: engines that ignore argv[2] and
     read a hardcoded x1 path pass step 4 but fail step 5 because their
     x10 row counts don't match DuckDB's.
  6. If MEWS_LOCKIN_BEST_KEEPER=1 + MEWS_LOCKIN_BENCH_PARQUET set, run
     a quick 3-iter engine bench. If new median is >5% slower than the
     last known-best, revert. Catches speedup-chasing edits that
     silently regress wall time.
  7. Any failure in 3-6 -> restore snapshot, recompile, surface the
     failure to the agent in the tool result.

When the marker is absent the wrapper is transparent. Step 5 + step 6
opt in via env so Stage-2 builds fast (correctness-first, no scale or
speed gating until Stage-3 specialization).

The lock-in is keyed on engine.cpp. Edits to other files are not gated.
"""

from __future__ import annotations

import json
import logging
import os
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any, Optional

from agents.run_context import RunContextWrapper
from agents.tool import FunctionTool

from tools.litellm_apply_patch import (  # type: ignore
    LitellmApplyPatchArgs,
    LitellmApplyPatchTool,
)
from utils.wandb_stats_logging import WandbRunHook  # type: ignore

logger = logging.getLogger("mews.lockin")

_GATED_FILES = {"engine.cpp"}
_LOCKIN_MARKER = ".lockin_armed"
_SNAPSHOT_DIR = ".lockin"
_BEST_BENCH_FILE = ".lockin_best_bench.json"
_INFRA_DIR = Path(__file__).resolve().parent.parent
_VALIDATOR = _INFRA_DIR / "validate_all.py"
_UV = os.environ.get("UV_BIN", "uv")

# Best-keeper regression threshold. New median engine wall must be at
# most BEST_KEEPER_TOLERANCE * old median or the patch is reverted.
# 1.05 = allow up to 5% slower; trades small noise tolerance for early
# detection of meaningful regressions. The 3-iter bench has bootstrap
# noise dominated by host jitter; this threshold was sized against the
# x100 paired bench's ~3% inter-iter variance.
BEST_KEEPER_TOLERANCE = 1.05


def _should_gate(rel_path: str) -> bool:
    name = Path(rel_path).name
    return name in _GATED_FILES


def _lockin_armed(workspace: Path) -> bool:
    return (workspace / _LOCKIN_MARKER).exists()


def _snapshot_path(workspace: Path, rel_path: str, tag: str) -> Path:
    snap_dir = workspace / _SNAPSHOT_DIR
    snap_dir.mkdir(exist_ok=True)
    return snap_dir / f"{Path(rel_path).name}.{tag}.snap"


def _recompile(workspace: Path) -> tuple[bool, str]:
    """Re-run a best-effort g++ build of the engine.

    Tries libsnappy-only first (Stage 3 bespoke shape), then falls back
    to libarrow + libparquet (Stage 2 shape). Returns (ok, output).
    """
    cmds = [
        # Stage 3 bespoke build
        ["g++", "-O3", "-march=native", "-std=c++20",
         "-o", "engine", "engine.cpp", "-lsnappy"],
        # Stage 2 libarrow build (let pkg-config expand)
        ["bash", "-c",
         "g++ -O3 -march=native -std=c++20 -o engine engine.cpp "
         "$(pkg-config --cflags --libs arrow parquet)"],
    ]
    last_err = ""
    for cmd in cmds:
        try:
            proc = subprocess.run(
                cmd, cwd=workspace, capture_output=True, text=True, timeout=60,
            )
        except subprocess.TimeoutExpired:
            last_err = "compile timeout"
            continue
        if proc.returncode == 0:
            return True, proc.stdout + proc.stderr
        last_err = proc.stdout + proc.stderr
    return False, last_err


def _validate_at(workspace: Path, parquet: str, tenants: str,
                 timeout: int = 120) -> tuple[bool, str]:
    """Run validate_all.py against a specific parquet."""
    cmd = [_UV, "run", "python", str(_VALIDATOR),
           "--workspace", str(workspace)]
    if parquet:
        cmd += ["--parquet", parquet]
    if tenants:
        cmd += ["--tenants", tenants]
    env = dict(os.environ)
    env["UV_CACHE_DIR"] = env.get("UV_CACHE_DIR", "/tmp/uv_cache")
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, env=env,
        )
    except subprocess.TimeoutExpired:
        return False, f"validate_all.py timeout ({timeout}s) at parquet={parquet or '<default>'}"
    output = proc.stdout + ("\n" + proc.stderr if proc.stderr else "")
    ok = proc.returncode == 0 and "OVERALL: PASS" in proc.stdout
    return ok, output


def _validate_all(workspace: Path) -> tuple[bool, str]:
    """Validate at the configured x1 parquet. Falls back to the synthetic
    Gate-1 fixture when MEWS_LOCKIN_PARQUET is unset (back-compat)."""
    return _validate_at(
        workspace,
        os.environ.get("MEWS_LOCKIN_PARQUET", "").strip(),
        os.environ.get("MEWS_LOCKIN_TENANTS", "").strip(),
    )


def _validate_at_x10(workspace: Path) -> Optional[tuple[bool, str]]:
    """Validate at x10 scale if MEWS_LOCKIN_PARQUET_X10 is configured.
    Returns None if x10 validation is not configured (skip silently).

    This catches the silent-x1 pattern: an engine that ignores argv[2]
    passes x1 validation (correct numbers for x1 parquet) but fails x10
    because its row counts don't scale with DuckDB's at the x10 parquet.
    """
    parquet_x10 = os.environ.get("MEWS_LOCKIN_PARQUET_X10", "").strip()
    if not parquet_x10:
        return None
    tenants = os.environ.get("MEWS_LOCKIN_TENANTS", "").strip()
    # x10 takes longer than x1; bump timeout proportionally.
    return _validate_at(workspace, parquet_x10, tenants, timeout=300)


def _quick_bench(workspace: Path, parquet: str, tenant: str,
                 iters: int = 3, timeout: float = 60.0) -> Optional[float]:
    """Wall-clock median in seconds over `iters` engine runs.
    Returns None on any failure (caller treats as bench unavailable).
    """
    engine = workspace / "engine"
    if not engine.exists():
        return None
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        try:
            proc = subprocess.run(
                [str(engine), tenant, parquet],
                cwd=workspace, capture_output=True, text=True, timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return None
        elapsed = time.perf_counter() - t0
        if proc.returncode != 0:
            return None
        times.append(elapsed)
    if not times:
        return None
    return statistics.median(times)


def _load_best_bench(workspace: Path) -> Optional[float]:
    p = workspace / _BEST_BENCH_FILE
    if not p.exists():
        return None
    try:
        return float(json.loads(p.read_text()).get("engine_s_median"))
    except (TypeError, ValueError, KeyError, json.JSONDecodeError):
        return None


def _save_best_bench(workspace: Path, engine_s_median: float) -> None:
    p = workspace / _BEST_BENCH_FILE
    p.write_text(json.dumps({
        "engine_s_median": engine_s_median,
        "saved_at": time.time(),
    }))


def make_lockin_apply_patch_tool(
    root: Path,
    wandb_metrics_hook: WandbRunHook | None = None,
) -> FunctionTool:
    """Construct an apply_patch tool with Mews lock-in semantics.

    Drop-in replacement for upstream's make_litellm_apply_patch_tool when
    the workspace's `.lockin_armed` marker is set. Transparent otherwise.
    """
    impl = LitellmApplyPatchTool(root=root, wandb_metrics_hook=wandb_metrics_hook)

    async def on_invoke(ctx: RunContextWrapper[Any], args_json: str) -> str:
        args = LitellmApplyPatchArgs.model_validate_json(args_json)

        # Fast path: lock-in not armed -> behave exactly like upstream.
        if not _lockin_armed(root):
            return await impl(args.type, args.path, args.diff)

        # Fast path: file isn't gated -> behave exactly like upstream.
        if not _should_gate(args.path):
            return await impl(args.type, args.path, args.diff)

        # Lock-in path: snapshot, apply, recompile, validate, revert if regressed.
        target = root / args.path
        had_file = target.exists()
        snapshot_bytes = target.read_bytes() if had_file else None
        # Persist a labeled snapshot so the trajectory is reconstructable.
        if snapshot_bytes is not None:
            snap = _snapshot_path(root, args.path, "pre")
            snap.write_bytes(snapshot_bytes)
            logger.info(f"lockin: snapshot saved to {snap}")

        # Apply the patch.
        apply_result = await impl(args.type, args.path, args.diff)

        # Recompile.
        compile_ok, compile_out = _recompile(root)
        if not compile_ok:
            # Compile broke -> revert to snapshot, surface failure.
            logger.warning("lockin: compile failed after patch, reverting")
            if snapshot_bytes is not None:
                target.write_bytes(snapshot_bytes)
                _recompile(root)  # restore prior binary; ignore result
            return (
                f"PATCH REVERTED (lockin): compile failed.\n"
                f"--- compile output ---\n{compile_out[:4000]}\n"
                f"--- prior engine.cpp restored from snapshot ---\n"
            )

        def _revert(reason: str, output: str) -> str:
            """Restore snapshot + recompile prior. Returns agent-facing message."""
            logger.warning(f"lockin: {reason}, reverting")
            if snapshot_bytes is not None:
                target.write_bytes(snapshot_bytes)
                rec_ok, _ = _recompile(root)
                if not rec_ok:
                    return (
                        f"PATCH REVERTED (lockin): {reason}, AND restoration of "
                        f"snapshot failed to recompile. Workspace may be in an "
                        f"inconsistent state.\n"
                        f"--- output ---\n{output[:4000]}\n"
                    )
            return (
                f"PATCH REVERTED (lockin): {reason}. The prior engine.cpp + "
                f"binary have been restored. Do not retry the same change. "
                f"Diagnose and try a different fix.\n"
                f"--- output ---\n{output[:4000]}\n"
            )

        # Validate at x1 (existing tenants list).
        ok, val_output = _validate_all(root)
        if not ok:
            return _revert("validate_all.py did not OVERALL: PASS at x1", val_output)

        # Validate at x10 (silent-x1 catch). Skipped silently when not configured.
        x10_result = _validate_at_x10(root)
        if x10_result is not None:
            ok_x10, val_x10_output = x10_result
            if not ok_x10:
                return _revert(
                    "validate_all.py did not OVERALL: PASS at x10 — likely "
                    "scale-dependent bug (e.g. hardcoded parquet path, "
                    "tenant filter, or aggregation that doesn't scale)",
                    val_x10_output,
                )

        # Best-keeper: revert if new engine wall is >5% slower than the last
        # known-best. Opt-in via env (Stage-3 only).
        if os.environ.get("MEWS_LOCKIN_BEST_KEEPER", "0").strip() == "1":
            bench_parquet = os.environ.get("MEWS_LOCKIN_BENCH_PARQUET", "").strip()
            bench_tenant = os.environ.get("MEWS_LOCKIN_BENCH_TENANT", "").strip()
            if bench_parquet and bench_tenant:
                new_med = _quick_bench(root, bench_parquet, bench_tenant)
                if new_med is None:
                    return _revert(
                        "best-keeper: engine bench failed to produce a "
                        "median (engine missing, returned non-zero, or timed "
                        "out at the bench parquet)",
                        f"bench_parquet={bench_parquet} tenant={bench_tenant}",
                    )
                old_med = _load_best_bench(root)
                if old_med is None:
                    # First successful patch under best-keeper -> initialize.
                    _save_best_bench(root, new_med)
                    logger.info(
                        f"lockin: best-keeper initialized to {new_med*1000:.0f}ms"
                    )
                else:
                    if new_med > old_med * BEST_KEEPER_TOLERANCE:
                        return _revert(
                            f"best-keeper: bench regression "
                            f"({old_med*1000:.0f}ms -> {new_med*1000:.0f}ms, "
                            f"+{(new_med/old_med - 1)*100:.1f}% over the "
                            f"{(BEST_KEEPER_TOLERANCE-1)*100:.0f}% tolerance)",
                            f"prior best: {old_med*1000:.0f}ms\n"
                            f"this patch: {new_med*1000:.0f}ms\n"
                            f"3-iter median, parquet={bench_parquet}",
                        )
                    # Equal-or-better -> update the high-water mark.
                    _save_best_bench(root, new_med)
                    logger.info(
                        f"lockin: best-keeper updated "
                        f"({old_med*1000:.0f}ms -> {new_med*1000:.0f}ms)"
                    )

        # All gates held -> persist the new state as the latest known-good snapshot.
        good_snap = _snapshot_path(root, args.path, "lastpass")
        good_snap.write_bytes(target.read_bytes())
        logger.info(f"lockin: {args.path} patch applied, all gates PASS")
        return f"{apply_result}\n[lockin: validate_all.py OVERALL: PASS{' (x1 + x10)' if x10_result else ''}, snapshot updated]"

    return FunctionTool(
        name="apply_patch",
        description="Applies a unified diff to create/update/delete a file",
        params_json_schema=LitellmApplyPatchArgs.model_json_schema(),
        on_invoke_tool=on_invoke,
    )
