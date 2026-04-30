"""Mews Gate-1 synthesis runner.

A minimal Mews-owned harness that drives a three-stage synthesis (storage plan,
base implementation, optim loop) of a C++ engine for the locked OpenInference
reference query. Reuses BespokeOLAP's vendored apply_patch + shell tool wrappers
from the mews-gate-0 branch (which already carry the 8000-char shell fix and
the LitellmModel adapter) but supplies a Mews-native prompt sequence and runner
loop.

CLI:
    python runner.py --stage {1,2,3} --tenant tenant-01 \
        --workspace ~/projects/mews/experiments/gate-1/run-1/output \
        --log-dir   ~/projects/mews/experiments/gate-1/run-1 \
        [--max-turns 75] [--model litellm/anthropic/claude-opus-4-6] [--dry-run]

Reads:
    contracts/openinference/schema.sql
    contracts/openinference/schema.json
    contracts/openinference/queries/latency_by_model_under_agent.sql
    artifacts/openinference_parquet/sf1/spans.parquet
    infra/prompts/system.txt
    infra/prompts/stage_storage_plan.txt   (stage 1)
    infra/prompts/stage_base_impl.txt      (stage 2)
    infra/prompts/stage_optim.txt          (stage 3)

Writes:
    {log-dir}/run.log                      (full log including the agent loop)
    {log-dir}/prompts.jsonl                (every prompt + response, JSONL)
    {log-dir}/manifest.json                (run pin: model, contract hash, timestamps)
    {workspace}/...                        (whatever the agent creates)
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Make BespokeOLAP imports available. Default: ../upstream/BespokeOLAP
# relative to this file; override with BESPOKE_ROOT env var if vendored
# elsewhere.
_REPO_ROOT = Path(__file__).resolve().parent.parent
BESPOKE_ROOT = Path(
    os.environ.get("BESPOKE_ROOT", str(_REPO_ROOT / "upstream" / "BespokeOLAP"))
).resolve()
sys.path.insert(0, str(BESPOKE_ROOT))

# Disable openai-agents tracing client.
# Reason: with a placeholder OPENAI_API_KEY (Anthropic-only setup), the
# tracing client retry-loops on 401 and can deadlock the agent process
# at exit (observed in Gate-1 run-1 stage 3, ~10min wall-clock hang).
os.environ.setdefault("OPENAI_AGENTS_DISABLE_TRACING", "1")

from agents import Agent, Runner  # type: ignore
from agents.exceptions import MaxTurnsExceeded  # type: ignore
from agents.extensions.models.litellm_model import LitellmModel  # type: ignore
from tools.litellm_shell import make_litellm_shell_tool  # type: ignore

# Mews-owned apply_patch wrapper. Replaces upstream's tool with the same
# `apply_patch` name + JSON schema; fast-paths to upstream behavior unless
# the workspace's `.lockin_armed` marker is set (post-stage-2 multi-tenant
# PASS), at which point engine.cpp edits are validate_all.py-gated with
# automatic revert on regression.
sys.path.insert(0, str(_REPO_ROOT))
from infra.tools.lockin_apply_patch import make_lockin_apply_patch_tool  # type: ignore
from infra.tools.resilient import make_resilient_tool  # type: ignore


# Per-token pricing for the synthesis model. Selected by model substring at
# instantiation time. If a new model is added, drop a row in MODEL_PRICING.
MODEL_PRICING = {
    # (input, output, cache_read) per token, USD
    "opus-4-6":    (5e-6,  25e-6,  0.5e-6),
    "sonnet-4-6":  (3e-6,  15e-6,  0.3e-6),
    "haiku-4-5":   (0.8e-6, 4e-6,  0.08e-6),
}
_DEFAULT_PRICING = MODEL_PRICING["opus-4-6"]


def _pricing_for(model_name: str) -> tuple[float, float, float]:
    for key, prices in MODEL_PRICING.items():
        if key in model_name:
            return prices
    return _DEFAULT_PRICING


class MewsLitellmModel(LitellmModel):
    """LitellmModel wrapper that logs per-call cost lines under the
    `llm_cache.cached_litellm` logger name so the existing infra/cost_watch.py
    regex picks them up. Pricing is selected from MODEL_PRICING by substring
    match on the model id at construction.
    """

    _cost_logger = logging.getLogger("llm_cache.cached_litellm")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        model_name = kwargs.get("model") or (args[0] if args else "")
        self._price_in, self._price_out, self._price_cache = _pricing_for(model_name)
        self._cost_logger.info(
            f"pricing for {model_name}: in=${self._price_in*1e6:.2f}/M  "
            f"out=${self._price_out*1e6:.2f}/M  cache=${self._price_cache*1e6:.2f}/M"
        )

    async def get_response(self, *args, **kwargs):
        resp = await super().get_response(*args, **kwargs)
        try:
            usage = getattr(resp, "usage", None)
            if usage is not None:
                input_tok = getattr(usage, "input_tokens", 0) or 0
                output_tok = getattr(usage, "output_tokens", 0) or 0
                cached_in = 0
                details = getattr(usage, "input_tokens_details", None)
                if details is not None:
                    if hasattr(details, "model_dump"):
                        cached_in = details.model_dump().get("cached_tokens", 0) or 0
                    elif hasattr(details, "cached_tokens"):
                        cached_in = details.cached_tokens or 0
                uncached_in = max(0, input_tok - cached_in)
                cost = (
                    uncached_in * self._price_in
                    + cached_in * self._price_cache
                    + output_tok * self._price_out
                )
                self._cost_logger.debug(f"Cost: ${cost:0.6f}")
        except Exception as e:
            self._cost_logger.warning(f"could not compute per-call cost: {e}")
        return resp




PROJECT_ROOT = Path(os.environ.get("MEWS_ROOT", str(_REPO_ROOT))).resolve()
CONTRACT_DIR = PROJECT_ROOT / "contracts" / "openinference"
INFRA_DIR    = PROJECT_ROOT / "infra"
PROMPTS_DIR  = INFRA_DIR / "prompts"
PARQUET_PATH = PROJECT_ROOT / "artifacts" / "openinference_parquet" / "sf1" / "spans.parquet"

DEFAULT_MODEL = "litellm/anthropic/claude-opus-4-6"
DEFAULT_MAX_TURNS = 75



def _build_substitutions(ctx) -> dict:
    schema_sql = (ctx.contract_dir / "schema.sql").read_text()
    ref_query  = (ctx.contract_dir / "queries" / "latency_by_model_under_agent.sql").read_text()
    schema_json = (ctx.contract_dir / "schema.json").read_text()
    val_tenants = ctx.validation_tenants or [ctx.tenant]
    return {
        "TENANT_ID":             ctx.tenant,
        "PARQUET_PATH":          str(ctx.parquet_path),
        "BENCH_PARQUET_PATH":    str(getattr(ctx, "bench_parquet_path", ctx.parquet_path)),
        "WORKSPACE":             str(ctx.workspace),
        "MAX_TURNS":             str(ctx.max_turns),
        "SCHEMA_SQL":            schema_sql,
        "REFERENCE_QUERY_SQL":   ref_query,
        "SCHEMA_JSON":           schema_json,
        "CONTRACT_DIR":          str(ctx.contract_dir),
        "MEWS_ROOT":             str(PROJECT_ROOT),
        # Validation tenants threaded through the prompt + validator invocation.
        # VALIDATION_TENANTS:      machine-readable comma-separated for `--tenants`
        # VALIDATION_TENANTS_LIST: human-readable comma+space form for prose
        # VALIDATION_TENANTS_COUNT: count for "all N tenants" phrasing
        "VALIDATION_TENANTS":       ",".join(val_tenants),
        "VALIDATION_TENANTS_LIST":  ", ".join(val_tenants),
        "VALIDATION_TENANTS_COUNT": str(len(val_tenants)),
    }


def _apply_subs(template: str, subs: dict) -> str:
    out = template
    for k, v in subs.items():
        out = out.replace("{{" + k + "}}", v)
    return out


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _setup_logging(log_path: Path) -> None:
    # Logging path: StreamHandler -> stdout -> launch script's `tee -a $LOG`.
    # log_path is kept as an arg for the manifest record but no FileHandler is
    # added — duplicating to FileHandler caused 2× ledger overcounting in
    # Gate 1 runs 1-3.
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s %(levelname)s:%(name)s:%(message)s"
    logging.basicConfig(
        level=logging.DEBUG,
        format=fmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )


@dataclass
class StageContext:
    stage: int
    tenant: str
    workspace: Path
    contract_dir: Path
    parquet_path: Path
    model: str
    max_turns: int
    validation_tenants: list[str] = field(default_factory=list)  # tenants for validate_all.py; empty -> Gate-1 default
    parquet_path_x10: Optional[Path] = None  # if set, lockin validates at x10 too (silent-x1 catch)
    bench_parquet_path: Optional[Path] = None  # if set + stage 3, lockin runs best-keeper bench at this scale
    bench_tenant: Optional[str] = None  # tenant id for best-keeper bench (defaults to ctx.tenant)
    stage3_prompt: Optional[str] = None  # override stage-3 prompt filename inside PROMPTS_DIR (default stage_optim.txt)
    arm_lockin_pre: bool = False  # if True, write .lockin_armed BEFORE the agent loop starts (use when starting from a known-good engine)


def _build_agent(ctx, cache_dir: Path) -> Agent:
    workspace = ctx.workspace
    model = ctx.model
    """Construct an Agent with apply_patch + shell tools rooted at workspace.

    cache_dir is a Mews-managed scratch path for the shell tool's pickle cache.
    Snapshotter is None in v0 — we rely on the experiments/gate-1/run-K/
    directory itself as the snapshot boundary.
    """
    workspace.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    apply_patch = make_lockin_apply_patch_tool(root=workspace, wandb_metrics_hook=None)
    shell = make_litellm_shell_tool(
        cwd=workspace,
        cache_dir=cache_dir,
        git_snapshotter=None,
        wandb_metrics_hook=None,
    )
    # Wrap both tools so sandbox refusals (sudo, workspace-boundary) and
    # apply_patch context-mismatch errors come back to the agent as
    # recoverable strings instead of aborting the run via UserError.
    apply_patch = make_resilient_tool(apply_patch)
    shell = make_resilient_tool(shell)
    system_prompt = _apply_subs((PROMPTS_DIR / "system.txt").read_text(), _build_substitutions(ctx))

    # The model arg here uses LiteLLM's "<provider>/<model>" form;
    # the runner accepts either "litellm/<provider>/<model>" or bare "<provider>/<model>"
    bare_model = model.removeprefix("litellm/")
    api_key = (
        os.environ.get("LITELLM_API_KEY")
        or os.environ.get("ANTHROPIC_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )
    return Agent(
        name="Mews Gate-1 synthesizer",
        instructions=system_prompt,
        model=MewsLitellmModel(model=bare_model, api_key=api_key),
        tools=[apply_patch, shell],
    )


def _stage_prompt(stage: int, ctx: StageContext) -> str:
    default_fname = {1: "stage_storage_plan.txt", 2: "stage_base_impl.txt", 3: "stage_optim.txt"}[stage]
    fname = ctx.stage3_prompt if (stage == 3 and ctx.stage3_prompt) else default_fname
    template = (PROMPTS_DIR / fname).read_text()

    # Substitute {{KEY}} placeholders

    return _apply_subs(template, _build_substitutions(ctx))


def _record_manifest(log_dir: Path, ctx: StageContext, started_at: str, finished_at: str, exit_code: int) -> None:
    manifest = {
        "stage": ctx.stage,
        "tenant": ctx.tenant,
        "model": ctx.model,
        "max_turns": ctx.max_turns,
        "workspace": str(ctx.workspace),
        "started_at": started_at,
        "finished_at": finished_at,
        "exit_code": exit_code,
        "contract_hash": {
            "schema.sql":  _hash_file(ctx.contract_dir / "schema.sql"),
            "schema.json": _hash_file(ctx.contract_dir / "schema.json"),
            "reference_query.sql": _hash_file(ctx.contract_dir / "queries" / "latency_by_model_under_agent.sql"),
        },
        "parquet_hash": _hash_file(ctx.parquet_path),
    }
    (log_dir / f"manifest_stage{ctx.stage}.json").write_text(json.dumps(manifest, indent=2))


async def _run_stage(ctx: StageContext, log_dir: Path, dry_run: bool) -> int:
    log = logging.getLogger("mews.runner")

    # Cache dir scoped to this run
    cache_dir = log_dir / "tool_cache"

    agent = _build_agent(ctx, cache_dir)
    prompt = _stage_prompt(ctx.stage, ctx)
    log.info("=" * 80)
    log.info(f"Mews Gate-1 stage {ctx.stage} | model={ctx.model} | tenant={ctx.tenant}")
    log.info(f"workspace={ctx.workspace} | max_turns={ctx.max_turns}")
    log.info("=" * 80)
    log.info("PROMPT:")
    log.info(prompt[:8000])  # cap log line at 8KB to keep run.log readable
    log.info("=" * 80)

    # Pre-arm lockin BEFORE the agent loop runs. Required when Stage-N starts
    # from a known-good engine (the post-success arming below would leave
    # in-loop apply_patch ungated for the entire run).
    # When best-keeper is also enabled, seed the baseline by benching the
    # starting engine so the agent's first patch is compared against the
    # *starting* engine's wall, not against itself.
    if ctx.arm_lockin_pre:
        marker = ctx.workspace / ".lockin_armed"
        marker.touch()
        log.info(f"lockin pre-armed: {marker} (caller asserts engine is known-good at start)")
        if ctx.stage == 3 and ctx.bench_parquet_path:
            from infra.tools.lockin_apply_patch import _quick_bench, _save_best_bench
            best_path = ctx.workspace / ".lockin_best_bench.json"
            if not best_path.exists():
                bench_tenant = ctx.bench_tenant or ctx.tenant
                seed_med = _quick_bench(ctx.workspace, str(ctx.bench_parquet_path), bench_tenant)
                if seed_med is not None:
                    _save_best_bench(ctx.workspace, seed_med)
                    log.info(f"lockin pre-armed: best-keeper seeded at {seed_med*1000:.0f}ms (3-iter median, tenant={bench_tenant})")
                else:
                    log.warning("lockin pre-armed: best-keeper seed bench failed; first successful patch will set the baseline")

    if dry_run:
        log.info("DRY RUN: skipping LLM call. Prompt assembled successfully.")
        return 0

    started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    prompts_path = log_dir / "prompts.jsonl"
    with prompts_path.open("a") as pf:
        pf.write(json.dumps({"stage": ctx.stage, "role": "user", "ts": started_at, "text": prompt}) + "\n")

    # Surface the parquet + validation tenants to the lockin_apply_patch
    # tool so its in-loop validate_all.py invocations target the correct
    # data, not the synthetic Gate-1 fallback fixture.
    os.environ["MEWS_LOCKIN_PARQUET"] = str(ctx.parquet_path)
    os.environ["MEWS_LOCKIN_TENANTS"] = ",".join(ctx.validation_tenants) if ctx.validation_tenants else ""

    # Silent-x1 catch: when a x10 replica is configured, lockin validates
    # at x10 in addition to x1. An engine that ignores argv[2] passes x1
    # but its row counts won't scale at x10 (DuckDB reference will).
    if ctx.parquet_path_x10:
        os.environ["MEWS_LOCKIN_PARQUET_X10"] = str(ctx.parquet_path_x10)
    else:
        os.environ.pop("MEWS_LOCKIN_PARQUET_X10", None)

    # Best-keeper: only meaningful at Stage 3 (specialization is
    # speed-driven; Stage 2 builds correctness-first).
    if ctx.stage == 3 and ctx.bench_parquet_path:
        os.environ["MEWS_LOCKIN_BEST_KEEPER"] = "1"
        os.environ["MEWS_LOCKIN_BENCH_PARQUET"] = str(ctx.bench_parquet_path)
        os.environ["MEWS_LOCKIN_BENCH_TENANT"] = ctx.bench_tenant or ctx.tenant
    else:
        os.environ.pop("MEWS_LOCKIN_BEST_KEEPER", None)
        os.environ.pop("MEWS_LOCKIN_BENCH_PARQUET", None)
        os.environ.pop("MEWS_LOCKIN_BENCH_TENANT", None)

    max_turns_exceeded = False
    result = None
    try:
        result = await Runner.run(
            agent,
            input=prompt,
            max_turns=ctx.max_turns,
        )
    except MaxTurnsExceeded as e:
        log.warning(f"stage {ctx.stage}: MaxTurnsExceeded ({e}); will run post-stage validate to check substantive correctness")
        max_turns_exceeded = True
    except Exception as e:
        finished_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        log.error(f"stage {ctx.stage} failed: {type(e).__name__}: {e}")
        _record_manifest(log_dir, ctx, started_at, finished_at, exit_code=1)
        return 1

    finished_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    final_text = (
        (getattr(result, "final_output", None) or str(result))
        if result is not None
        else "(MaxTurnsExceeded -- no final_output produced)"
    )
    with prompts_path.open("a") as pf:
        pf.write(json.dumps({"stage": ctx.stage, "role": "assistant", "ts": finished_at, "text": str(final_text)}) + "\n")

    log.info("=" * 80)
    log.info(f"stage {ctx.stage} FINAL OUTPUT (max_turns_exceeded={max_turns_exceeded}):")
    log.info(str(final_text)[:8000])
    log.info("=" * 80)

    # Programmatic post-stage validation. Stages 2 and 3 must produce a
    # multi-tenant-correct engine; the runner verifies this regardless of
    # what the agent printed -- including when the agent hit
    # MaxTurnsExceeded without a clean yield. Stage 1 has no engine.
    if ctx.stage in (2, 3):
        validate_ok = _post_stage_validate(ctx, log)
        if not validate_ok:
            if max_turns_exceeded:
                log.error(f"stage {ctx.stage}: MaxTurnsExceeded AND validate_all.py FAIL -- genuine failure")
                _record_manifest(log_dir, ctx, started_at, finished_at, exit_code=1)
                return 1
            log.error(f"stage {ctx.stage} agent reported success but validate_all.py did not OVERALL: PASS")
            _record_manifest(log_dir, ctx, started_at, finished_at, exit_code=2)
            return 2

        # validate PASS -- arm lockin + capture schema snapshot regardless of
        # whether the agent yielded cleanly or hit MaxTurnsExceeded. Both
        # paths produced a substantively-correct artifact.
        marker = ctx.workspace / ".lockin_armed"
        if not marker.exists():
            marker.touch()
            log.info(f"lockin armed: {marker} (tool-level apply_patch enforcement is now active)")

        # Capture a synthesis-time parquet schema snapshot. Used by the
        # Gate-6 drift detector as the schema-drift baseline; without it,
        # post-deploy drift detection has to be told the baseline out-of-
        # band. Cheap to compute (one parquet metadata read) and idempotent
        # within a single workspace.
        schema_snap = ctx.workspace / ".synthesis_schema.json"
        if not schema_snap.exists():
            try:
                import pyarrow.parquet as _pq
                _schema = _pq.read_schema(ctx.parquet_path)
                schema_snap.write_text(json.dumps({
                    "fields": [
                        {"name": f.name, "type": str(f.type), "nullable": bool(f.nullable)}
                        for f in _schema
                    ],
                    "captured_at": finished_at,
                    "source_parquet": str(ctx.parquet_path),
                }, indent=2))
                log.info(f"synthesis schema snapshot: {schema_snap}")
            except Exception as e:  # noqa: BLE001
                log.warning(f"failed to capture synthesis schema snapshot: {type(e).__name__}: {e}")

        if max_turns_exceeded:
            log.info(f"stage {ctx.stage}: substantive success -- MaxTurnsExceeded but artifact validates clean (exit code 3)")
            _record_manifest(log_dir, ctx, started_at, finished_at, exit_code=3)
            return 3

    _record_manifest(log_dir, ctx, started_at, finished_at, exit_code=0)
    return 0


def _post_stage_validate(ctx: "StageContext", log: logging.Logger) -> bool:
    """Run validate_all.py against the workspace and return True if OVERALL: PASS."""
    cmd = [
        os.environ.get("UV_BIN", "uv"), "run", "python",
        str(INFRA_DIR / "validate_all.py"),
        "--workspace", str(ctx.workspace),
        "--parquet",   str(ctx.parquet_path),
    ]
    if ctx.validation_tenants:
        cmd += ["--tenants", ",".join(ctx.validation_tenants)]
    env = dict(os.environ)
    env["UV_CACHE_DIR"] = env.get("UV_CACHE_DIR", "/tmp/uv_cache")
    log.info(f"post-stage validate: {' '.join(cmd)}")
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120, env=env)
    except subprocess.TimeoutExpired:
        log.error("post-stage validate: timeout (120s)")
        return False
    log.info(f"validate_all.py exit={proc.returncode}")
    if proc.stdout:
        log.info(f"validate_all.py stdout:\n{proc.stdout}")
    if proc.stderr:
        log.info(f"validate_all.py stderr:\n{proc.stderr}")
    return proc.returncode == 0 and "OVERALL: PASS" in proc.stdout


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", type=int, required=True, choices=[1, 2, 3])
    p.add_argument("--tenant", type=str, default="tenant-01")
    p.add_argument("--workspace", type=Path, required=True)
    p.add_argument("--log-dir",   type=Path, required=True)
    p.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS)
    p.add_argument("--model",     type=str, default=DEFAULT_MODEL)
    p.add_argument("--parquet",   type=Path, default=PARQUET_PATH)
    p.add_argument("--parquet-x10", type=Path, default=None,
                   help="x10 replica parquet for the silent-x1 catch in lockin. "
                        "When set, lockin runs validate_all.py at x10 in addition "
                        "to x1; engines that ignore argv[2] pass x1 but fail x10.")
    p.add_argument("--bench-parquet", type=Path, default=None,
                   help="parquet for best-keeper bench. When set with --stage 3, "
                        "lockin runs a 3-iter engine bench after each successful "
                        "patch and reverts if new median is >5%% slower than the "
                        "last known-best.")
    p.add_argument("--bench-tenant", type=str, default=None,
                   help="tenant_id used for the best-keeper bench. Defaults to --tenant.")
    p.add_argument("--validation-tenants", type=str, default="",
                   help="comma-separated tenant_ids passed to validate_all.py. "
                        "Empty = use validate_all.py's Gate-1 defaults (tenant-01,02,03).")
    p.add_argument("--stage3-prompt", type=str, default=None,
                   help="filename inside infra/prompts/ to use as the stage-3 prompt. "
                        "Defaults to stage_optim.txt. Use 'stage_optim_warm.txt' for the "
                        "engine-warm dual-mode synthesis variant.")
    p.add_argument("--arm-lockin-pre", action="store_true",
                   help="Write .lockin_armed BEFORE the agent loop starts. Use when "
                        "Stage-N begins from a known-good engine that should be lockin-"
                        "protected from the first patch. When --bench-parquet is also "
                        "set on Stage-3, the best-keeper baseline is seeded by benching "
                        "the starting engine.")
    p.add_argument("--dry-run",   action="store_true")
    args = p.parse_args(argv)

    args.log_dir.mkdir(parents=True, exist_ok=True)
    args.workspace.mkdir(parents=True, exist_ok=True)
    _setup_logging(args.log_dir / "run.log")

    val_tenants = [t.strip() for t in args.validation_tenants.split(",") if t.strip()]
    ctx = StageContext(
        stage=args.stage,
        tenant=args.tenant,
        workspace=args.workspace.resolve(),
        contract_dir=CONTRACT_DIR,
        parquet_path=args.parquet.resolve(),
        model=args.model,
        max_turns=args.max_turns,
        validation_tenants=val_tenants,
        parquet_path_x10=args.parquet_x10.resolve() if args.parquet_x10 else None,
        bench_parquet_path=args.bench_parquet.resolve() if args.bench_parquet else None,
        bench_tenant=args.bench_tenant,
        stage3_prompt=args.stage3_prompt,
        arm_lockin_pre=args.arm_lockin_pre,
    )
    return asyncio.run(_run_stage(ctx, args.log_dir.resolve(), args.dry_run))


if __name__ == "__main__":
    sys.exit(main())
