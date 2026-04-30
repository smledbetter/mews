# Mews

**Autonomous bespoke OLAP for LLM observability.** Per-tenant compiled C++ analytic engines, synthesized by an LLM agent, deployed and maintained autonomously: detect drift → regenerate → differential-validate → hot-deploy, no human in the loop.

A research artifact applying the [BespokeOLAP](https://github.com/DataManagementLab/BespokeOLAP) methodology (Wehrstein, Eckmann, Jasny, Binnig — TU Darmstadt — [arXiv:2603.02001](https://arxiv.org/abs/2603.02001)) to LLM-observability span schemas, with a deterministic outer orchestrator wrapping the LLM-driven Stage-3 synthesis.

## Headline results

On the locked OpenInference workload (parent/child trace tree, AGENT→LLM self-join, latency-by-model rollup):

- **2.09× faster than DuckDB-warm** at x100 China (n=15 paired warm-vs-warm, 95% CI [2.03×, 2.31×])
- **5.4× lower RSS** than DuckDB-warm at x100 (engine 383 MB vs DuckDB 2,072 MB)
- **Fits where DuckDB-warm doesn't** — at x500 (20M rows / 1.85 GB parquet) DuckDB-warm OOMs during ingest on a 3.9 GB host; engine streams at 1.78 GB peak
- **Engine binary: ~75 KB** (vs DuckDB shared lib ~50–80 MB — *~700–1000× smaller on disk*)
- **Per-tenant synthesis: $3–5 / 25-min Stage-3 run** (25–40× cheaper than the source paper's $120 / 6–12h)
- **Autonomous loop closes end-to-end** on dramatic drift (silent-engine correctness regression): pre-detect → regen-from-canonical → hot-deploy → post-detect MATCH, in ~28 min wall, no human intervention. 2/2 reproductions across two independent runs.

## Honest scope

The autonomous loop **closes reliably on dramatic drift** — situations where the deployed engine produces categorically wrong output (e.g., silent regression from a hardcoded path) and re-synthesis from a canonical seed trivially produces a correct engine for the new data shape.

It **does not close on subtle data-shape drift** in its current form. Three independent Stage-3 attempts on a 0.06% cardinality delta (induced by id-suffix string-length mutations) all produced engines that fail differential validation. Root cause is architectural: the orchestrator does not pass the drift report into Stage-3's prompt, so the agent must rediscover sub-percent errors in 75 turns from zero. Threading the drift report into the prompt is a clean v2 extension.

The published claim is therefore: *autonomous-loop A closes on dramatic drift; subtle data-shape drift requires drift-report prompt-threading as a v2 extension.* Both halves are real findings.

## What it's good for

- **Memory-bounded LLM-observability self-hosts.** Phoenix / Langfuse / Helicone-shape deploys on a $10–40/mo VPS (Hetzner CX22, Fly small) where one DuckDB-warm process per tenant doesn't fit.
- **Long-tail multi-tenant SaaS.** Free-tier or freemium where per-tenant ARR is too small to give each customer their own DuckDB process — 30 bespoke engines fit in 1.7 GB; 30 DuckDB-warm processes saturate a 4 GB host.
- **Edge / serverless / on-device.** Cloudflare Workers (10 MB module limit), AWS Lambda (cold-start scales with bundle), Raspberry-Pi-class IoT gateways. A 75 KB engine deploys where a 50–80 MB DuckDB shared lib literally cannot.
- **Agent-internal observability.** An agent runtime that wants its own trace analytics without burning 2 GB on a sidecar query engine.
- **Per-corpus eval-result analytics.** HELM-style leaderboards, BIRD, Inspect AI run archives — schema-locked, query templates baked into the harness.

## What it's not for

- A Phoenix-as-a-product replacement. Phoenix uses SQLite (default) or PostgreSQL (production), never DuckDB. Mews benchmarks against DuckDB as the differential-validation oracle and the source paper's baseline; OpenInference (Phoenix's standardized span schema) is the representative real-world workload shape.
- Vendor-economics claims about specific platforms — those need partnership data this corpus doesn't license.
- Ad-hoc analyst exploration — Mews assumes a stable query template, schema-locked input, and per-tenant data partitioning.
- Cross-tenant aggregations — per-tenant compilation works against you when the natural query spans tenants.
- Single fat tenant on a 64+ GB host — just use DuckDB; Mews's lower-RSS isn't load-bearing there.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   Deterministic outer orchestrator (~600 LoC, no LLM here)      │
│                                                                 │
│   ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐      │
│   │   detect     │→ │    regen     │→ │   hot-deploy     │      │
│   │   drift      │  │  (LLM call)  │  │   + re-detect    │      │
│   └──────────────┘  └──────┬───────┘  └──────────────────┘      │
│                            │                                    │
└────────────────────────────┼────────────────────────────────────┘
                             ▼
            ┌──────────────────────────────────┐
            │  Stage-3 LLM synthesis           │
            │   (Claude Opus 4.6 via LiteLLM)  │
            │                                  │
            │   - reads OpenInference parquet  │
            │   - generates engine.cpp         │
            │   - compiles + tests itself      │
            │   - tool-level lockin gates      │
            │     every patch (snapshot →      │
            │     apply → recompile →          │
            │     validate_all.py → revert     │
            │     on regression)               │
            └──────────────────────────────────┘
```

The autonomous *agent* is Stage-3 (the synthesizer); everything around it is deterministic infrastructure.

### Methodology contributions (the prerequisite stack for honest autonomous synthesis)

1. **Tool-level lockin** — `apply_patch` wrapper that snapshots, applies, recompiles, runs `validate_all.py` (multi-tenant differential validation against DuckDB), and reverts on regression. Bespoke specialization is the source of speedup *and* fragility; without lockin, silent correctness regressions are routine.
2. **Silent-x1 catch** — running validate at x1 alongside x10 catches engines that hardcode a parquet path and ignore `argv[2]`. Caught a real silent-engine regression in the Stage-4 four-run comparison.
3. **Best-keeper bench-regression revert** — patches that compile + validate but make the engine slower get reverted with a 5% tolerance. Avoids stochastic "improvement" walks that drift downhill.
4. **Drift detector** — schema/correctness/perf axes; INCONCLUSIVE handling for DuckDB-OOM cases at scales where the oracle isn't usable. 0 false positives across 8 independent inducer cells.
5. **Generic prompt with no architectural hints** — the synthesis prompt does not leak workload-specific code patterns. Without this, measured speedups inflate but the methodology becomes non-transferable. The 2.09× speedup is what survives a generic prompt.
6. **`--arm-lockin-pre`** — for regen from a known-good seed, the lockin marker is set *before* the agent loop runs, so every patch from turn 1 is validate-gated.
7. **Substantive-success exit code (3)** — when Stage-3 hits MaxTurnsExceeded but the engine still validates clean, runner.py exits 3 (vs 0 for clean yield). The orchestrator treats both as success; this catches the common pattern where the agent does correct work but doesn't yield within budget.

## Repo layout

```
infra/                     # core orchestration code
├── runner.py              # Stage-3 launcher + lockin arming + post-validate
├── loop_lib.py            # autonomous-loop primitives (run_one_cycle, etc.)
├── drift_detector.py      # schema/correctness/perf axes
├── validate.py            # single-tenant differential validation
├── validate_all.py        # multi-tenant differential validation (the gate)
├── bench.py               # paired runs + 10K-resample bootstrap CI
├── bench_warm.py          # warm-mode bench harness (sidecar + engine-warm)
├── duckdb_sidecar.py      # long-lived DuckDB-warm process for paired bench
├── smoke_warm.py          # warm-mode N-query smoke test
├── cost_watch.py          # log tail + LiteLLM cost ledger + threshold trip
├── prompts/               # Stage 1/2/3 prompts (no architectural hints)
└── tools/
    ├── lockin_apply_patch.py  # validate-gated apply_patch with auto-revert
    └── resilient.py            # tool wrapper: RuntimeError → recoverable msg

contracts/openinference/   # locked OpenInference query + schema contract
adapters/                  # public-data → OpenInference parquet adapters
├── wildchat/              # 838K real chat conversations (ODC-BY)
├── helm/                  # 104K eval scenario runs (real timestamps)
├── swerebench/            # 67K agent trajectories with native tool calls
├── replicate.py           # mint xN benchmark parquets with unique IDs per copy
└── inducers/mint.py       # mint schema/cardinality drift parquets

examples/                  # reference outputs + driver scripts
├── canonical-engine.cpp   # one Stage-3 output (~11 KB synthesized C++)
├── g7-loop.py             # single-cycle orchestrator driver
├── g8-loop.py             # sustained-loop orchestrator driver
└── g8-cycle-1/            # one closed cycle's full artifact set:
    ├── run.log            #   - human-readable orchestrator trace
    ├── manifest.json      #   - structured cycle outcome
    ├── pre-regen.report.json   #   - drift detector verdict before
    ├── post-regen.report.json  #   - drift detector verdict after
    └── manifest_stage3.json    #   - Stage-3 synthesis metadata
```

## Quickstart

> **This is a reproduction recipe.** Mews depends on the [smledbetter/BespokeOLAP](https://github.com/smledbetter/BespokeOLAP) fork on the `mews-gate-0` branch (10 patches on top of upstream — clone instructions in [setup.md](setup.md)) and on pre-minted `x10` / `x100` / `x500` OpenInference parquets. The adapters in `adapters/` mint a base `spans.parquet`; `adapters/replicate.py` produces the `xN` replicas with unique span/parent/trace ids per copy; `adapters/inducers/mint.py` produces the schema- and cardinality-perturbed parquets the autonomous loop runs against.

Once those prerequisites are in place:

```sh
uv run python examples/g7-loop.py --run-dir /tmp/mews-g7-run
```

The first cycle runs a Stage-3 synthesis (~25–30 min, ~$25–35 in API cost), validates the result, hot-deploys to `/tmp/mews-g7-run/deployed/engine`, and runs a final post-detect to confirm the loop closed.

## Origin and related work

- **BespokeOLAP** (the methodology Mews wraps): [DataManagementLab/BespokeOLAP](https://github.com/DataManagementLab/BespokeOLAP), [arXiv:2603.02001](https://arxiv.org/abs/2603.02001).
- **OpenInference** (the schema target): [Arize-ai/openinference](https://github.com/Arize-ai/openinference), used by Arize Phoenix as its canonical observability span schema.
- **Phoenix** (where OpenInference comes from): [Arize-ai/phoenix](https://github.com/Arize-ai/phoenix). Phoenix uses SQLite (default) or PostgreSQL (production) as its backing store; Mews does not replace Phoenix's storage — it benchmarks a different system (compiled streaming engine vs DuckDB) on Phoenix's schema as a representative workload shape.
- **Statistical scaffold** for correctness rates at scale: Stratified Prediction-Powered Inference ([Fisch et al., arXiv:2406.04291](https://arxiv.org/abs/2406.04291)).

## License

[MIT](LICENSE).

## Citation

If this work is useful in academic context, please cite as:

```
Ledbetter, S. (2026). Mews: Autonomous bespoke OLAP for LLM observability.
https://github.com/smledbetter/mews
```

And the source methodology:

```
Wehrstein, J.-M., Eckmann, M., Jasny, M., & Binnig, C. (2026).
Bespoke OLAP: Synthesizing Workload-Specific One-size-fits-one Database Engines.
arXiv:2603.02001.
```
