# g8-cycle-1: artifacts from one closed loop cycle (and one failed one)

This directory shows the **full G8 run-1 sequence** — not a curated single
clean cycle. Read alongside `manifest.json`'s `status` field.

What the artifacts demonstrate:

- **Cycle 1 — `cycle-1-correctness` — `cycle_closed: true`.** This is the
  positive evidence: drift detector fires on a deployed silent engine
  (categorically wrong output on x10), Stage-3 regenerates from canonical,
  hot-deploy, post-detect MATCH on all axes (engine 168 ms vs DuckDB 626 ms,
  ratio 0.27, schema MATCH, correctness MATCH). 27.6 min wall, $31.35.
  *This is what the README means by "the loop closes on dramatic drift."*

- **`baseline-1` — `cycle_closed: true`, `pre_regen_recommended: false`.**
  No-drift idle check. Detector confirms the regenerated engine is clean
  on the same parquet — no false positive. 37 s wall, $0.

- **Cycle 2 — `cycle-2-cardinality` — `cycle_closed: false`,
  `regen_artifact_ok: false`.** This is the **documented failure case** the
  README's "Honest scope" section describes. Drift detector correctly
  identifies a 0.06% delta on the China tenant from a v2 id-suffix
  cardinality inducer (`china-10x.parquet`); Stage-3 runs the full 75-turn
  budget but produces an engine that fails differential validation. The
  orchestrator does not pass the drift report into Stage-3's prompt, so
  the agent must rediscover the sub-percent error on its own and apparently
  cannot in 75 turns. The orchestrator kills the cycle (exit code 1 in the
  manifest at `regen_exit_code: 1`).

- **Manifest `status: "killed_at_cycle_2"`.** This is honest, expected, and
  documented — *not* a bug or a regression. The orchestrator design
  correctly refuses to hot-deploy a non-validating engine.

## What's in each file

| File | What it is |
|---|---|
| `run.log` | Human-readable orchestrator trace. Phase-by-phase timeline of all three cycles + baseline check. The most readable single artifact. |
| `manifest.json` | Structured per-cycle outcome — wall time, regen cost, drift detector reports before/after, cycle-closure verdict. |
| `pre-regen.report.json` | Drift detector verdict for cycle-1 *before* regen (the DRIFT signal that triggered Stage-3). |
| `post-regen.report.json` | Drift detector verdict for cycle-1 *after* regen + hot-deploy (the MATCH that closes the cycle). |
| `manifest_stage3.json` | Stage-3 synthesis metadata: model, exit code, wall, lockin state. |

## What's not here

- Stage-3 `regen-stdout.log` (~106 MB of LiteLLM debug traffic + agent
  transcript). Excluded for size; the structured artifacts above carry
  the load-bearing signal.
- `regen-ws/` (the actual engine.cpp + binary the agent produced). Not
  reproducible from the public repo today; see project README "Honest
  scope" section.
- The cycle-2 retry attempt artifacts (also failed; lives in `run-2/` in
  the original experiment tree but not extracted here — same story as
  cycle-2 primary).

## Why this is in the public repo

To prove the loop-closure claim is honestly reported. A casual reader can
verify cycle-1's `cycle_closed: true` against the run.log timestamps
($31.35 / 27.6 min / engine 168ms vs DuckDB 626ms), and verify the
documented failure mode against cycle-2's `regen_artifact_ok: false`. The
artifact set matches the README's claims line-for-line.
