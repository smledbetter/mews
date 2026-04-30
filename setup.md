# Setup

## Dependencies

1. **Python 3.11+** with `uv` installed.
2. **Anthropic API key** for Stage-3 LLM-driven synthesis: `export ANTHROPIC_API_KEY=...`.
3. **C++ toolchain**: `g++` (or `clang++`) supporting `-std=c++20`, plus
   `libarrow` + `libparquet` development headers (`pkg-config --cflags --libs arrow parquet` should resolve).
4. **DuckDB** (Python lib via `requirements.txt`; also useful as a CLI for ad-hoc inspection).
5. **The BespokeOLAP fork.** Mews depends on a Mews-specific fork of [DataManagementLab/BespokeOLAP](https://github.com/DataManagementLab/BespokeOLAP), published at [smledbetter/BespokeOLAP](https://github.com/smledbetter/BespokeOLAP) on the `mews-gate-0` branch (10 patches on top of upstream merge-base `8f077ff825aec56f08ed5290251a53706b458258`). The patches add the lockin marker plumbing, the `--arm-lockin-pre` flag, and the substantive-success exit code (3) handling that Mews's `runner.py` imports — a clean clone of upstream without these patches will fail to satisfy the imports at the top of `infra/runner.py` and `infra/tools/lockin_apply_patch.py`.

   Clone it as a sibling of this repo (or anywhere; point `BESPOKE_ROOT` env var at it):

   ```sh
   git clone -b mews-gate-0 https://github.com/smledbetter/BespokeOLAP.git ./upstream/BespokeOLAP
   ```

   The Python deps Mews relies on from BespokeOLAP are the agents-SDK shell tool, the LiteLLM model wrapper, and the apply_patch tool — all imported at runtime from `BESPOKE_ROOT`.

## Install

```sh
pip install -r requirements.txt
# or, with uv:
uv pip install -r requirements.txt
```

## Environment

Useful env vars (all optional, with sensible defaults):

| Var | Default | Purpose |
|---|---|---|
| `MEWS_ROOT` | derived from `__file__` | Repo root (override if running from outside the tree) |
| `BESPOKE_ROOT` | `$MEWS_ROOT/upstream/BespokeOLAP` | BespokeOLAP fork location |
| `UV_BIN` | `uv` (assumes on PATH) | Path to the uv binary |
| `ANTHROPIC_API_KEY` | — required for Stage-3 — | LiteLLM/Anthropic auth |
| `OPENAI_API_KEY` | placeholder | The openai-agents SDK initializes an OpenAI client at import time even when routing to Anthropic via LiteLLM; any non-empty string satisfies that import. Mews never makes an OpenAI API call. |

## Data adapters

The adapters in `adapters/` mint OpenInference-shaped parquets from public datasets. Each is standalone:

```sh
cd adapters/wildchat && uv run --with datasets --with pandas --with duckdb python adapt.py
cd adapters/helm     && uv run --with pyarrow --with requests python adapt.py
cd adapters/swerebench && uv run --with pyarrow python adapt.py
# (swerebench adapter expects ./trajectories.parquet pre-downloaded; see adapt.py docstring)
```

Each writes a `spans.parquet` to the adapter's `output/` directory.

### `xN` benchmark replicas

The autonomous-loop benches need `x10` / `x100` / `x500` replicas of the base `spans.parquet`. `adapters/replicate.py` streams a parquet through with `_k{idx}` suffixes on `span_id`, `parent_id`, and `trace_id` per copy, so the locked self-join doesn't cross-product across replicas (the join on `(parent_id, tenant_id)` stays within-copy).

```sh
uv run --with pyarrow python adapters/replicate.py \
    --in adapters/wildchat/output/spans.parquet \
    --out adapters/wildchat/output/spans-x100.parquet \
    --factor 100
```

### Drift inducers

`adapters/inducers/mint.py` produces the four drift parquets the autonomous loop runs against (extra column, renamed tenant column, type-cast `start_time`, China-10x cardinality bump) plus a `baseline-schema.json` snapshot the drift detector uses as the synthesis-time reference.

```sh
uv run --with pyarrow python adapters/inducers/mint.py \
    --src adapters/wildchat/output/spans.parquet \
    --out-dir adapters/wildchat/output/inducers/ \
    --prefix wildchat-x1
```

## Running a Stage-3 synthesis

> **Reproduction recipe.** Stage-3 runs against an OpenInference-shaped parquet that is not in this repo (mint via `adapters/`). The autonomous-loop benches additionally need pre-minted `x10` / `x100` / `x500` replicas via `adapters/replicate.py` (see "`xN` benchmark replicas" above).

```sh
export MEWS_ROOT=/path/to/mews
export BESPOKE_ROOT=/path/to/upstream/BespokeOLAP
export ANTHROPIC_API_KEY=sk-ant-...

uv run python infra/runner.py \
    --stage 3 \
    --tenant 'China' \
    --workspace /tmp/mews-stage3-ws \
    --log-dir /tmp/mews-stage3-log \
    --max-turns 75 \
    --model litellm/anthropic/claude-opus-4-6 \
    --parquet /path/to/spans.parquet \
    --validation-tenants 'China,United States,Russia' \
    --bench-parquet /path/to/spans-x100.parquet \
    --arm-lockin-pre
```

Notes on the invocation:

- **Stage-3 runs with `cwd=$BESPOKE_ROOT`**, not the Mews repo root. The agent loop, lockin marker, and apply_patch tool all resolve paths relative to the BespokeOLAP fork; the Mews orchestrator hands it absolute paths into `--workspace`.
- **`--validation-tenants` values are dataset-specific.** `China,United States,Russia` is the locked tenant triple for the **WildChat** OpenInference-shaped parquet — three high-volume, well-distributed tenants used for differential validation. The HELM and SWE-rebench adapters partition along different axes (model name, repo) — pick three high-volume partition values from whichever `spans.parquet` you minted.

Exit codes:
- `0` — agent yielded; engine validates clean
- `1` — exception or MaxTurnsExceeded with invalid artifact
- `2` — agent yielded but engine fails validate
- `3` — MaxTurnsExceeded but engine validates clean (substantive success)

## Running the autonomous loop

Same caveat as above — these depend on pre-minted `x10` / `x100` / `x500` parquets that aren't in the repo yet.

```sh
# G7-style single-cycle (one drift event):
uv run python examples/g7-loop.py --run-dir /tmp/mews-g7-run

# G8-style sustained loop (N cycles across distinct drift inducers):
uv run python examples/g8-loop.py --run-dir /tmp/mews-g8-run
```

The G7/G8 driver scripts are the deterministic outer orchestrator; Stage-3 is the only LLM-driven step. See `examples/g8-cycle-1/` for what one closed cycle's outputs look like (orchestrator log, manifests, drift detector reports) — that artifact set is the closest thing to a "what does success look like" reference until the fork is published.

## Reproducibility notes

- The 75-turn budget is documented in `gate-7-first-autonomous-regen` and `gate-8-sustained-loop` writeups in the project vault. Two-thirds of cycles close inside that budget on dramatic drift; subtle data-shape drift may not (see [README §"Honest scope"](README.md#honest-scope)).
- The canonical bench parquet sizes (`x10`, `x100`, `x500`) are minted by `adapters/replicate.py`, which suffixes span/parent/trace ids per copy.
- The differential-validation oracle is DuckDB on the same parquet. If you change the locked query in `contracts/openinference/queries/`, the oracle changes accordingly.
