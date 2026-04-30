# Setup

## Dependencies

1. **Python 3.11+** with `uv` installed.
2. **Anthropic API key** for Stage-3 LLM-driven synthesis: `export ANTHROPIC_API_KEY=...`.
3. **C++ toolchain**: `g++` (or `clang++`) supporting `-std=c++20`, plus
   `libarrow` + `libparquet` development headers (`pkg-config --cflags --libs arrow parquet` should resolve).
4. **DuckDB** (Python lib via `requirements.txt`; also useful as a CLI for ad-hoc inspection).
5. **The BespokeOLAP fork.** Mews vendors and adapts the [DataManagementLab/BespokeOLAP](https://github.com/DataManagementLab/BespokeOLAP) agent loop.
   Clone it as a sibling of this repo (or anywhere; point `BESPOKE_ROOT` env var at it):

   ```sh
   git clone https://github.com/DataManagementLab/BespokeOLAP.git ./upstream/BespokeOLAP
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
| `OPENAI_API_KEY` | placeholder | Required by openai-agents SDK; any non-empty string works in Anthropic-only setup |

## Data adapters

The adapters in `adapters/` mint OpenInference-shaped parquets from public datasets. Each is standalone:

```sh
cd adapters/wildchat && uv run --with datasets --with pandas --with duckdb python adapt.py
cd adapters/helm     && uv run --with pyarrow --with requests python adapt.py
cd adapters/swerebench && uv run --with pyarrow python adapt.py
# (swerebench adapter expects ./trajectories.parquet pre-downloaded; see adapt.py docstring)
```

Each writes a `spans.parquet` to the adapter's `output/` directory.

## Running a Stage-3 synthesis

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

Exit codes:
- `0` — agent yielded; engine validates clean
- `1` — exception or MaxTurnsExceeded with invalid artifact
- `2` — agent yielded but engine fails validate
- `3` — MaxTurnsExceeded but engine validates clean (substantive success)

## Running the autonomous loop

```sh
# G7-style single-cycle (one drift event):
uv run python examples/g7-loop.py --run-dir /tmp/mews-g7-run

# G8-style sustained loop (N cycles across distinct drift inducers):
uv run python examples/g8-loop.py --run-dir /tmp/mews-g8-run
```

The G7/G8 driver scripts are the deterministic outer orchestrator; Stage-3 is the only LLM-driven step. See `examples/g8-cycle-1/` for what one closed cycle's outputs look like (orchestrator log, manifests, drift detector reports).

## Reproducibility notes

- The 75-turn budget is documented in `gate-7-first-autonomous-regen` and `gate-8-sustained-loop` writeups in the project vault. Two-thirds of cycles close inside that budget on dramatic drift; subtle data-shape drift may not (see [README §"Honest scope"](README.md#honest-scope)).
- The canonical bench parquet sizes (`x10`, `x100`, `x500`) are minted by replicating the base `spans.parquet` with mutated span/parent/trace ids; a minter script will land in this repo when the blog post is written.
- The differential-validation oracle is DuckDB on the same parquet. If you change the locked query in `contracts/openinference/queries/`, the oracle changes accordingly.
