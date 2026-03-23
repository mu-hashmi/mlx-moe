# Agent Instructions

Instructions for coding agents working on this codebase.

## What This Project Is

MLX-MoE runs large MoE models on memory-constrained Macs by loading only routed experts from SSD into Metal memory. The primary target is `mlx-community/Qwen3-Coder-Next-4bit` (512 experts/layer, 48 MoE layers).

## Project Structure

```text
mlx_moe/                      # Package
  __init__.py                 # Exports: generate, stream_generate, Session
  cli.py                      # CLI entrypoint: mlx-moe serve
  server.py                   # OpenAI + Anthropic API server
  lazy_experts/
    core.py                   # enable/upgrade/reset, cache stats, dynamic updates
    modules.py                # Lazy/Cached/Predictive switch modules + caches
    loading.py                # Capacity selection, shard maps, selective loading
    discovery.py              # Router-only discovery
    warmup.py                 # Delta warmup and warmup helpers
    persistence.py            # Cache state, profiles, prepacked weights
    generate.py               # generate, stream_generate, Session, _startup

benchmarks/
  test_model.py               # Quick local smoke run (throughput + fallback)
  validate_quality.py         # Quality/warmup/memory/adaptive experiments
  profile_experts.py          # Build expert profiles (diverse/coding/tool-chat/mixed)
  benchmark_mlx_server.py     # mlx-moe-only server benchmark with streamed output
  benchmark_backends.py       # llama.cpp vs mlx-moe comparison (optional)
  sweep_profile_pinning.py    # Mix/pin sweep using real tool-chat traffic
  tool_chat_scenarios.py      # Tool schemas + agentic prompt scenarios

tests/
  test_unit_core.py
  test_unit_persistence.py
  test_integration.py         # Synthetic MoE + server endpoint tests

profiles/                     # Checked-in profile JSONs
logs/                         # Local benchmark/sweep outputs (gitignored)
docs/README.md                # High-level architecture overview
```

Depends on stock `mlx-lm >= 0.30.0` (no fork).

## Dev Setup

This is a uv project (`pyproject.toml`, Python 3.13).

```bash
uv sync
uv run python -c "from mlx_moe import generate; print('ok')"
uv run pytest
```

## Testing and Benchmarks

Core tests:

```bash
uv run pytest
```

Quick model smoke checks:

```bash
uv run python benchmarks/test_model.py mlx-community/Qwen3-Coder-Next-4bit
uv run python benchmarks/test_model.py mlx-community/Qwen3-Coder-Next-4bit --capacity 208 --tokens 50
```

Validation suite (`validate_quality.py` uses positional experiment name):

```bash
uv run python benchmarks/validate_quality.py quality
uv run python benchmarks/validate_quality.py warmup
uv run python benchmarks/validate_quality.py memory
uv run python benchmarks/validate_quality.py memory-predictive
uv run python benchmarks/validate_quality.py delta-warmup
uv run python benchmarks/validate_quality.py adaptive
uv run python benchmarks/validate_quality.py expert-size
```

Profile generation:

```bash
uv run python benchmarks/profile_experts.py --model mlx-community/Qwen3-Coder-Next-4bit --prompts mixed --coding-weight 70 --toolchat-weight 30
```

mlx-moe-only server benchmark with live output:

```bash
uv run python benchmarks/benchmark_mlx_server.py \
  --model mlx-community/Qwen3-Coder-Next-4bit \
  --profile profiles/qwen3-coder-next-4bit.json \
  --capacity 208 --pin-top-k 32 --tools-mode field
```

This writes timestamped artifacts under:

```text
logs/model/<model_slug>/<profile_slug>/<datetime>/
  benchmark.json
  benchmark.md
  server.log
```

Pinning sweep (long-running):

```bash
uv run python benchmarks/sweep_profile_pinning.py
```

Cross-backend comparison (`benchmark_backends.py`) is optional and requires working llama.cpp.

## Architecture

### Startup Pipeline

`_startup()` in `mlx_moe/lazy_experts/generate.py` performs:

1. Load model lazy (`mlx_lm.load(..., lazy=True)`), detect MoE structure.
2. Auto-select capacity if omitted (`select_capacity` against Metal recommended working set).
3. Replace SwitchLinear modules with predictive-capable lazy modules.
4. Materialize non-expert weights (`mx.eval(model.parameters())`).
5. Build shard maps + `SafetensorsMap`.
6. Startup path (in order):
   - Prepacked weights (`*.weights.safetensors`) if present.
   - Cache-state upgrade (`*.json`) if present.
   - Else one of:
     - `warmup=full`: LCP warmup + predictive upgrade
     - profile-based upgrade (`--profile`)
     - router-only discovery + upgrade
7. Optional hybrid refinement (`warmup=hybrid`, only on fresh startup path).
8. Save cache state / prepacked snapshot if newly built.
9. Enable skip-fallback mode.
10. Apply `mx.set_wired_limit(...)` after expert loading completes.

### Module Replacement Chain

`QuantizedSwitchLinear` -> `LazyQuantizedSwitchLinear` -> `CachedQuantizedSwitchLinear` -> `PredictiveCachedSwitchLinear`

Final dispatch is zero-eval via pre-stacked tensors and lookup remap (`gather_qmm` path).

### Dynamic Cache Updates

`dynamic_cache_update()` runs between tokens and is invoked in server `_stream()` with adaptive interval/budget policy. Telemetry includes:

- `prefill`
- `ttft`
- `decode tok/s`
- `dcu_calls`
- `swaps`
- `fallback_rate`

### Wired Memory

`mx.set_wired_limit()` is applied after startup to pin the working set in a Metal residency set. Keep this ordering; moving/removing it regresses throughput.

### Key Constraints

- Capacity too low on Qwen3-Coder degrades quality sharply.
- Capacity too high on 32 GB machines hits Metal pressure cliffs.
- Do not call `mx.eval()` in predictive forward paths.
- MLX GPU eval is not thread-safe; server serialization is intentional.

## API Server (`mlx-moe serve`)

```bash
mlx-moe serve MODEL \
  [--host 127.0.0.1] [--port 8080] \
  [--capacity N] [--profile PATH] [--pin-top-k N] \
  [--max-tokens N] [--max-input-tokens N] \
  [--kv-bits N] [--kv-cache-slots N] \
  [--warmup hybrid|full|none] \
  [--shutdown-timeout N]
```

Endpoints:

- `POST /v1/chat/completions`
- `POST /v1/messages`
- `GET /v1/models`

Sampling:

- Request fields `temperature`, `top_p`, `top_k` are passed through.
- Default for Qwen3-Coder family is `temp=0.2`, `top_p=0.95`, `top_k=40`.

Profile resolution:

- If `--profile` is omitted, server auto-detects from `profiles/`:
  - `<model-slug>-toolchat.json`
  - `<model-slug>.json`

KV cache behavior:

- Keyed LRU cache (`--kv-cache-slots`, default 1).
- Reuse is based on longest common token prefix for each cache key.
- Cache is invalidated before generation and restored only on successful completion.

Hybrid startup refinement:

- With `--warmup hybrid`, server load runs two short coding prompts and dynamic updates before `Ready.`.

Limits:

- `--max-input-tokens` rejects oversized requests.
- `--max-tokens` caps request output.
- Anthropic non-streaming responses are additionally capped to 512 output tokens.

## Code Style

- Comments explain why, not what.
- No defensive checks for impossible states.
- No silent fallbacks that hide failures.
- Remove dead code paths cleanly; no compatibility shims for deleted behavior.

### Python Style Guidelines

**Imports** (ordered by PEP 8 with custom groupings):
1. Standard library (`asyncio`, `json`, `pathlib`, `re`, `time`, `uuid`)
2. Third-party packages (`mlx`, `mlx.nn`, `mlx_lm`, `numpy`, `pytest`, `starlette`, `uvicorn`)
3. Local imports (`from mlx_moe.lazy_experts import ...`)

Use absolute imports for the package (`from mlx_moe.lazy_experts.modules import ...`), not relative (`from .modules import ...`).

**Formatting**:
- Maximum line length: 100 characters
- Use 4 spaces for indentation (no tabs)
- Use hanging indents for long function signatures
- Put imports in a single block (no multiple `import` statements scattered)

**Types**:
- Use Python 3.12+ type hints: `def foo(x: int) -> str:`
- Use `X | None` instead of `Optional[X]`
- Use `dict[str, int]` instead of `Dict[str, int]`
- For internal/private functions, type hints are optional but encouraged
- Use `Path` from `pathlib` for file paths

**Naming Conventions**:
- `snake_case` for functions, methods, and variables
- `SCREAMING_SNAKE_CASE` for constants
- `PascalCase` for classes
- Leading underscore (`_func`) for private functions
- Double leading underscore (`__method`) for name mangling (use sparingly)
- Suffix `_cb` for callback functions
- Suffix `_map` for dict-based mappings
- Prefix `num_` or `n_` for counts

**Error Handling**:
- Raise specific exceptions with clear messages
- Do not catch bare `Exception` unless re-raising or logging
- Use `assert` for internal invariants, not for runtime validation
- Fail fast with descriptive errors, not silent fallbacks

**MLX-Specific**:
- Never call `mx.eval()` in the forward pass of predictive modules (critical for performance)
- Use `mx.stop_gradient()` where needed to prevent unwanted gradient flow
- Prefer in-place operations when possible to reduce memory allocation
- Remember MLX GPU eval is not thread-safe; serialization in server is intentional

## Testing

**Run all tests**:
```bash
uv run pytest
```

**Run a single test file**:
```bash
uv run pytest tests/test_unit_core.py
```

**Run a single test class**:
```bash
uv run pytest tests/test_unit_core.py::TestExpertCache
```

**Run a single test function**:
```bash
uv run pytest tests/test_unit_core.py::TestExpertCache::test_put_and_lookup -v
```

**Run tests matching a pattern**:
```bash
uv run pytest -k "test_lcp"
```

**Run with output capture disabled** (see print statements):
```bash
uv run pytest -s
```

**Run integration tests only**:
```bash
uv run pytest tests/test_integration.py
```

Note: Some tests in `test_unit_core.py` use synthetic mocks and don't require model files. Integration tests may require a real model to be available.
