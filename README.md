# MLX-MoE

Run large Mixture-of-Experts models on memory-constrained Macs by loading only router-selected experts on demand from SSD. A 46 GB Qwen3-Coder-Next model runs on a 32 GB Mac at 6-23 tok/s using 19 GB, with coherent output through 1000+ tokens.

## Quick Start

```bash
git clone https://github.com/mu-hashmi/mlx-moe.git
cd mlx-moe
pip install .
```

For development:

```bash
uv sync
```

### Server

Start an OpenAI- and Anthropic-compatible API server:

```bash
mlx-moe serve mlx-community/Qwen3-Coder-Next-4bit
```

Point any compatible client at `http://127.0.0.1:8080`:

```bash
# OpenAI-compatible clients
OPENAI_BASE_URL=http://localhost:8080/v1 OPENAI_API_KEY=mlx-moe <your-client>

# Anthropic-compatible clients
ANTHROPIC_BASE_URL=http://localhost:8080 ANTHROPIC_API_KEY=mlx-moe <your-client>
```

Endpoints:

- `POST /v1/chat/completions` — OpenAI chat completions (streaming and non-streaming)
- `POST /v1/messages` — Anthropic Messages API (SSE streaming)
- `GET /v1/models` — model discovery

Options: `--port`, `--host`, `--capacity`, `--profile`, `--kv-bits`.

Sampling parameters (`temperature`, `top_p`, `top_k`) are passed through from the request body. When not specified, the server applies model-recommended defaults (e.g. `temp=1.0, top_p=0.95, top_k=40` for Qwen3-Coder). Client values always override.

### Python API

```python
from mlx_moe import generate

print(generate("mlx-community/Qwen3-Coder-Next-4bit",
               "Write a Python hello world program",
               max_tokens=200))
```

Streaming:

```python
from mlx_moe import stream_generate

for response in stream_generate("mlx-community/Qwen3-Coder-Next-4bit",
                                "Write a Flask server", max_tokens=200):
    print(response.text, end="", flush=True)
```

Multi-turn sessions:

```python
from mlx_moe import Session

session = Session("mlx-community/Qwen3-Coder-Next-4bit",
                  cache_dir="~/.cache/mlx-moe")
for response in session.stream("Write a linked list in Python"):
    print(response.text, end="", flush=True)
print()
print(session.generate("Now add type hints"))
```

First launch downloads the model (~24 GB) and warms up experts (~13s). Subsequent launches: ~6s to first token.

## Supported Models

| Model                                                                               | Experts | Top-K | MoE Layers |   Full Size | mlx-moe Memory | tok/s |
| ----------------------------------------------------------------------------------- | ------: | ----: | ---------: | ----------: | -------------: | ----: |
| [Qwen3-Coder-Next-4bit](https://huggingface.co/mlx-community/Qwen3-Coder-Next-4bit) |     512 |    10 |         48 | 46 GB (OOM) |    **19.1 GB** |    23 |

Any MLX model using `SwitchGLU` is supported — covers Qwen, Mixtral, GLM, DeepSeek, Hunyuan, PhiMoE, Jamba, OLMoE, MiniMax, and GraniteMoE families. Both stacked and per-expert safetensors formats are handled automatically. Models with complex gates (e.g. GLM's `MoEGate`) are also supported.

### Which Models Benefit?

mlx-moe is valuable when a model **exceeds your Mac's RAM** and has favorable MoE architecture. Three factors predict whether output quality holds up at reduced expert coverage:

1. **High expert ratio** (expert_count / top_k ≥ 10) — more experts means more can stay on SSD
2. **Shared expert** — provides a quality floor when routed experts miss
3. **Concentrated routing** — a small set of "universal" experts handles most tokens; the long tail stays on SSD

| Model                      | Ratio | Shared Expert |    Routing     | Verdict                                 |
| -------------------------- | ----: | :-----------: | :------------: | :-------------------------------------- |
| Qwen3-Coder-Next (512 exp) |   51x |      Yes      | 15% universal  | **Works at 40% coverage, 0% fallback**  |
| Qwen3-Coder-480B (512 exp) |   51x |      Yes      |   Same arch    | **Primary target** — same arch, ~200 GB |
| Qwen3-235B (128 exp)       |   16x |      Yes      |   Same arch    | Good target — ~130 GB                   |
| GLM-4.7-Flash (64 exp)     |   16x |      Yes      | 44% universal  | Works, but fits on 32 GB natively       |
| Qwen3-30B-A3B (128 exp)    |   16x |      No       | 14% universal  | Needs ~75% coverage — fits natively     |
| Qwen2-MoE-57B (64 exp)     |    8x |      Yes      | 95% universal  | All experts needed — quality degrades   |
| Mixtral-8x7B (8 exp)       |    4x |      No       | 100% universal | Ratio too low                           |

Models without a shared expert collapse to garbage below ~75% expert coverage. Models with uniform routing can't benefit from selective caching. The sweet spot is high ratio + shared expert + concentrated routing.

## Hardware Requirements

| System RAM | Capacity | Memory | tok/s | Notes                |
| ---------: | -------: | -----: | ----: | -------------------- |
|      32 GB |      208 |  19 GB |  8-23 | Recommended          |
|      48 GB |      320 | ~28 GB | 15-25 | Estimated            |
|      64 GB |      432 | ~37 GB | 20-30 | Estimated            |
|     128 GB |      512 | ~44 GB | 30-40 | Full expert coverage |

"Capacity" = experts cached per MoE layer. Auto-selected based on available RAM.

## How It Works

Each MoE layer has hundreds of experts but only routes to a few per token (e.g. 10 out of 512 for Qwen). mlx-moe replaces the expert weight modules with lazy-loading versions that:

1. **Load the model without expert weights** (~1.4 GB instead of 46 GB)
2. **Discover which experts matter** via router-only forward pass (~1s)
3. **Load only those experts** into GPU-resident stacked tensors (~15s)
4. **Generate with zero-eval dispatch** — no `mx.eval()` in the forward pass, preserving MLX's async pipeline

### Expert Profiles

Pre-computed profiles in `profiles/` identify universally-activated experts per model. These are determined by the model's router weights (same for everyone) and serve two purposes:

- **Pinning** — universal experts stay in cache permanently, preventing quality degradation at 300+ tokens
- **Fast cold start** — skip the discovery step entirely (13s instead of 17s)

A profile is shipped for Qwen3-Coder-Next. Generate profiles for other models with:

```bash
uv run python benchmarks/profile_experts.py --model mlx-community/Some-MoE-Model-4bit
```

## Performance

### Qwen3-Coder-Next-4bit on 32 GB Mac

| Config                      | tok/s | Repetition @ 1000 tokens |
| --------------------------- | ----: | -----------------------: |
| Baseline                    |   8.8 |  0.20 (degrades at ~250) |
| + Pinning                   |   8.7 |                     0.03 |
| + Wired limit               |  10.6 |                     0.03 |
| + All optimizations (burst) |   ~23 |                     0.03 |

### Startup Times

| Scenario                                  |  Time |
| ----------------------------------------- | ----: |
| Warm start (prepacked weights exist)      |   ~6s |
| Cold start (with profile)                 |  ~13s |
| Cold start (no profile, router discovery) |  ~17s |
| Domain switch (delta warmup)              | ~2-3s |

### KV Cache Quantization

Quantize the KV cache to 8-bit to reduce memory for longer contexts. Saves ~45% KV memory with no quality degradation.

```python
text = generate("mlx-community/Qwen3-Coder-Next-4bit",
                       "Your prompt", kv_bits=8)

# Or via the server
# mlx-moe serve mlx-community/Qwen3-Coder-Next-4bit --kv-bits 8
```

| Context Length | fp16 KV | 8-bit KV |
| -------------- | ------: | -------: |
| 4K tokens      |  384 MB |  ~210 MB |
| 8K tokens      |  768 MB |  ~420 MB |
| 16K tokens     |  1.5 GB |  ~820 MB |
| 32K tokens     |  3.0 GB |  ~1.6 GB |

Uses mlx-lm's `QuantizedKVCache` with `quantized_kv_start=0` (quantize from the first token). Available on all APIs: `generate`, `stream_generate`, `Session`, and `mlx-moe serve`.

### Multi-Turn Stability

Tested over 20 turns with varied prompts:

- Memory growth: **+0.00 GB** (19.04 GB flat)
- Speed: 5-20 tok/s (improves as caches warm)
- No KV cache leak, no degradation wall

## API

### `generate(model_name, prompt, **kwargs) -> str`

One-call generation. Auto-selects capacity, loads cached state, applies pinning.

```python
text = generate(
    "mlx-community/Qwen3-Coder-Next-4bit",
    "Your prompt",
    max_tokens=200,
    cache_dir="~/.cache/mlx-moe",                  # persist state between runs
    profile_path="profiles/qwen3-coder-next-4bit.json", # enable pinning
    kv_bits=8,                                        # quantize KV cache (optional)
)
```

### `stream_generate(model_name, prompt, **kwargs) -> Generator`

Same as `generate` but yields `GenerationResponse` objects token-by-token.

### `Session(model_name, **kwargs)`

Reusable session for multi-turn generation. Loads the model once, runs delta warmup automatically when prompts change domain.

```python
session = Session("mlx-community/Qwen3-Coder-Next-4bit",
                       cache_dir="~/.cache/mlx-moe")
session.stream(prompt, max_tokens=200)   # yields GenerationResponse
session.generate(prompt, max_tokens=200) # returns str
session.memory_gb                        # current GPU memory
session.close()                          # release model
```

### Step-by-Step API

For full control over the loading pipeline:

```bash
uv run python -c "
import mlx.core as mx
import mlx_lm
from mlx_moe.lazy_experts import enable_lazy_experts, upgrade_to_predictive, get_fallback_stats
from mlx_moe.lazy_experts.loading import _find_switch_mlp
from mlx_moe.lazy_experts.discovery import router_only_discovery
from mlx_moe.lazy_experts.core import dynamic_cache_update, enable_skip_fallback
model_path = '/Users/steven/.lmstudio/models/lmstudio-community/Qwen3-Coder-Next-MLX-4bit'
print('Loading model...')
model, tokenizer = mlx_lm.load(model_path, lazy=True)
print('Model loaded')
print('Enable cached capacity=208...')
enable_lazy_experts(model, model_path, cache_capacity_per_layer=208, predictive=True)
mx.eval(model.parameters())
switch, _ = _find_switch_mlp(model.layers[0], 0)
print('Module type:', type(switch.gate_proj).__name__)
print('=== Router-only discovery (does NOT load experts) ===')
router_only_discovery(model, tokenizer, 'git pull origin main', max_tokens=10)
print('Discovery done')
print('Upgrade to predictive...')
upgrade_to_predictive(model, model_path, 208)
switch, _ = _find_switch_mlp(model.layers[0], 0)
print('After upgrade:', type(switch.gate_proj).__name__)
print('Enable skip fallback...')
enable_skip_fallback(model)
print('=== Hybrid warmup (10 tokens) ===')
for resp in mlx_lm.stream_generate(model, tokenizer, prompt='git pull origin main', max_tokens=10):
    pass
dynamic_cache_update(model, max_layer_updates=48)
print('Hybrid done')
print()
print('=== First generate ===')
output = mlx_lm.generate(model, tokenizer, prompt='git pull origin main', max_tokens=50, verbose=False)
print('Output:', repr(output[:150]))
stats = get_fallback_stats(model)
rate = stats['fallback_rate'] * 100
print('Fallback rate:', rate, '%')
"
```

## Testing

```bash
uv run pytest                # 100 tests, ~0.5s (no model download needed)
```

The test suite covers unit tests (ExpertCache, select_capacity, module detection, persistence roundtrips) and integration tests (synthetic 8-expert model through the full lazy → predictive → generate pipeline, skip-fallback with cache misses, golden reference comparison, server endpoints).

Smoke tests against real models are in `benchmarks/` and run manually:

```bash
uv run python benchmarks/test_model.py mlx-community/Qwen3-Coder-Next-4bit
uv run python benchmarks/test_model.py mlx-community/Qwen3-Coder-Next-4bit --capacity 208 --tokens 50
```

Quality and memory validation across diverse prompts:

```bash
uv run python benchmarks/validate_quality.py quality    # diverse prompt quality test
uv run python benchmarks/validate_quality.py memory     # memory growth over 500 tokens
```

These take minutes (model download + cold start + generation) and require a Mac with enough RAM for the model.

## Known Limitations

- **Quality cliff below capacity 192** — garbled output regardless of warmup (Qwen)
- **Metal pressure cliff above capacity 208** on 32 GB — 3x eval degradation
- **Mild repetition at 1000+ tokens** — model limitation at 40% expert coverage
- **Chat template required** for some models (GLM-4.7-Flash)
- **Not thread-safe** — MLX GPU eval is single-threaded; the server serializes requests
- **Semaphore leak warning on Ctrl+C** — cosmetic; the `os._exit(0)` handler bypasses Python's cleanup because MLX Metal ops block the GIL

## License

MIT
