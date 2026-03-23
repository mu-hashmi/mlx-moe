# Server Architecture

[Back to Architecture Overview](README.md)

This page covers `mlx_moe/server.py`: startup/refinement, request execution, KV reuse, and telemetry.

## Boot and Warmup

```mermaid
flowchart TD
    A["mlx-moe serve"] --> B["Server.load()"]
    B --> C["_startup(..., warmup=...)"]
    C --> D{"warmup == hybrid?"}
    D -- yes --> E["_run_startup_refinement()"]
    E --> E1["2 short coding prompts (max_tokens=24)"]
    E1 --> E2["adaptive dynamic_cache_update budgets (48 then 32)"]
    E2 --> F["Ready."]
    D -- no --> F
```

Hybrid refinement cost is paid before first client request.

## Request Path

```mermaid
flowchart TD
    A["HTTP request"] --> B{"endpoint"}
    B -- "/v1/chat/completions" --> C["OpenAI message normalization"]
    B -- "/v1/messages" --> D["Anthropic message normalization"]
    C --> E["tokenize chat template"]
    D --> E
    E --> F["input length check"]
    F --> G{"stream?"}
    G -- no --> H["acquire lock + _generate_sync(...)"]
    G -- yes --> I["acquire lock + streaming generator"]
    H --> J["response payload"]
    I --> J
```

## KV Cache Reuse

```mermaid
flowchart TD
    A["Request body"] --> B["derive cache_key (cache_key/session_id/metadata/user/default)"]
    B --> C{"_kv_cache has entry for key?"}
    C -- yes --> D["find longest common prefix"]
    D --> E["trim prompt_cache to prefix"]
    E --> F["prefill suffix only"]
    C -- no --> G["new prompt_cache"]
    G --> F
    F --> H["stream generation"]
    H --> I{"completed successfully?"}
    I -- yes --> J["store (prompt_tokens, prompt_cache) in keyed LRU"]
    I -- no --> K["do not restore cache entry"]
```

`_kv_cache` is an LRU `OrderedDict` bounded by `--kv-cache-slots`.

## Dynamic Update Policy During Streaming

```mermaid
flowchart LR
    A["fallback rate + swap streaks + generation tokens"] --> B["select update interval and budget"]
    B --> C["periodic dynamic_cache_update(...) calls"]
    C --> D["accumulate swaps/fallbacks/requests"]
    D --> E["emit telemetry"]
```

Telemetry fields:

- `prefill`
- `ttft`
- `decode tok/s`
- `dcu_calls`
- `swaps`
- `fallback_rate`
