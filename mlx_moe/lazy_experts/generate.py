# Copyright © 2023-2025 Apple Inc.

import os
import time
from pathlib import Path

import mlx.core as mx

from .loading import (
    _find_switch_mlp,
    _detect_num_experts,
    select_capacity,
    _with_cache_limit_zero,
    _build_shard_map,
    SafetensorsMap,
)
from .core import enable_lazy_experts
from .warmup import fast_delta_warmup
from .discovery import router_only_discovery
from .persistence import (
    save_cache_state,
    load_cache_state,
    save_prepacked_weights,
    load_prepacked_weights,
    upgrade_from_saved_state,
    load_universal_profile,
    upgrade_from_profile,
)


_WARMUP_CACHE = 256 * 1024 * 1024


def _fix_group_size(model):
    """Fix group_size mismatch by inferring from weight/scales shapes.

    Some models (e.g., Qwen3-Coder-Next-MLX-8bit) have incorrect group_size
    in config.json. The actual group_size must be computed from tensor shapes.
    """
    for layer in model.layers:
        switch, _ = _find_switch_mlp(layer)
        if switch is None:
            continue
        for name in ("gate_proj", "up_proj", "down_proj"):
            proj = getattr(switch, name, None)
            if proj is None or not hasattr(proj, "weight") or not hasattr(proj, "scales"):
                continue
            # Infer actual group_size from shapes
            # weight shape: (num_experts, output_dim, input_dim)
            # scales shape: (num_experts, output_dim, num_groups)
            # group_size = input_dim / num_groups
            actual = proj.weight.shape[-1] // proj.scales.shape[-1]
            if actual != proj.group_size:
                proj.group_size = actual


def _startup(
    model_name,
    prompt,
    cache_dir=None,
    profile_path=None,
    prepacked=True,
    capacity=None,
    warmup="hybrid",
    pin_top_k: int | None = None,
):
    """Shared startup for generate and stream_generate.

    Args:
        warmup: "hybrid" (default) — profile/discovery + short real generation to
            refine expert cache. "full" — Phase 2 LCP warmup with actual expert
            loading (~90s, 0% fallback). "none" — profile/discovery only, no
            refinement.

    Returns (model, tokenizer, model_path) with all warmup/upgrade/wiring done.
    """
    from .core import upgrade_to_predictive
    import mlx_lm as _mlx_lm
    from mlx_lm.utils import hf_repo_to_path
    from huggingface_hub.errors import HFValidationError

    t_total_start = time.perf_counter()

    # First check if the path exists at all
    if os.path.exists(model_name):
        if not os.path.isdir(model_name):
            raise FileNotFoundError(
                f"Model path exists but is not a directory: {model_name}"
            )
        model_path = Path(model_name)
    elif "/" not in model_name:
        # Local path without slash - try as relative path
        raise FileNotFoundError(
            f"Model path does not exist: {model_name}\n"
            f"Use a valid local path or HuggingFace model ID (e.g., mlx-community/Qwen3-Coder-Next-4bit)"
        )
    else:
        # HuggingFace model ID - verify it looks like one (has namespace/repo format)
        # and doesn't look like a file path
        if model_name.startswith("/") or model_name.startswith("."):
            raise FileNotFoundError(
                f"Model path does not exist: {model_name}\n"
                f"Use a valid local path or HuggingFace model ID (e.g., mlx-community/Qwen3-Coder-Next-4bit)"
            )
        try:
            model_path = hf_repo_to_path(model_name)
        except HFValidationError:
            raise FileNotFoundError(
                f"Invalid model ID or model not found: {model_name}\n"
                f"HuggingFace model IDs should be in the format 'namespace/repo_name'"
            )
        except Exception as e:
            raise FileNotFoundError(
                f"Failed to download model: {model_name}\n{str(e)}"
            )

    if not model_path.is_dir():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")

    config_path = model_path / "config.json"
    if not config_path.is_file():
        safetensors_files = list(model_path.glob("*.safetensors"))
        if not safetensors_files:
            raise FileNotFoundError(
                f"Model directory is empty or missing model files: {model_path}\n"
                f"Please download the model first."
            )
        else:
            raise FileNotFoundError(
                f"config.json not found in: {model_path}\n"
                f"This may not be a valid HuggingFace-compatible model directory."
            )

    t0 = time.perf_counter()
    model, tokenizer = _mlx_lm.load(str(model_path), lazy=True)

    # Fix group_size mismatch: config.json may declare wrong group_size
    # Actual group_size must be inferred from weight/scales shapes
    # _fix_group_size(model)

    # Detect MoE architecture
    num_moe_layers = 0
    num_experts = 0
    expert_slot_mb = 0.0
    for layer in model.layers:
        switch, _ = _find_switch_mlp(layer)
        if switch is not None:
            num_moe_layers += 1
            if num_experts == 0:
                num_experts = _detect_num_experts(switch)
            if expert_slot_mb == 0.0:
                proj = getattr(switch, "gate_proj", None) or getattr(switch, "up_proj", None)
                if proj is not None and hasattr(proj, "weight"):
                    per_expert = proj.weight.nbytes // num_experts
                    for attr in ("scales", "biases"):
                        t = getattr(proj, attr, None)
                        if t is not None:
                            per_expert += t.nbytes // num_experts
                    expert_slot_mb = per_expert * 3 / 1e6

    recommended_gb = mx.device_info()["max_recommended_working_set_size"] / 1e9
    gc_limit_gb = recommended_gb * 0.95

    if capacity is None:
        enable_lazy_experts(model, model_path,
                            cache_capacity_per_layer=0,
                            predictive=True)
        mx.eval(model.parameters())
        base_gb = mx.get_active_memory() / 1e9

        capacity = select_capacity(base_gb, recommended_gb,
                                   num_moe_layers=num_moe_layers,
                                   expert_slot_mb=expert_slot_mb)

    if num_experts > 0:
        capacity = min(capacity, num_experts)

    # Always use predictive mode (zero-eval forward pass). Even at cap < num_experts,
    # predictive mode avoids the mx.eval() calls inside the forward pass that break
    # async_eval pipelining and cause OOM with large experts (~170 MB for Mixtral 22B).
    # Fallbacks map to slot 0; between-token swapping corrects misses progressively.
    enable_lazy_experts(model, model_path,
                        cache_capacity_per_layer=capacity,
                        predictive=True)
    mx.eval(model.parameters())

    shard_map = _build_shard_map(model_path)
    all_shards = sorted(set(shard_map.values()))
    model._st_map = SafetensorsMap(all_shards)

    t_load = time.perf_counter() - t0
    print(f"  Model load: {t_load:.1f}s ({mx.get_active_memory() / 1e9:.1f} GB) "
          f"[{num_experts} experts, cap={capacity}]")

    cache_path = None
    prepacked_path = None
    if cache_dir is not None:
        cache_dir = os.path.expanduser(cache_dir)
        os.makedirs(cache_dir, exist_ok=True)
        model_path_str = str(model_path).rstrip("/")
        safe_name_base = model_path_str.replace("/", "--").replace("\\", "--")

        cache_path = os.path.join(cache_dir, f"{safe_name_base}.json")
        prepacked_path = cache_path.replace(".json", ".weights.safetensors")
        meta_path = prepacked_path + ".meta.json"

        if not os.path.exists(meta_path):
            safe_name_alt = safe_name_base + "--"
            cache_path = os.path.join(cache_dir, f"{safe_name_alt}.json")
            prepacked_path = cache_path.replace(".json", ".weights.safetensors")

    used_saved_state = False
    mx.reset_peak_memory()

    if prepacked and prepacked_path and os.path.exists(prepacked_path):
        t0 = time.perf_counter()
        with _with_cache_limit_zero(_WARMUP_CACHE):
            load_prepacked_weights(model, prepacked_path, model_path=model_path)
        print(f"  Prepacked load: {time.perf_counter() - t0:.1f}s")

        if cache_path and os.path.exists(cache_path):
            cache_state = load_cache_state(cache_path)
            saved_prompt = cache_state.get("metadata", {}).get("prompt")
            if saved_prompt and saved_prompt != prompt:
                t0 = time.perf_counter()
                with _with_cache_limit_zero(_WARMUP_CACHE):
                    fast_delta_warmup(model, tokenizer, model_path, prompt,
                                      discovery_tokens=10)
                print(f"  Delta warmup: {time.perf_counter() - t0:.1f}s")
        used_saved_state = True

    elif cache_path and os.path.exists(cache_path):
        t0 = time.perf_counter()
        cache_state = load_cache_state(cache_path)
        with _with_cache_limit_zero(_WARMUP_CACHE):
            upgrade_from_saved_state(model, model_path, cache_state, capacity)
        print(f"  Cache state upgrade: {time.perf_counter() - t0:.1f}s")

        saved_prompt = cache_state.get("metadata", {}).get("prompt")
        if saved_prompt and saved_prompt != prompt:
            t0 = time.perf_counter()
            with _with_cache_limit_zero(_WARMUP_CACHE):
                fast_delta_warmup(model, tokenizer, model_path, prompt,
                                  discovery_tokens=10)
            print(f"  Delta warmup: {time.perf_counter() - t0:.1f}s")
        used_saved_state = True

    else:
        if warmup == "full":
            t0 = time.perf_counter()
            _mlx_lm.generate(model, tokenizer, prompt=prompt,
                             max_tokens=50, verbose=False)
            with _with_cache_limit_zero(_WARMUP_CACHE):
                upgrade_to_predictive(model, model_path, capacity,
                                      st_map=model._st_map)
            print(f"  Full LCP warmup + upgrade: {time.perf_counter() - t0:.1f}s")
        elif profile_path is not None:
            t0 = time.perf_counter()
            profile = load_universal_profile(profile_path)
            with _with_cache_limit_zero(_WARMUP_CACHE):
                upgrade_from_profile(
                    model, model_path, capacity, profile, pin_top_k=pin_top_k
                )
            print(f"  Profile-based upgrade: {time.perf_counter() - t0:.1f}s")
        else:
            t0 = time.perf_counter()
            with _with_cache_limit_zero(_WARMUP_CACHE):
                router_only_discovery(model, tokenizer, prompt, max_tokens=10)
                upgrade_to_predictive(model, model_path, capacity)
            print(f"  Router-only discovery + upgrade: {time.perf_counter() - t0:.1f}s")

        if warmup == "hybrid" and not used_saved_state:
            from .core import dynamic_cache_update, enable_skip_fallback
            enable_skip_fallback(model)
            t0 = time.perf_counter()
            for resp in _mlx_lm.stream_generate(model, tokenizer, prompt=prompt,
                                                 max_tokens=10):
                dynamic_cache_update(model, max_layer_updates=48)
            swaps = sum(
                s.get("swaps", 0)
                for s in dynamic_cache_update(model, max_layer_updates=48)
            )
            print(f"  Hybrid warmup: {time.perf_counter() - t0:.1f}s ({swaps} expert swaps)")

    peak_gb = mx.get_peak_memory() / 1e9
    if peak_gb > gc_limit_gb:
        print(f"  WARNING: upgrade peak {peak_gb:.1f} GB exceeded gc_limit {gc_limit_gb:.1f} GB")
        print(f"  Generation may be slow. Try --capacity {max(8, capacity - 8)}")

    if cache_path and not used_saved_state:
        save_cache_state(model, cache_path,
                         metadata={"prompt": prompt, "capacity": capacity})

    if prepacked and prepacked_path and not os.path.exists(prepacked_path):
        t0 = time.perf_counter()
        save_prepacked_weights(model, prepacked_path)
        print(f"  Save prepacked: {time.perf_counter() - t0:.1f}s")

    from .core import enable_skip_fallback
    if warmup != "hybrid" or used_saved_state:
        enable_skip_fallback(model)

    t_total = time.perf_counter() - t_total_start
    print(f"  Total startup: {t_total:.1f}s")

    if hasattr(mx, "set_wired_limit"):
        active = mx.get_active_memory()
        limit = int(mx.device_info()["memory_size"] * 0.75)
        wired = min(active, limit)
        mx.set_wired_limit(wired)
        print(f"  Wired {wired / 1e9:.1f} GB in residency set")

    return model, tokenizer, model_path


def generate(model_name: str, prompt: str, max_tokens: int = 200,
                   cache_dir: str | None = None,
                   profile_path: str | None = None,
                   prepacked: bool = True,
                   capacity: int | None = None,
                   kv_bits: int | None = None,
                   warmup: str = "hybrid",
                   pin_top_k: int | None = None) -> str:
    """One-call generation with all optimizations.

    Args:
        model_name: HuggingFace model name (e.g. "mlx-community/Qwen3-Coder-Next-4bit").
        prompt: Text prompt for generation.
        max_tokens: Maximum tokens to generate.
        cache_dir: Directory for cache state persistence. None disables caching.
        profile_path: Path to universal expert profile JSON for pinning.
        prepacked: Save/load prepacked weight files for fastest warm start.
        capacity: Experts per layer. None = auto-select based on available RAM.
        kv_bits: Quantize KV cache to this many bits (8 recommended). Saves ~45%
            KV memory at 8-bit. None = fp16 (default).
        warmup: "hybrid" (default), "full", or "none".
        pin_top_k: If set, pin top-K profile experts per layer. Use 0 for no pinning.

    Returns:
        Generated text string.
    """
    import mlx_lm as _mlx_lm

    model, tokenizer, _ = _startup(
        model_name, prompt, cache_dir=cache_dir,
        profile_path=profile_path, prepacked=prepacked, capacity=capacity,
        warmup=warmup, pin_top_k=pin_top_k)

    kv_kwargs = {}
    if kv_bits is not None:
        kv_kwargs = dict(kv_bits=kv_bits, kv_group_size=64, quantized_kv_start=0)

    return _mlx_lm.generate(model, tokenizer, prompt=prompt,
                             max_tokens=max_tokens, verbose=False, **kv_kwargs)


def stream_generate(model_name: str, prompt: str, max_tokens: int = 200,
                          cache_dir: str | None = None,
                          profile_path: str | None = None,
                          prepacked: bool = True,
                          capacity: int | None = None,
                          kv_bits: int | None = None,
                          warmup: str = "hybrid",
                          pin_top_k: int | None = None):
    """Streaming variant of generate.

    Startup is blocking. After startup, yields GenerationResponse objects
    from mlx_lm.stream_generate.

    Yields:
        mlx_lm.generate.GenerationResponse with token-by-token output.
    """
    import mlx_lm as _mlx_lm

    model, tokenizer, _ = _startup(
        model_name, prompt, cache_dir=cache_dir,
        profile_path=profile_path, prepacked=prepacked, capacity=capacity,
        warmup=warmup, pin_top_k=pin_top_k)

    kv_kwargs = {}
    if kv_bits is not None:
        kv_kwargs = dict(kv_bits=kv_bits, kv_group_size=64, quantized_kv_start=0)

    yield from _mlx_lm.stream_generate(model, tokenizer, prompt=prompt,
                                        max_tokens=max_tokens, **kv_kwargs)


class Session:
    """Reusable session for multi-turn generation.

    Loads the model once on first use. Subsequent calls reuse the loaded model
    and run delta warmup if the prompt domain changed.

    Usage:
        session = Session("mlx-community/Qwen3-Coder-Next-4bit",
                               cache_dir="~/.cache/mlx-moe")
        for resp in session.stream("Write a Flask server"):
            print(resp.text, end="")
        text = session.generate("Now add tests")
        session.close()
    """

    def __init__(self, model_name: str, cache_dir: str | None = None,
                 profile_path: str | None = None, prepacked: bool = True,
                 capacity: int | None = None, kv_bits: int | None = None,
                 pin_top_k: int | None = None):
        self._model_name = model_name
        self._cache_dir = cache_dir
        self._profile_path = profile_path
        self._prepacked = prepacked
        self._capacity = capacity
        self._kv_bits = kv_bits
        self._pin_top_k = pin_top_k
        self._model = None
        self._tokenizer = None
        self._model_path = None
        self._last_prompt = None

    def _ensure_loaded(self, prompt: str):
        if self._model is None:
            self._model, self._tokenizer, self._model_path = _startup(
                self._model_name, prompt, cache_dir=self._cache_dir,
                profile_path=self._profile_path, prepacked=self._prepacked,
                capacity=self._capacity, pin_top_k=self._pin_top_k)
            self._last_prompt = prompt
        elif self._last_prompt != prompt:
            t0 = time.perf_counter()
            with _with_cache_limit_zero(_WARMUP_CACHE):
                fast_delta_warmup(self._model, self._tokenizer,
                                  self._model_path, prompt, discovery_tokens=10)
            print(f"  Delta warmup: {time.perf_counter() - t0:.1f}s")
            self._last_prompt = prompt

    def _kv_kwargs(self):
        if self._kv_bits is None:
            return {}
        return dict(kv_bits=self._kv_bits, kv_group_size=64, quantized_kv_start=0)

    def stream(self, prompt: str, max_tokens: int = 200):
        """Stream tokens for a single turn."""
        import mlx_lm as _mlx_lm

        self._ensure_loaded(prompt)
        yield from _mlx_lm.stream_generate(self._model, self._tokenizer,
                                            prompt=prompt, max_tokens=max_tokens,
                                            **self._kv_kwargs())

    def generate(self, prompt: str, max_tokens: int = 200) -> str:
        """Non-streaming generation for a single turn."""
        import mlx_lm as _mlx_lm

        self._ensure_loaded(prompt)
        return _mlx_lm.generate(self._model, self._tokenizer,
                                 prompt=prompt, max_tokens=max_tokens, verbose=False,
                                 **self._kv_kwargs())

    @property
    def memory_gb(self) -> float:
        return mx.get_active_memory() / 1e9

    def close(self):
        """Release model and clear GPU memory."""
        if self._model is not None:
            st_map = getattr(self._model, "_st_map", None)
            if st_map is not None:
                st_map.close()
        self._model = None
        self._tokenizer = None
        self._model_path = None
        self._last_prompt = None
        mx.clear_cache()
