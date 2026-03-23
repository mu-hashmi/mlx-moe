import asyncio
from collections import OrderedDict
import json
import re
import time
import uuid
from pathlib import Path

MAX_TOKENS_CAP = 4096
MAX_INPUT_TOKENS = 16384
DEFAULT_SHUTDOWN_TIMEOUT = 5

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route

import mlx.core as mx

import sys
if not sys.stdout.isatty():
    import functools
    print = functools.partial(print, flush=True)


def _find_profile(model_name: str, profiles_dir: Path | None = None) -> str | None:
    if profiles_dir is None:
        profiles_dir = Path(__file__).parent.parent / "profiles"
    if not profiles_dir.is_dir():
        return None
    model_slug = model_name.split("/")[-1].lower()
    candidate = profiles_dir / f"{model_slug}.json"
    if candidate.is_file():
        return str(candidate)
    return None


def _format_messages(messages: list, system: str | None = None) -> list:
    """Normalize Anthropic/OpenAI message formats to HF chat template format."""
    formatted = []
    if system:
        formatted.append({"role": "system", "content": system})
    for msg in messages:
        content = msg.get("content", "")
        role = msg["role"]
        if isinstance(content, list):
            parts = []
            tool_uses = []
            for block in content:
                if block.get("type") == "text":
                    parts.append(block["text"])
                elif block.get("type") == "tool_use":
                    tool_uses.append({
                        "id": block["id"],
                        "type": "function",
                        "function": {
                            "name": block["name"],
                            "arguments": block.get("input", {}),
                        },
                    })
                elif block.get("type") == "tool_result":
                    tool_content = block.get("content", "")
                    if isinstance(tool_content, list):
                        tool_content = "\n".join(
                            b["text"] for b in tool_content if b.get("type") == "text"
                        )
                    formatted.append({
                        "role": "tool",
                        "tool_call_id": block.get("tool_use_id", ""),
                        "content": tool_content,
                    })
                    continue
            if parts or tool_uses:
                entry = {"role": role, "content": "\n".join(parts) if parts else ""}
                if tool_uses:
                    entry["tool_calls"] = tool_uses
                formatted.append(entry)
        elif role == "tool":
            formatted.append(msg)
        else:
            entry = {"role": role, "content": content or ""}
            if "tool_calls" in msg:
                tool_calls = []
                for tc in msg["tool_calls"]:
                    tc_copy = dict(tc)
                    if "function" in tc_copy:
                        func = dict(tc_copy["function"])
                        if isinstance(func.get("arguments"), str):
                            func["arguments"] = json.loads(func["arguments"])
                        tc_copy["function"] = func
                    tool_calls.append(tc_copy)
                entry["tool_calls"] = tool_calls
            formatted.append(entry)
    return formatted


def _convert_tools_anthropic_to_openai(tools: list) -> list:
    """Convert Anthropic tool format to OpenAI format for chat templates."""
    return [{
        "type": "function",
        "function": {
            "name": t["name"],
            "description": t.get("description", ""),
            "parameters": t.get("input_schema", {}),
        },
    } for t in tools]


MODEL_SAMPLING_DEFAULTS = {
    "qwen3-coder": {"temp": 0.2, "top_p": 0.95, "top_k": 40},
}


def _sampling_defaults(model_name: str, model_path: Path | None = None) -> dict:
    if model_path is not None:
        config_path = model_path / "generation_config.json"
        if config_path.is_file():
            try:
                with open(config_path) as f:
                    config = json.load(f)
                defaults = {}
                if "temperature" in config:
                    defaults["temp"] = config["temperature"]
                if "top_p" in config:
                    defaults["top_p"] = config["top_p"]
                if "top_k" in config:
                    defaults["top_k"] = config["top_k"]
                if defaults:
                    return defaults
            except (json.JSONDecodeError, OSError):
                pass

    slug = model_name.split("/")[-1].lower()
    for prefix, defaults in MODEL_SAMPLING_DEFAULTS.items():
        if prefix in slug:
            return dict(defaults)
    return {}


def _sampling_kwargs(body: dict, model_name: str, model_path: Path | None = None) -> dict:
    defaults = _sampling_defaults(model_name, model_path)
    if "temperature" in body:
        defaults["temp"] = float(body["temperature"])
    if "top_p" in body:
        defaults["top_p"] = float(body["top_p"])
    if "top_k" in body:
        defaults["top_k"] = int(body["top_k"])
    return defaults


_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
_MAX_SUPPRESSED_THINK_TOKENS = 256


def _strip_thinking(text: str) -> str:
    return _THINK_RE.sub("", text)


def _filter_thinking_piece(
    text: str,
    in_thinking: bool,
    think_start: str,
    think_end: str,
) -> tuple[str, bool]:
    if not text:
        return text, in_thinking

    out = []
    i = 0
    n = len(text)
    while i < n:
        if not in_thinking:
            j = text.find(think_start, i)
            if j < 0:
                out.append(text[i:])
                break
            out.append(text[i:j])
            i = j + len(think_start)
            in_thinking = True
        else:
            j = text.find(think_end, i)
            if j < 0:
                i = n
                break
            i = j + len(think_end)
            in_thinking = False

    return "".join(out), in_thinking


def _make_msg_id():
    return f"msg_{uuid.uuid4().hex[:24]}"


def _make_chatcmpl_id():
    return f"chatcmpl-{uuid.uuid4().hex[:24]}"


class Server:
    def __init__(self, model_name: str, capacity: int | None = None,
                 profile_path: str | None = None,
                 pin_top_k: int | None = None,
                 max_tokens: int = MAX_TOKENS_CAP,
                 max_input_tokens: int = MAX_INPUT_TOKENS,
                 kv_bits: int | None = None,
                 warmup: str = "hybrid"):
        self._model_name = model_name
        self._capacity = capacity
        self._profile_path = profile_path or _find_profile(model_name)
        self._pin_top_k = pin_top_k
        self._model = None
        self._tokenizer = None
        self._model_path = None
        self._lock = asyncio.Lock()
        self._model_id = model_name.split("/")[-1]
        self._max_tokens = max_tokens
        self._max_input_tokens = max_input_tokens
        self._kv_bits = kv_bits
        self._warmup = warmup

    def load(self):
        from .lazy_experts.generate import _startup
        warmup_prompt = "Write a Python function that implements binary search on a sorted array."
        model, tokenizer, model_path = _startup(
            self._model_name, warmup_prompt,
            cache_dir=str(Path.home() / ".cache" / "mlx-moe"),
            profile_path=self._profile_path,
            capacity=self._capacity,
            warmup=self._warmup,
            pin_top_k=self._pin_top_k,
        )
        self._model = model
        self._tokenizer = tokenizer
        self._model_path = model_path
        self._memory_gb = mx.get_active_memory() / 1e9
        if self._warmup == "hybrid":
            self._run_startup_refinement()

    def _run_startup_refinement(self) -> None:
        import mlx_lm as _mlx_lm
        from .lazy_experts.core import dynamic_cache_update

        prompts = [
            "Write Python code that validates JSON input and returns structured errors.",
            "Write a Python function with unit tests for edge cases.",
        ]

        kv_kwargs = {}
        if self._kv_bits is not None:
            kv_kwargs = dict(kv_bits=self._kv_bits, kv_group_size=64, quantized_kv_start=0)

        t0 = time.perf_counter()
        calls = 0
        total_swaps = 0
        total_requests = 0
        total_fallbacks = 0

        def _accumulate(stats: list[dict]) -> tuple[int, int]:
            nonlocal calls, total_swaps, total_requests, total_fallbacks
            calls += 1
            swaps = sum(s.get("swaps", 0) for s in stats)
            requests = sum(s.get("requests", 0) for s in stats)
            fallbacks = sum(s.get("fallbacks", 0) for s in stats)
            total_swaps += swaps
            total_requests += requests
            total_fallbacks += fallbacks
            return swaps, requests

        round_rates = []
        for prompt_text in prompts:
            prompt = prompt_text
            if self._tokenizer.has_chat_template:
                prompt = self._tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt_text}],
                    add_generation_prompt=True, tokenize=False,
                )

            req_before = total_requests
            fb_before = total_fallbacks
            token_i = 0
            for _ in _mlx_lm.stream_generate(
                self._model, self._tokenizer, prompt=prompt, max_tokens=24, **kv_kwargs
            ):
                token_i += 1
                budget = 48 if token_i <= 12 else 32
                _accumulate(dynamic_cache_update(self._model, max_layer_updates=budget))

            for _ in range(8):
                swaps, requests = _accumulate(dynamic_cache_update(self._model, max_layer_updates=32))
                if swaps == 0 and requests == 0:
                    break

            round_requests = total_requests - req_before
            round_fallbacks = total_fallbacks - fb_before
            if round_requests > 0:
                round_rates.append(round_fallbacks / round_requests)

        fallback_rate = (total_fallbacks / total_requests) if total_requests > 0 else 0.0
        round_str = ",".join(f"{r * 100:.2f}%" for r in round_rates) if round_rates else "n/a"
        print(
            f"  Startup refine: {time.perf_counter() - t0:.1f}s "
            f"(calls={calls}, swaps={total_swaps}, fallback_rate={fallback_rate * 100:.2f}%, "
            f"round_rates={round_str})"
        )

    def _encode_prompt(self, prompt: str) -> list[int]:
        tokenizer = self._tokenizer
        add_special = tokenizer.bos_token is None or not prompt.startswith(tokenizer.bos_token)
        return tokenizer.encode(prompt, add_special_tokens=add_special)

    def _cache_key_from_body(self, body: dict) -> str:
        key = body.get("cache_key") or body.get("session_id")
        if key is None and isinstance(body.get("metadata"), dict):
            key = body["metadata"].get("cache_key") or body["metadata"].get("session_id")
        if key is None and body.get("user") is not None:
            key = str(body["user"])
        return str(key) if key is not None else "default"

    @staticmethod
    def _dynamic_update_policy(
        generation_tokens: int,
        last_fallback_rate: float,
        no_swap_streak: int,
        low_fallback_streak: int,
    ) -> tuple[int, int]:
        if last_fallback_rate >= 0.35:
            interval, budget = 1, 48
        elif last_fallback_rate >= 0.20:
            interval, budget = 1, 32
        elif last_fallback_rate >= 0.10:
            interval, budget = 2, 24
        elif generation_tokens <= 32:
            interval, budget = 2, 24
        elif generation_tokens <= 96:
            interval, budget = 3, 16
        else:
            interval, budget = 4, 12

        if no_swap_streak >= 4 and last_fallback_rate < 0.15:
            interval *= 2
            budget = max(8, budget // 2)

        if low_fallback_streak >= 6 and last_fallback_rate < 0.08:
            interval = max(interval, 12)
            budget = 8

        if low_fallback_streak >= 12 and last_fallback_rate < 0.05:
            interval = max(interval, 20)
            budget = 8

        return interval, budget

    def _stream(self, prompt_tokens: list[int], max_tokens: int,
                cache_key: str, **sampling):
        import mlx_lm as _mlx_lm
        from mlx_lm.sample_utils import make_sampler
        from .lazy_experts.core import dynamic_cache_update

        # No KV cache - always generate from scratch
        suffix = prompt_tokens
        prompt_cache = None

        kv_kwargs = {}
        if self._kv_bits is not None:
            kv_kwargs = dict(kv_bits=self._kv_bits, kv_group_size=64, quantized_kv_start=0)

        generated_tokens = []
        request_t0 = time.perf_counter()
        first_token_at = None
        prompt_tps = 0.0
        prompt_len = len(suffix)
        gen_tokens = 0
        decode_tps = 0.0
        dcu_calls = 0
        dcu_swaps = 0
        dcu_requests = 0
        dcu_fallbacks = 0
        no_swap_streak = 0
        low_fallback_streak = 0
        last_window_fallback_rate = 1.0
        complete = False

        try:
            # When suffix is empty (exact cache match), don't pass prompt to avoid empty array error
            prompt_arg = mx.array(suffix) if suffix else None
            for resp in _mlx_lm.stream_generate(
                self._model, self._tokenizer, prompt=prompt_arg,
                max_tokens=max_tokens, sampler=make_sampler(**sampling),
                prompt_cache=prompt_cache, **kv_kwargs,
            ):
                if first_token_at is None:
                    first_token_at = time.perf_counter()
                    prompt_tps = resp.prompt_tps
                    prompt_len = int(resp.prompt_tokens)
                generated_tokens.append(resp.token)
                gen_tokens = resp.generation_tokens
                decode_tps = resp.generation_tps
                yield resp

                interval, budget = self._dynamic_update_policy(
                    resp.generation_tokens,
                    last_window_fallback_rate,
                    no_swap_streak,
                    low_fallback_streak,
                )
                if resp.generation_tokens % interval != 0:
                    continue

                stats = dynamic_cache_update(self._model, max_layer_updates=budget)
                dcu_calls += 1
                swaps = sum(s.get("swaps", 0) for s in stats)
                fallbacks = sum(s.get("fallbacks", 0) for s in stats)
                requests = sum(s.get("requests", 0) for s in stats)
                dcu_swaps += swaps
                dcu_fallbacks += fallbacks
                dcu_requests += requests

                if requests > 0:
                    last_window_fallback_rate = fallbacks / requests

                if swaps == 0:
                    no_swap_streak += 1
                else:
                    no_swap_streak = 0

                if requests > 0:
                    if (fallbacks / requests) <= 0.005:
                        low_fallback_streak += 1
                    else:
                        low_fallback_streak = 0

            complete = True
        finally:
            if first_token_at is not None:
                prefill_ms = (prompt_len / prompt_tps * 1000.0) if prompt_tps > 0 else 0.0
                ttft_ms = (first_token_at - request_t0) * 1000.0
                fallback_rate = (dcu_fallbacks / dcu_requests) if dcu_requests > 0 else 0.0
                print(
                    f"  [telemetry] cache_key={cache_key} "
                    f"prefill={prefill_ms:.1f}ms ttft={ttft_ms:.1f}ms "
                    f"decode={decode_tps:.1f} tok/s tokens={gen_tokens} "
                    f"dcu_calls={dcu_calls} swaps={dcu_swaps} "
                    f"fallback_rate={fallback_rate * 100:.2f}%"
                )

    def _tokenize_messages(self, messages: list, tools: list | None = None) -> str:
        tokenizer = self._tokenizer
        if tokenizer.has_chat_template:
            kwargs = {"add_generation_prompt": True, "tokenize": False}
            if tools and tokenizer.has_tool_calling:
                kwargs["tools"] = tools
            if tokenizer.has_thinking:
                kwargs["enable_thinking"] = False
            return tokenizer.apply_chat_template(messages, **kwargs)
        parts = []
        for msg in messages:
            parts.append(f"{msg['role']}: {msg.get('content', '')}")
        parts.append("assistant:")
        return "\n".join(parts)

    def _check_input_length(self, input_tokens: int):
        if input_tokens > self._max_input_tokens:
            return JSONResponse({
                "type": "error",
                "error": {
                    "type": "invalid_request_error",
                    "message": f"Prompt too long: {input_tokens} tokens > {self._max_input_tokens} limit",
                },
            }, status_code=400)
        return None

    async def handle_models(self, request: Request) -> JSONResponse:
        return JSONResponse({
            "object": "list",
            "data": [{
                "id": self._model_id,
                "object": "model",
                "owned_by": "mlx-moe",
            }],
        })

    async def handle_chat_completions(self, request: Request):
        body = await request.json()
        messages = body["messages"]
        raw = body.get("max_tokens")
        if raw is None:
            raw = body.get("max_completion_tokens")
        max_tokens = min(raw if raw is not None else self._max_tokens, self._max_tokens)
        stream = body.get("stream", False)
        tools = body.get("tools")
        sampling = _sampling_kwargs(body, self._model_name, self._model_path)
        cache_key = self._cache_key_from_body(body)

        formatted = _format_messages(messages)
        prompt = self._tokenize_messages(formatted, tools=tools)
        prompt_tokens = self._encode_prompt(prompt)
        input_tokens = len(prompt_tokens)

        if err := self._check_input_length(input_tokens):
            return err

        print(
            f"  [openai] max_tokens={max_tokens} stream={stream} "
            f"tools={len(tools) if tools else 0} input_tokens={input_tokens} "
            f"cache_key={cache_key}"
        )

        if stream:
            return StreamingResponse(
                self._stream_openai(prompt_tokens, max_tokens, tools, cache_key, **sampling),
                media_type="text/event-stream",
            )

        async with self._lock:
            text, gen_tokens, tps = self._generate_sync(
                prompt_tokens, max_tokens, cache_key, **sampling
            )

        return JSONResponse({
            "id": _make_chatcmpl_id(),
            "object": "chat.completion",
            "model": self._model_id,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": input_tokens,
                "completion_tokens": gen_tokens,
                "total_tokens": input_tokens + gen_tokens,
            },
        })

    async def handle_messages(self, request: Request):
        body = await request.json()
        messages = body["messages"]
        system = body.get("system")
        if isinstance(system, list):
            system = "\n".join(
                b["text"] for b in system if b.get("type") == "text"
            )
        max_tokens = min(body.get("max_tokens", self._max_tokens), self._max_tokens)
        stream = body.get("stream", False)
        sampling = _sampling_kwargs(body, self._model_name, self._model_path)
        cache_key = self._cache_key_from_body(body)

        tools_anthropic = body.get("tools")
        tools_openai = _convert_tools_anthropic_to_openai(tools_anthropic) if tools_anthropic else None

        formatted = _format_messages(messages, system=system)
        prompt = self._tokenize_messages(formatted, tools=tools_openai)
        prompt_tokens = self._encode_prompt(prompt)
        input_tokens = len(prompt_tokens)

        if err := self._check_input_length(input_tokens):
            return err

        if not stream:
            max_tokens = min(max_tokens, 512)

        print(
            f"  [anthropic] max_tokens={max_tokens} stream={stream} "
            f"tools={len(tools_anthropic) if tools_anthropic else 0} "
            f"msgs={len(messages)} input_tokens={input_tokens} cache_key={cache_key}"
        )

        if stream:
            return StreamingResponse(
                self._stream_anthropic(
                    prompt_tokens, max_tokens, tools_openai, input_tokens,
                    cache_key, **sampling
                ),
                media_type="text/event-stream",
            )

        async with self._lock:
            text, gen_tokens, tps = self._generate_sync(
                prompt_tokens, max_tokens, cache_key, **sampling
            )

        return JSONResponse({
            "id": _make_msg_id(),
            "type": "message",
            "role": "assistant",
            "model": self._model_id,
            "content": [{"type": "text", "text": text}],
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": gen_tokens,
            },
        })

    def _generate_sync(
        self,
        prompt_tokens: list[int],
        max_tokens: int,
        cache_key: str,
        **sampling,
    ) -> tuple[str, int, float]:
        text = ""
        gen_tokens = 0
        tps = 0.0
        for resp in self._stream(
            prompt_tokens, max_tokens=max_tokens, cache_key=cache_key, **sampling
        ):
            text += resp.text
            gen_tokens = resp.generation_tokens
            tps = resp.generation_tps
        text = _strip_thinking(text)
        if gen_tokens > 1:
            print(f"  [{gen_tokens} tokens, {tps:.1f} tok/s]")
        return text, gen_tokens, tps

    async def _stream_openai(self, prompt_tokens: list[int], max_tokens: int,
                             tools: list | None, cache_key: str, **sampling):
        chat_id = _make_chatcmpl_id()
        created = int(time.time())

        chunk = {
            "id": chat_id, "object": "chat.completion.chunk",
            "created": created, "model": self._model_id,
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}],
        }
        yield f"data: {json.dumps(chunk)}\n\n"

        gen_tokens = 0
        tps = 0.0
        finish = "stop"

        tokenizer = self._tokenizer
        has_tools = tools and tokenizer.has_tool_calling
        in_tool_call = False
        tool_text = ""
        tool_idx = 0
        in_thinking = False
        suppressed_think_tokens = 0
        thinking_filter_enabled = tokenizer.has_thinking
        emitted_payload = False

        try:
            async with self._lock:
                for resp in self._stream(
                    prompt_tokens, max_tokens=max_tokens, cache_key=cache_key, **sampling
                ):
                    gen_tokens = resp.generation_tokens
                    tps = resp.generation_tps

                    text_piece = resp.text
                    if thinking_filter_enabled:
                        filtered, in_thinking = _filter_thinking_piece(
                            text_piece, in_thinking, tokenizer.think_start, tokenizer.think_end
                        )
                        if in_thinking or filtered != text_piece:
                            suppressed_think_tokens += 1
                        if in_thinking and suppressed_think_tokens >= _MAX_SUPPRESSED_THINK_TOKENS:
                            thinking_filter_enabled = False
                            in_thinking = False
                            filtered = text_piece
                            print(
                                "  [WARNING: disabling thinking suppression "
                                f"after {_MAX_SUPPRESSED_THINK_TOKENS} tokens]"
                            )
                        text_piece = filtered
                        if not text_piece:
                            continue

                    if has_tools and text_piece == tokenizer.tool_call_start:
                        in_tool_call = True
                        continue
                    if in_tool_call:
                        if text_piece == tokenizer.tool_call_end:
                            in_tool_call = False
                            parsed = tokenizer.tool_parser(tool_text, tools)
                            if not isinstance(parsed, list):
                                parsed = [parsed]
                            for tc in parsed:
                                tc_id = tc.pop("id", None) or f"call_{uuid.uuid4().hex[:12]}"
                                args_str = json.dumps(tc["arguments"], ensure_ascii=False)
                                chunk = {
                                    "id": chat_id, "object": "chat.completion.chunk",
                                    "created": created, "model": self._model_id,
                                    "choices": [{"index": 0, "delta": {
                                        "tool_calls": [{"index": tool_idx, "id": tc_id, "type": "function",
                                                        "function": {"name": tc["name"], "arguments": ""}}],
                                    }, "finish_reason": None}],
                                }
                                yield f"data: {json.dumps(chunk)}\n\n"
                                emitted_payload = True
                                chunk = {
                                    "id": chat_id, "object": "chat.completion.chunk",
                                    "created": created, "model": self._model_id,
                                    "choices": [{"index": 0, "delta": {
                                        "tool_calls": [{"index": tool_idx, "function": {"arguments": args_str}}],
                                    }, "finish_reason": None}],
                                }
                                yield f"data: {json.dumps(chunk)}\n\n"
                                emitted_payload = True
                                tool_idx += 1
                            tool_text = ""
                            finish = "tool_calls"
                            continue
                        tool_text += text_piece
                        continue

                    chunk = {
                        "id": chat_id, "object": "chat.completion.chunk",
                        "created": created, "model": self._model_id,
                        "choices": [{"index": 0, "delta": {"content": text_piece}, "finish_reason": None}],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                    emitted_payload = True
                    await asyncio.sleep(0)

                    if resp.finish_reason == "length":
                        finish = "length"

            if in_tool_call and tool_text:
                print(f"  [WARNING: tool call truncated at {gen_tokens} tokens, {len(tool_text)} chars buffered]")
                try:
                    parsed = tokenizer.tool_parser(tool_text, tools)
                    if not isinstance(parsed, list):
                        parsed = [parsed]
                    for tc in parsed:
                        tc_id = tc.pop("id", None) or f"call_{uuid.uuid4().hex[:12]}"
                        args_str = json.dumps(tc["arguments"], ensure_ascii=False)
                        chunk = {
                            "id": chat_id, "object": "chat.completion.chunk",
                            "created": created, "model": self._model_id,
                            "choices": [{"index": 0, "delta": {
                                "tool_calls": [{"index": tool_idx, "id": tc_id, "type": "function",
                                                "function": {"name": tc["name"], "arguments": ""}}],
                            }, "finish_reason": None}],
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"
                        emitted_payload = True
                        chunk = {
                            "id": chat_id, "object": "chat.completion.chunk",
                            "created": created, "model": self._model_id,
                            "choices": [{"index": 0, "delta": {
                                "tool_calls": [{"index": tool_idx, "function": {"arguments": args_str}}],
                            }, "finish_reason": None}],
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"
                        emitted_payload = True
                        tool_idx += 1
                    finish = "tool_calls"
                except Exception:
                    print("  [tool call unrecoverable, dropping]")

            if not emitted_payload and gen_tokens > 0:
                print(
                    f"  [WARNING: generated {gen_tokens} tokens but emitted no visible output; "
                    "model likely stayed in an unterminated control block]"
                )

            chunk = {
                "id": chat_id, "object": "chat.completion.chunk",
                "created": created, "model": self._model_id,
                "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
            }
            yield f"data: {json.dumps(chunk)}\n\n"
            yield "data: [DONE]\n\n"

            print(f"  [{gen_tokens} tokens, {tps:.1f} tok/s]")
        except (OSError, asyncio.CancelledError):
            print(f"  [client disconnected after {gen_tokens} tokens]")

    async def _stream_anthropic(self, prompt_tokens: list[int], max_tokens: int,
                                tools: list | None, input_tokens: int,
                                cache_key: str, **sampling):
        msg_id = _make_msg_id()

        event = {
            "type": "message_start",
            "message": {
                "id": msg_id, "type": "message", "role": "assistant",
                "content": [], "model": self._model_id, "stop_reason": None,
                "usage": {"input_tokens": input_tokens, "output_tokens": 0},
            },
        }
        yield f"event: message_start\ndata: {json.dumps(event)}\n\n"

        gen_tokens = 0
        tps = 0.0
        stop_reason = "end_turn"
        content_idx = 0

        tokenizer = self._tokenizer
        has_tools = tools and tokenizer.has_tool_calling
        in_tool_call = False
        tool_text = ""
        text_block_open = False
        in_thinking = False
        suppressed_think_tokens = 0
        thinking_filter_enabled = tokenizer.has_thinking

        try:
            async with self._lock:
                for resp in self._stream(
                    prompt_tokens, max_tokens=max_tokens, cache_key=cache_key, **sampling
                ):
                    gen_tokens = resp.generation_tokens
                    tps = resp.generation_tps

                    text_piece = resp.text
                    if thinking_filter_enabled:
                        filtered, in_thinking = _filter_thinking_piece(
                            text_piece, in_thinking, tokenizer.think_start, tokenizer.think_end
                        )
                        if in_thinking or filtered != text_piece:
                            suppressed_think_tokens += 1
                        if in_thinking and suppressed_think_tokens >= _MAX_SUPPRESSED_THINK_TOKENS:
                            thinking_filter_enabled = False
                            in_thinking = False
                            filtered = text_piece
                            print(
                                "  [WARNING: disabling thinking suppression "
                                f"after {_MAX_SUPPRESSED_THINK_TOKENS} tokens]"
                            )
                        text_piece = filtered
                        if not text_piece:
                            continue

                    if has_tools and text_piece == tokenizer.tool_call_start:
                        if text_block_open:
                            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': content_idx - 1})}\n\n"
                            text_block_open = False
                        in_tool_call = True
                        continue
                    if in_tool_call:
                        if text_piece == tokenizer.tool_call_end:
                            in_tool_call = False
                            parsed = tokenizer.tool_parser(tool_text, tools)
                            if not isinstance(parsed, list):
                                parsed = [parsed]
                            for tc in parsed:
                                tc_id = tc.pop("id", None) or f"toolu_{uuid.uuid4().hex[:24]}"
                                event = {
                                    "type": "content_block_start", "index": content_idx,
                                    "content_block": {"type": "tool_use", "id": tc_id, "name": tc["name"], "input": {}},
                                }
                                yield f"event: content_block_start\ndata: {json.dumps(event)}\n\n"
                                args_json = json.dumps(tc["arguments"], ensure_ascii=False)
                                event = {
                                    "type": "content_block_delta", "index": content_idx,
                                    "delta": {"type": "input_json_delta", "partial_json": args_json},
                                }
                                yield f"event: content_block_delta\ndata: {json.dumps(event)}\n\n"
                                yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': content_idx})}\n\n"
                                content_idx += 1
                            tool_text = ""
                            stop_reason = "tool_use"
                            continue
                        tool_text += text_piece
                        continue

                    if not text_block_open:
                        event = {
                            "type": "content_block_start", "index": content_idx,
                            "content_block": {"type": "text", "text": ""},
                        }
                        yield f"event: content_block_start\ndata: {json.dumps(event)}\n\n"
                        text_block_open = True
                        content_idx += 1

                    event = {
                        "type": "content_block_delta", "index": content_idx - 1,
                        "delta": {"type": "text_delta", "text": text_piece},
                    }
                    yield f"event: content_block_delta\ndata: {json.dumps(event)}\n\n"
                    await asyncio.sleep(0)

                    if resp.finish_reason == "length":
                        stop_reason = "max_tokens"

            # Salvage truncated tool calls — if the model hit the token cap
            # mid-tool-call, try to parse what we have before dropping it.
            if in_tool_call and tool_text:
                print(f"  [WARNING: tool call truncated at {gen_tokens} tokens, {len(tool_text)} chars buffered]")
                try:
                    parsed = tokenizer.tool_parser(tool_text, tools)
                    if not isinstance(parsed, list):
                        parsed = [parsed]
                    for tc in parsed:
                        tc_id = tc.pop("id", None) or f"toolu_{uuid.uuid4().hex[:24]}"
                        event = {
                            "type": "content_block_start", "index": content_idx,
                            "content_block": {"type": "tool_use", "id": tc_id, "name": tc["name"], "input": {}},
                        }
                        yield f"event: content_block_start\ndata: {json.dumps(event)}\n\n"
                        args_json = json.dumps(tc["arguments"], ensure_ascii=False)
                        event = {
                            "type": "content_block_delta", "index": content_idx,
                            "delta": {"type": "input_json_delta", "partial_json": args_json},
                        }
                        yield f"event: content_block_delta\ndata: {json.dumps(event)}\n\n"
                        yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': content_idx})}\n\n"
                        content_idx += 1
                    stop_reason = "tool_use"
                except Exception:
                    print(f"  [tool call unrecoverable, dropping]")

            if text_block_open:
                yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': content_idx - 1})}\n\n"

            event = {
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason},
                "usage": {"output_tokens": gen_tokens},
            }
            yield f"event: message_delta\ndata: {json.dumps(event)}\n\n"

            yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"

            print(f"  [{gen_tokens} tokens, {tps:.1f} tok/s]")
        except (OSError, asyncio.CancelledError):
            print(f"  [client disconnected after {gen_tokens} tokens]")


def run_server(model_name: str, host: str = "127.0.0.1", port: int = 8080,
               capacity: int | None = None, profile_path: str | None = None,
               pin_top_k: int | None = None,
                max_tokens: int = MAX_TOKENS_CAP,
                max_input_tokens: int = MAX_INPUT_TOKENS,
                kv_bits: int | None = None,
                warmup: str = "hybrid",
                shutdown_timeout: int = DEFAULT_SHUTDOWN_TIMEOUT):
    import uvicorn

    server = Server(model_name, capacity=capacity, profile_path=profile_path,
                    pin_top_k=pin_top_k,
                    max_tokens=max_tokens, max_input_tokens=max_input_tokens,
                    kv_bits=kv_bits, warmup=warmup)

    print(f"mlx-moe serve")
    print(f"  Model:    {model_name}")
    print(f"  Profile:  {server._profile_path or 'none'}")
    if pin_top_k is not None:
        print(f"  Pin top-k: {pin_top_k}")
    print(f"  Endpoint: http://{host}:{port}")
    print(f"  Limits:   {max_input_tokens} input, {max_tokens} output")
    print(f"  Shutdown: {shutdown_timeout}s graceful timeout")
    print()
    print("Loading model...")
    server.load()
    defaults = _sampling_defaults(model_name, server._model_path)
    if defaults:
        print(f"  Sampling: {defaults}")
    print(f"  Memory: {server._memory_gb:.1f} GB")
    print(f"  Ready.")
    print()

    app = Starlette(
        routes=[
            Route("/v1/models", server.handle_models, methods=["GET"]),
            Route("/v1/chat/completions", server.handle_chat_completions, methods=["POST"]),
            Route("/v1/messages", server.handle_messages, methods=["POST"]),
        ],
    )

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="warning",
        timeout_graceful_shutdown=shutdown_timeout,
    )
