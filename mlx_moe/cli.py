import argparse
import sys


def main():
    parser = argparse.ArgumentParser(prog="mlx-moe")
    sub = parser.add_subparsers(dest="command")

    serve = sub.add_parser("serve", help="Start OpenAI/Anthropic-compatible API server")
    serve.add_argument("model", help="HuggingFace model name (e.g. mlx-community/Qwen3-Coder-Next-4bit)")
    serve.add_argument("--port", type=int, default=8080)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--capacity", type=int, default=None,
                       help="Experts cached per MoE layer (auto-selected if omitted)")
    serve.add_argument("--profile", default=None, help="Path to expert profile JSON")
    serve.add_argument("--pin-top-k", type=int, default=32,
                       help="Pin top-K profile experts per layer (default: 32, 0 disables pinning).")
    serve.add_argument("--max-tokens", type=int, default=4096,
                       help="Max output tokens per request (default: 4096)")
    serve.add_argument("--max-input-tokens", type=int, default=16384,
                       help="Max input tokens — rejects requests over this (default: 16384)")
    serve.add_argument("--kv-bits", type=int, default=None,
                       help="Quantize KV cache to N bits (8 recommended). Saves ~45%% KV memory.")
    serve.add_argument("--shutdown-timeout", type=int, default=5,
                       help="Graceful shutdown timeout in seconds before cancelling active requests (default: 5).")
    serve.add_argument("--warmup", choices=["hybrid", "full", "none"], default="hybrid",
                       help="Warmup strategy: hybrid (default, profile + real generation), "
                            "full (Phase 2 LCP, ~90s, 0%% fallback), none (profile/discovery only)")

    args = parser.parse_args()

    if args.command == "serve":
        from .server import run_server
        run_server(
            model_name=args.model,
            host=args.host,
            port=args.port,
            capacity=args.capacity,
            profile_path=args.profile,
            pin_top_k=args.pin_top_k,
            max_tokens=args.max_tokens,
            max_input_tokens=args.max_input_tokens,
            kv_bits=args.kv_bits,
            warmup=args.warmup,
            shutdown_timeout=args.shutdown_timeout,
        )
    else:
        parser.print_help()
        sys.exit(1)
