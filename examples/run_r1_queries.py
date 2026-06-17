"""Run Router-R1 with your own queries.

Examples:
  # Single query
  python examples/run_r1_queries.py --query "Explain how transformers work"

  # Multiple queries
  python examples/run_r1_queries.py --query "What is quantum computing?" --query "Explain neural networks"

  # From a text file (one query per line)
  python examples/run_r1_queries.py --file my_queries.txt

  # Interactive mode
  python examples/run_r1_queries.py --interactive

  # Save results to JSON
  python examples/run_r1_queries.py --query "Hello" --output results.json

Requires:
  - OPENROUTER_API_KEY in .env
  - Optional: vLLM + CUDA GPU for local inference (set backend: vllm in config)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running from repo root or examples/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llmrouter import RouterR1, RouterR1Config, load_env


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "router_r1.yaml"


def load_queries_from_file(path: Path) -> list[str]:
    if path.suffix == ".jsonl":
        queries = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            queries.append(record.get("query", record.get("text", "")))
        return [q for q in queries if q]
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def print_result(result: dict, verbose: bool = False) -> None:
    print("=" * 60)
    print(f"Query: {result['query']}")
    print(f"Main model (reasoning): {result['model_name']}")
    if result.get("coordinator_mode"):
        print("Coordinator mode: ON (weak main model routes to expert models)")
    if result.get("route_models"):
        print(f"Route models (<search> calls, round-robin): {', '.join(result['route_models'])}")
        print("\nExpert prompt pool:")
        from llmrouter import RoutePromptPool

        for model_id in result["route_models"]:
            profile = RoutePromptPool.get_profile(model_id)
            print(f"  - {profile.name}: {profile.specialty}")
    elif result.get("route_model"):
        print(f"Route model (<search> calls): {result['route_model']}")
    print(f"Iterations: {result['iterations']}")
    tokens = result["total_tokens"]
    print(
        f"Tokens — prompt: {tokens['prompt_tokens']}, "
        f"completion: {tokens['completion_tokens']}, "
        f"route: {tokens['route_tokens']}"
    )
    if verbose and result.get("reasoning_trace"):
        print("\n--- Reasoning trace ---")
        for trace in result["reasoning_trace"]:
            print(f"\n[Iteration {trace.iteration}] via {result['model_name']}")
            print(trace.generation[:500] + ("..." if len(trace.generation) > 500 else ""))
            if trace.search_query:
                route_model = (
                    trace.route_info.get("model", "unknown")
                    if trace.route_info
                    else "unknown"
                )
                print(f"  >>> Routed to: {route_model}", end="")
                if trace.route_info and trace.route_info.get("expert_name"):
                    print(f" ({trace.route_info['expert_name']})", end="")
                if trace.route_info and trace.route_info.get("forced"):
                    print(" (auto-routed — coordinator forced search)", end="")
                print()
                if trace.route_info and trace.route_info.get("specialty"):
                    print(f"  >>> Expert role: {trace.route_info['specialty']}")
                print(f"  >>> Search query: {trace.search_query}")
                if trace.route_info and trace.route_info.get("result"):
                    snippet = trace.route_info["result"][:300]
                    print(f"  >>> Response: {snippet}{'...' if len(trace.route_info['result']) > 300 else ''}")
    print("\n--- Answer ---")
    print(result["response"])
    print()


def build_router(config_path: Path | None, model_id: str | None) -> RouterR1:
    if config_path and config_path.exists():
        router = RouterR1(yaml_path=str(config_path))
    else:
        import os

        router = RouterR1(
            config=RouterR1Config(
                backend="api",
                coordinator_mode=True,
                model_id=model_id or "meta-llama/llama-3.2-3b-instruct",
                route_models=[
                    "anthropic/claude-3.5-haiku",
                    "google/gemini-2.0-flash-001",
                    "openai/gpt-4.1",
                ],
                api_base=os.getenv("API_BASE", "https://openrouter.ai/api/v1"),
                api_key=os.getenv("OPENROUTER_API_KEY") or os.getenv("API_KEYS"),
            )
        )
    return router


def run_interactive(router: RouterR1, verbose: bool) -> list[dict]:
    print("Router-R1 interactive mode. Type a question and press Enter.")
    print("Commands: quit / exit / q to stop\n")
    results = []
    while True:
        try:
            query = input("Query> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not query:
            continue
        if query.lower() in {"quit", "exit", "q"}:
            break
        result = router.route_single(query, return_details=verbose)
        print_result(result, verbose=verbose)
        results.append(result)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Router-R1 with custom queries")
    parser.add_argument("--query", "-q", action="append", default=[], help="Query text (repeatable)")
    parser.add_argument("--file", "-f", type=Path, help="File with queries (.txt one per line, or .jsonl)")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive prompt loop")
    parser.add_argument("--config", "-c", type=Path, default=DEFAULT_CONFIG, help="YAML config path")
    parser.add_argument("--model", "-m", help="Override model_id from config")
    parser.add_argument("--output", "-o", type=Path, help="Save all results to JSON")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show full reasoning trace")
    args = parser.parse_args()

    load_env()

    queries: list[str] = list(args.query)
    if args.file:
        if not args.file.exists():
            print(f"Error: file not found: {args.file}", file=sys.stderr)
            return 1
        queries.extend(load_queries_from_file(args.file))

    if not queries and not args.interactive:
        parser.print_help()
        print("\nProvide --query, --file, or --interactive.", file=sys.stderr)
        return 1

    print("Loading Router-R1...")
    router = build_router(args.config if args.config.exists() else None, args.model)

    results: list[dict] = []
    if args.interactive:
        results.extend(run_interactive(router, args.verbose))

    if queries:
        batch = [{"query": q} for q in queries]
        results.extend(router.route_batch(batch, task_name="custom_queries", return_details=args.verbose))
        for result in results[-len(queries) :]:
            print_result(result, verbose=args.verbose)

    if args.output and results:
        # Serialize reasoning_trace dataclasses
        serializable = []
        for r in results:
            item = dict(r)
            if item.get("reasoning_trace"):
                item["reasoning_trace"] = [
                    {
                        "iteration": t.iteration,
                        "generation": t.generation,
                        "search_query": t.search_query,
                        "route_info": t.route_info,
                        "tokens_used": t.tokens_used,
                    }
                    for t in item["reasoning_trace"]
                ]
            serializable.append(item)
        args.output.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
        print(f"Saved {len(serializable)} result(s) to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
