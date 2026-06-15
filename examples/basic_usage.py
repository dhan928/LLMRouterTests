from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llmrouter import (
    KNNRouter,
    MLPRouter,
    OpenRouterClient,
    SVMRouter,
    TRAINING_LABELS,
    TRAINING_QUERIES,
    default_model_catalog,
    load_env,
)


def build_routers():
    catalog = default_model_catalog()
    routers = {
        "knnrouter": KNNRouter(catalog, n_neighbors=3),
        "svmrouter": SVMRouter(catalog),
        "mlprouter": MLPRouter(catalog, hidden_layer_sizes=(12,), max_iter=1000),
    }
    for router in routers.values():
        router.fit_texts(TRAINING_QUERIES, TRAINING_LABELS)
    return routers


def main() -> None:
    parser = argparse.ArgumentParser(description="Route a real query with KNN, SVM, and MLP.")
    parser.add_argument("query", help="The real user query to route.")
    parser.add_argument(
        "--call-api",
        action="store_true",
        help="Actually call the selected OpenRouter model. Requires OPENROUTER_API_KEY.",
    )
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=512)
    args = parser.parse_args()

    load_env()
    routers = build_routers()
    client = OpenRouterClient()

    print(f"\n🔍 Query: '{args.query}'\n")
    print("-" * 70)

    for name, router in routers.items():
        decision = router.route_text(args.query)
        openrouter_model = decision.model_info["openrouter_model"]
        print(f"\n{name.upper()}:")
        print(f"  Model selected: {decision.model_name}")
        print(f"  OpenRouter: {openrouter_model}")

        if args.call_api:
            print(f"  Calling OpenRouter...")
            result = router.answer(
                args.query,
                client,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
            print(f"  Response:\n    {result.response}")
        print()


if __name__ == "__main__":
    main()
