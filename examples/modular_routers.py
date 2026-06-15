#!/usr/bin/env python3
"""
Modular Router Usage Example - Use the unified RouterManager to select routers.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llmrouter import (
    RouterManager,
    OpenRouterClient,
    default_model_catalog,
    TRAINING_QUERIES,
    TRAINING_LABELS,
    load_env,
)


def main():
    print("\n" + "=" * 70)
    print("LLM Router - Modular System Example".center(70))
    print("=" * 70 + "\n")

    # Initialize the RouterManager
    manager = RouterManager()
    catalog = default_model_catalog()

    print("📋 Available routers:", manager.list_available())
    print()

    # Example 1: Create a single router
    print("1️⃣  Creating a KNN router...")
    knn_router = manager.create_router("knn", catalog, n_neighbors=3)
    knn_router.fit_texts(TRAINING_QUERIES, TRAINING_LABELS)
    print(f"   ✓ KNN router created and trained\n")

    # Example 2: Create another router
    print("2️⃣  Creating an SVM router...")
    svm_router = manager.create_router("svm", catalog, kernel="rbf", C=1.0)
    svm_router.fit_texts(TRAINING_QUERIES, TRAINING_LABELS)
    print(f"   ✓ SVM router created and trained\n")

    # Example 3: Create all three routers at once
    print("3️⃣  Creating MLP router...")
    mlp_router = manager.create_router("mlp", catalog, hidden_layer_sizes=(12,), max_iter=1000)
    mlp_router.fit_texts(TRAINING_QUERIES, TRAINING_LABELS)
    print(f"   ✓ MLP router created and trained\n")

    # Show what's created
    print(f"📊 Created routers: {manager.list_created()}\n")

    # Example 4: Use routers individually
    test_query = "Design a scalable microservices architecture"
    print(f"🔍 Test query: \"{test_query}\"\n")

    for router_type in manager.list_created():
        router = manager.get_router(router_type)
        decision = router.route_text(test_query)
        print(f"   {router_type.upper():6} → {decision.model_name:8} ({decision.model_info['openrouter_model']})")

    print()

    # Example 5: Switch routers dynamically
    print("5️⃣  Dynamically switching routers for different queries:\n")
    
    queries = [
        ("What is 2+2?", "weak"),
        ("Design a database", "strong"),
    ]
    
    for query, expected in queries:
        knn = manager.get_router("knn")
        decision = knn.route_text(query)
        match = "✓" if decision.model_name == expected else "✗"
        print(f"   {match} \"{query}\" → {decision.model_name} (expected: {expected})")

    print()

    # Example 6: Get live responses (if API key is set)
    print("6️⃣  Optional: Get live responses with OpenRouter API\n")
    
    load_env()
    client = OpenRouterClient()
    
    if client.api_key:
        print("   ✓ API key configured\n")
        
        query = "What is machine learning?"
        router = manager.get_router("knn")
        response = router.answer(query, client)
        
        print(f"   Query: {query}")
        print(f"   Routed to: {response.decision.model_name}")
        print(f"   Response: {response.response[:100]}...\n")
    else:
        print("   ℹ️  API key not configured (optional)")
        print("   Set OPENROUTER_API_KEY in .env to enable\n")

    print("=" * 70)
    print("✅ Example complete!".center(70))
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
