#!/usr/bin/env python3
"""
Training Script - Train all three routers (KNN, SVM, MLP)

This script demonstrates how to:
1. Create routers
2. Train them on the built-in training data
3. Test them with sample queries
"""

from llmrouter import (
    KNNRouter,
    SVMRouter,
    MLPRouter,
    default_model_catalog,
    TRAINING_QUERIES,
    TRAINING_LABELS,
)


def main():
    print("\n" + "=" * 70)
    print("LLM Router - Training Script".center(70))
    print("=" * 70 + "\n")

    # Get the model catalog
    catalog = default_model_catalog()
    
    print("📊 Training Data Info:")
    print(f"   Total queries: {len(TRAINING_QUERIES)}")
    print(f"   Weak queries: {TRAINING_LABELS.count('weak')}")
    print(f"   Strong queries: {TRAINING_LABELS.count('strong')}")
    print()

    # ============================================================================
    # Train KNN Router
    # ============================================================================
    print("1️⃣  Training KNN Router (K-Nearest Neighbors)...\n")
    
    knn_router = KNNRouter(catalog, n_neighbors=3)
    knn_router.fit_texts(TRAINING_QUERIES, TRAINING_LABELS)
    print("   ✅ KNN Router trained successfully!")
    print(f"   Parameters: n_neighbors=3, metric='euclidean'\n")

    # ============================================================================
    # Train SVM Router
    # ============================================================================
    print("2️⃣  Training SVM Router (Support Vector Machine)...\n")
    
    svm_router = SVMRouter(catalog, kernel="rbf", C=1.0, gamma="scale")
    svm_router.fit_texts(TRAINING_QUERIES, TRAINING_LABELS)
    print("   ✅ SVM Router trained successfully!")
    print(f"   Parameters: kernel='rbf', C=1.0, gamma='scale'\n")

    # ============================================================================
    # Train MLP Router
    # ============================================================================
    print("3️⃣  Training MLP Router (Multi-Layer Perceptron)...\n")
    
    mlp_router = MLPRouter(catalog, hidden_layer_sizes=(12,), max_iter=1000)
    mlp_router.fit_texts(TRAINING_QUERIES, TRAINING_LABELS)
    print("   ✅ MLP Router trained successfully!")
    print(f"   Parameters: hidden_layer_sizes=(12,), max_iter=1000\n")

    # ============================================================================
    # Test All Routers
    # ============================================================================
    print("=" * 70)
    print("Testing All Trained Routers".center(70))
    print("=" * 70 + "\n")

    test_queries = [
        ("What is the capital of France?", "weak"),
        ("Tell me a joke", "weak"),
        ("Summarize this paragraph", "weak"),
        ("Design a scalable microservices architecture", "strong"),
        ("Debug a critical production issue", "strong"),
        ("Explain quantum computing in detail", "strong"),
    ]

    for query, expected in test_queries:
        print(f"📝 Query: \"{query}\"")
        print(f"   Expected: {expected}\n")

        # Route with KNN
        knn_decision = knn_router.route_text(query)
        knn_match = "✓" if knn_decision.model_name == expected else "✗"
        print(f"   {knn_match} KNN  → {knn_decision.model_name:8} ({knn_decision.model_info['openrouter_model']})")

        # Route with SVM
        svm_decision = svm_router.route_text(query)
        svm_match = "✓" if svm_decision.model_name == expected else "✗"
        print(f"   {svm_match} SVM  → {svm_decision.model_name:8} ({svm_decision.model_info['openrouter_model']})")

        # Route with MLP
        mlp_decision = mlp_router.route_text(query)
        mlp_match = "✓" if mlp_decision.model_name == expected else "✗"
        print(f"   {mlp_match} MLP  → {mlp_decision.model_name:8} ({mlp_decision.model_info['openrouter_model']})")

        print()

    # ============================================================================
    # Summary
    # ============================================================================
    print("=" * 70)
    print("✅ Training Complete!".center(70))
    print("=" * 70 + "\n")

    print("Your trained routers are ready to use!\n")

    print("Next steps:")
    print("  1. Try the interactive chatbot:")
    print("     python examples/chatbot.py")
    print()
    print("  2. Use routers in your code:")
    print("     from llmrouter import KNNRouter, default_model_catalog")
    print("     router = KNNRouter(default_model_catalog())")
    print("     router.fit_texts(TRAINING_QUERIES, TRAINING_LABELS)")
    print("     decision = router.route_text('Your query')")
    print()


if __name__ == "__main__":
    main()
