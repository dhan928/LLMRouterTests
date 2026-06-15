#!/usr/bin/env python3
"""
Quick demo showing the chatbot in action with sample queries.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llmrouter import (
    KNNRouter,
    MLPRouter,
    SVMRouter,
    TRAINING_LABELS,
    TRAINING_QUERIES,
    default_model_catalog,
)


DEMO_QUERIES = [
    ("What is the capital of France?", "weak"),
    ("How do I make pancakes?", "weak"),
    ("Design a scalable microservices architecture", "strong"),
    ("Debug a critical production memory leak", "strong"),
    ("Explain quantum entanglement", "strong"),
]


def main():
    print("\n" + "=" * 70)
    print("LLM Router - Chatbot Demo".center(70))
    print("=" * 70)
    print("\nThis shows how the routing system works.\n")
    
    # Setup routers
    catalog = default_model_catalog()
    routers = {
        "KNN": KNNRouter(catalog, n_neighbors=3),
        "SVM": SVMRouter(catalog),
        "MLP": MLPRouter(catalog, hidden_layer_sizes=(12,), max_iter=1000),
    }
    
    for router in routers.values():
        router.fit_texts(TRAINING_QUERIES, TRAINING_LABELS)
    
    # Demo queries
    for query, expected in DEMO_QUERIES:
        print(f"📝 Query: \"{query}\"")
        print(f"   Expected: {expected}\n")
        
        results = {}
        for name, router in routers.items():
            decision = router.route_text(query)
            results[name] = decision.model_name
            model_id = decision.model_info.get("openrouter_model")
            
            emoji = "⚡" if decision.model_name == "weak" else "🧠"
            print(f"   {name:6} {emoji} {decision.model_name:8} ({model_id})")
        
        # Check consensus
        if len(set(results.values())) == 1:
            print(f"   ✓ All routers agree: {results['KNN']}")
        else:
            print(f"   ⚠️  Routers disagree")
        
        print()
    
    print("-" * 70)
    print("\n🚀 Try the interactive chatbot:\n")
    print("   python examples/chatbot.py\n")
    print("-" * 70 + "\n")


if __name__ == "__main__":
    main()
