"""
Detailed demonstration of the LLM Router with real text queries.

This example shows:
1. Training routers with real text queries
2. Routing different types of queries to weak or strong models
3. Calling OpenRouter with the selected model
4. Comparing router decisions across KNN, SVM, and MLP algorithms
"""

from __future__ import annotations

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


# Example queries of varying complexity
EXAMPLE_QUERIES = [
    # Simple queries (should route to weak model)
    "Say hello to the user",
    "What is the capital of France?",
    "Summarize this text in one sentence",
    
    # Complex queries (should route to strong model)
    "Design a microservices architecture for a real-time analytics platform",
    "Debug a critical production issue with memory leaks in a Python service",
    "Explain quantum computing and its applications in cryptography",
]


def build_routers() -> dict:
    """Build and train all three router types."""
    catalog = default_model_catalog()
    routers = {
        "KNN (n_neighbors=3)": KNNRouter(catalog, n_neighbors=3),
        "SVM (RBF kernel)": SVMRouter(catalog),
        "MLP (hidden=12)": MLPRouter(catalog, hidden_layer_sizes=(12,), max_iter=1000),
    }
    
    print("📚 Training routers with text-based queries...")
    for router in routers.values():
        router.fit_texts(TRAINING_QUERIES, TRAINING_LABELS)
    print(f"✓ Trained on {len(TRAINING_QUERIES)} queries\n")
    
    return routers


def demo_routing_only():
    """Demo: Route queries without calling OpenRouter API."""
    print("=" * 80)
    print("DEMO 1: Routing Queries (No API Calls)")
    print("=" * 80)
    
    routers = build_routers()
    
    for query in EXAMPLE_QUERIES:
        print(f"\n📝 Query: {query}")
        print("   Routing decisions:")
        
        for router_name, router in routers.items():
            decision = router.route_text(query)
            model = decision.model_info["openrouter_model"]
            print(f"      {router_name:20} → {decision.model_name:8} ({model})")


def demo_routing_with_api():
    """Demo: Route queries and call OpenRouter API for responses."""
    print("\n" + "=" * 80)
    print("DEMO 2: Routing + OpenRouter API Calls")
    print("=" * 80)
    print("\n⚠️  This requires OPENROUTER_API_KEY in .env file")
    print("   Set it as: OPENROUTER_API_KEY=your_api_key_here\n")
    
    load_env()
    routers = build_routers()
    client = OpenRouterClient()
    
    # Check if API key is set
    if not client.api_key:
        print("❌ OPENROUTER_API_KEY not found. Skipping API calls.")
        print("   To enable: Create a .env file with OPENROUTER_API_KEY=your_key")
        return
    
    print("✓ API key configured\n")
    
    # Demo with just first query to avoid high API costs
    query = "Explain the concept of neural networks in machine learning"
    print(f"📝 Query: {query}\n")
    
    for router_name, router in routers.items():
        print(f"\n{router_name}:")
        
        try:
            result = router.answer(
                query,
                client,
                temperature=0.7,
                max_tokens=256,
            )
            
            model_name = result.decision.model_name
            model_id = result.decision.model_info["openrouter_model"]
            
            print(f"  Model: {model_name} ({model_id})")
            print(f"  Response:\n    {result.response}")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")


def demo_custom_queries():
    """Demo: Route custom user queries."""
    print("\n" + "=" * 80)
    print("DEMO 3: Custom Query Routing")
    print("=" * 80)
    
    routers = build_routers()
    
    # Custom test queries
    custom_queries = [
        "Tell me a joke",
        "Write a paper on distributed consensus algorithms",
        "Give me a recipe for pancakes",
        "Solve this complex optimization problem using reinforcement learning",
    ]
    
    for query in custom_queries:
        print(f"\n📝 '{query}'")
        
        # Use KNN router for this demo
        router = routers["KNN (n_neighbors=3)"]
        decision = router.route_text(query)
        model = decision.model_info["openrouter_model"]
        
        print(f"   → {decision.model_name:8} model: {model}")


def demo_model_catalog():
    """Show the configured model catalog."""
    print("\n" + "=" * 80)
    print("DEMO 4: Model Catalog Configuration")
    print("=" * 80)
    
    catalog = default_model_catalog()
    
    print("\nConfigured Models:")
    for model_type, config in catalog.items():
        print(f"\n  {model_type.upper()}:")
        for key, value in config.items():
            print(f"    {key}: {value}")
    
    print("\n💡 You can override these with environment variables:")
    print("   OPENROUTER_WEAK_MODEL=...")
    print("   OPENROUTER_STRONG_MODEL=...")


def main() -> None:
    """Run all demonstrations."""
    print("\n🚀 LLM Router - Comprehensive Demo\n")
    
    demo_routing_only()
    demo_custom_queries()
    demo_model_catalog()
    demo_routing_with_api()
    
    print("\n" + "=" * 80)
    print("✓ Demo complete!")
    print("=" * 80)
    print("\n📖 To use in your own code:")
    print("""
    from llmrouter import KNNRouter, OpenRouterClient, default_model_catalog

    # Build router
    router = KNNRouter(default_model_catalog())
    router.fit_texts(training_queries, training_labels)

    # Route a query
    decision = router.route_text("Your query here")
    print(f"Route to: {decision.model_name}")

    # Get response from OpenRouter
    client = OpenRouterClient()
    response = router.answer("Your query here", client)
    print(response.response)
    """)


if __name__ == "__main__":
    main()
