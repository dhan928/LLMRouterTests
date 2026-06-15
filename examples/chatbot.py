#!/usr/bin/env python3
"""
Interactive LLM Router Chatbot - Test routing decisions in real-time.

Ask queries and see which model (weak or strong) they get routed to.
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


class RouterChatbot:
    def __init__(self):
        self.catalog = default_model_catalog()
        self.routers = {
            "KNN": KNNRouter(self.catalog, n_neighbors=3),
            "SVM": SVMRouter(self.catalog),
            "MLP": MLPRouter(self.catalog, hidden_layer_sizes=(12,), max_iter=1000),
        }
        
        # Train all routers
        for router in self.routers.values():
            router.fit_texts(TRAINING_QUERIES, TRAINING_LABELS)
        
        self.client = None
        self.api_key_set = False
        
    def setup_api(self):
        """Try to load OpenRouter API credentials."""
        load_env()
        try:
            self.client = OpenRouterClient()
            if self.client.api_key:
                self.api_key_set = True
                return True
        except Exception:
            pass
        return False
    
    def print_header(self):
        """Print welcome banner."""
        print("\n" + "=" * 70)
        print("🤖  LLM Router Interactive Chatbot".center(70))
        print("=" * 70)
        print("\nWelcome! Type your questions and watch how they get routed.")
        print("Each query will be classified as 'weak' (simple) or 'strong' (complex).\n")
        
        if self.api_key_set:
            print("✅ OpenRouter API configured - you can get live responses!")
            print("   Type 'api' after a query to get a real LLM response\n")
        else:
            print("ℹ️  Set OPENROUTER_API_KEY in .env to enable live responses\n")
        
        print("Commands:")
        print("  • Type a question or statement")
        print("  • Type 'api' after routing to call OpenRouter (if configured)")
        print("  • Type 'compare' to see all three routers")
        print("  • Type 'help' for more options")
        print("  • Type 'quit' or 'exit' to leave")
        print("\n" + "-" * 70 + "\n")
    
    def route_query(self, query: str) -> dict:
        """Route a query with all three routers."""
        results = {}
        for name, router in self.routers.items():
            decision = router.route_text(query)
            results[name] = {
                "model": decision.model_name,
                "openrouter": decision.model_info.get("openrouter_model"),
                "router": router,
            }
        return results
    
    def display_routing(self, query: str, results: dict, highlight: str | None = None):
        """Display routing results nicely."""
        print(f"\n📝 Query: \"{query}\"\n")
        
        # Find consensus
        models = [r["model"] for r in results.values()]
        consensus = models[0] if len(set(models)) == 1 else "mixed"
        
        if consensus == "mixed":
            print("⚠️  Routers disagree on routing:\n")
        else:
            emoji = "🟢" if consensus == "weak" else "🔴"
            model_name = results["KNN"]["openrouter"]
            print(f"{emoji} Consensus: {consensus.upper()} model ({model_name})\n")
        
        # Show each router
        for router_name, result in results.items():
            prefix = "→ " if highlight == router_name else "  "
            model = result["model"]
            model_id = result["openrouter"]
            
            if model == "weak":
                emoji = "⚡"
            else:
                emoji = "🧠"
            
            print(f"{prefix}{router_name:6} {emoji} {model:8} ({model_id})")
    
    def call_api(self, query: str, router_name: str, result: dict):
        """Call OpenRouter API with the selected model."""
        if not self.api_key_set:
            print("\n❌ API key not configured. Set OPENROUTER_API_KEY in .env")
            return
        
        try:
            print(f"\n🔄 Calling OpenRouter with {router_name} router...")
            
            router = result["router"]
            response = router.answer(
                query,
                self.client,
                temperature=0.7,
                max_tokens=256,
            )
            
            model = response.decision.model_name
            model_id = response.decision.model_info["openrouter_model"]
            
            print(f"\n✨ Response from {model} model ({model_id}):\n")
            print(f"   {response.response}\n")
            
        except Exception as e:
            print(f"\n❌ Error calling API: {e}\n")
    
    def show_help(self):
        """Show help information."""
        print("\n" + "=" * 70)
        print("HELP - LLM Router Commands".center(70))
        print("=" * 70)
        print("""
Basic Usage:
  • Just type any question or statement
  • Routers will classify it as 'weak' or 'strong'

Commands:
  • 'compare'  - Show detailed comparison of all three routers
  • 'api'      - Get a live response (after routing a query)
  • 'help'     - Show this help message
  • 'quit'     - Exit the chatbot

Examples:
  • "What is the capital of France?" → weak (simple fact)
  • "Design a scalable microservices architecture" → strong (complex)
  • "Debug a critical production issue" → strong (reasoning needed)
  • "Tell me a joke" → weak (simple response)

Router Types:
  • KNN (K-Nearest Neighbors) - Fast and simple
  • SVM (Support Vector Machine) - Strong generalization
  • MLP (Multi-Layer Perceptron) - Learns complex patterns

Models:
  • Weak:   openai/gpt-4o-mini  (fast, cheap)
  • Strong: openai/gpt-4.1      (powerful, expensive)
""")
        print("-" * 70 + "\n")
    
    def show_comparison(self):
        """Show router comparison."""
        print("\n" + "=" * 70)
        print("Router Comparison".center(70))
        print("=" * 70)
        print("""
KNN Router (K-Nearest Neighbors)
  ✓ Speed:        ⚡⚡⚡ Very fast
  ✓ Accuracy:     ⭐⭐⭐ Good
  ✓ Best for:     Quick decisions on balanced data
  ✓ Parameter:    n_neighbors=3

SVM Router (Support Vector Machine)
  ✓ Speed:        ⚡⚡ Moderate
  ✓ Accuracy:     ⭐⭐⭐⭐ Excellent generalization
  ✓ Best for:     Robust routing across diverse queries
  ✓ Parameter:    kernel='rbf', C=1.0

MLP Router (Multi-Layer Perceptron)
  ✓ Speed:        ⚡ Slower
  ✓ Accuracy:     ⭐⭐⭐⭐⭐ Best for complex patterns
  ✓ Best for:     Learning non-linear relationships
  ✓ Parameter:    hidden_layer_sizes=(12,), max_iter=1000

Recommendation: SVM is the best all-rounder for most cases.
""")
        print("-" * 70 + "\n")
    
    def run(self):
        """Main chatbot loop."""
        self.print_header()
        self.setup_api()
        
        last_results = None
        last_query = None
        
        try:
            while True:
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.lower() in ["quit", "exit", "q"]:
                    print("\n👋 Goodbye!\n")
                    break
                
                if user_input.lower() == "help":
                    self.show_help()
                    continue
                
                if user_input.lower() == "compare":
                    self.show_comparison()
                    continue
                
                if user_input.lower() == "api":
                    if last_results is None:
                        print("\n❌ Please ask a query first!\n")
                        continue
                    
                    # Use KNN for API call
                    self.call_api(last_query, "KNN", last_results["KNN"])
                    continue
                
                # Route the query
                results = self.route_query(user_input)
                self.display_routing(user_input, results)
                
                last_results = results
                last_query = user_input
                
                print("\n💡 Tip: Type 'api' to get a live response from the selected model")
                print("   Type 'help' for more options\n")
        
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!\n")


def main():
    chatbot = RouterChatbot()
    chatbot.run()


if __name__ == "__main__":
    main()
