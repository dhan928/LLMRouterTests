#!/usr/bin/env python
"""Test script to verify Router implementations work correctly."""

import sys

def test_base_routers():
    """Test KNN, SVM, MLP routers."""
    print("=" * 60)
    print("Testing Base Routers (KNN, SVM, MLP)")
    print("=" * 60)
    
    try:
        from llmrouter import KNNRouter, SVMRouter, MLPRouter, RouterManager
        print("✓ Imported routers successfully")
        
        # Test data
        model_catalog = {
            "gpt-4": {"openrouter_model": "openai/gpt-4"},
            "gpt-3.5": {"openrouter_model": "openai/gpt-3.5-turbo"},
        }
        
        # Create routers
        knn = KNNRouter(model_catalog)
        svm = SVMRouter(model_catalog)
        mlp = MLPRouter(model_catalog)
        print("✓ Created KNN, SVM, MLP routers")
        
        # Test data for fitting
        training_queries = [
            "What is machine learning?",
            "Explain neural networks",
            "How do transformers work?",
            "What is deep learning?",
        ]
        labels = ["gpt-4", "gpt-4", "gpt-3.5", "gpt-3.5"]
        
        # Fit routers
        knn.fit_texts(training_queries, labels)
        svm.fit_texts(training_queries, labels)
        mlp.fit_texts(training_queries, labels)
        print("✓ Fitted all routers with training data")
        
        # Test routing
        test_query = "Explain artificial intelligence"
        knn_decision = knn.route_text(test_query)
        svm_decision = svm.route_text(test_query)
        mlp_decision = mlp.route_text(test_query)
        
        print(f"\nRouting test query: '{test_query}'")
        print(f"  KNN routed to: {knn_decision.model_name}")
        print(f"  SVM routed to: {svm_decision.model_name}")
        print(f"  MLP routed to: {mlp_decision.model_name}")
        
        # Test RouterManager
        manager = RouterManager()
        manager.create_all(model_catalog)
        print(f"\n✓ RouterManager created all routers: {manager.list_created()}")
        
        print("\n✓ All base router tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Error testing base routers: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_router_r1_imports():
    """Test if Router-R1 can be imported."""
    print("\n" + "=" * 60)
    print("Testing Router-R1 Imports")
    print("=" * 60)
    
    try:
        from llmrouter import (
            RouterR1, RouterR1Config, RouterR1Response,
            PromptPool, RouteService
        )
        print("✓ Successfully imported Router-R1 components")
        
        # Check if vllm is available
        try:
            import vllm
            print("✓ vllm is installed")
            return True
        except ImportError:
            print("⚠ vllm not installed (required for Router-R1 functionality)")
            print("  Install with: pip install vllm==0.6.3 torch==2.4.0")
            return False
            
    except Exception as e:
        print(f"✗ Error importing Router-R1: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("LLMRouter Implementation Test Suite")
    print("=" * 60 + "\n")
    
    results = []
    
    # Test base routers
    results.append(("Base Routers", test_base_routers()))
    
    # Test Router-R1 imports
    results.append(("Router-R1", test_router_r1_imports()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(passed for _, passed in results)
    if all_passed:
        print("\n✓ All tests passed!")
    else:
        print("\n✗ Some tests failed - see details above")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
