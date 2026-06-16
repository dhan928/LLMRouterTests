"""Example: Using Router-R1 for multi-round reasoning.

This example demonstrates how to use the Router-R1 router for queries
that require iterative reasoning and dynamic routing.
"""

from llmrouter import RouterR1, RouterR1Config
import yaml
import os


def example_basic_usage():
    """Basic usage of Router-R1."""
    
    # Create a minimal YAML config
    config_data = {
        'hparam': {
            'model_id': 'Qwen/Qwen2.5-7B-Instruct',
            'api_base': os.getenv('API_BASE', 'https://api.openai.com/v1'),
            'api_key': os.getenv('API_KEYS', ''),
            'max_iterations': 5,
            'temperature': 1.0,
            'max_tokens': 512,
        }
    }
    
    # Initialize router
    config = RouterR1Config(**config_data['hparam'])
    router = RouterR1(config=config)
    
    # Route a single query
    query = "Explain how transformers work in machine learning"
    result = router.route_single(query, return_details=True)
    
    print(f"Query: {result['query']}")
    print(f"Model: {result['model_name']}")
    print(f"\nFinal Answer:\n{result['response']}")
    print(f"\nToken Usage:")
    print(f"  - Prompt: {result['total_tokens']['prompt_tokens']}")
    print(f"  - Completion: {result['total_tokens']['completion_tokens']}")
    print(f"  - Route: {result['total_tokens']['route_tokens']}")
    print(f"  - Total: {sum(result['total_tokens'].values())}")


def example_batch_routing():
    """Batch routing with Router-R1."""
    
    config = RouterR1Config(
        model_id='Qwen/Qwen2.5-7B-Instruct',
        api_base=os.getenv('API_BASE'),
        api_key=os.getenv('API_KEYS'),
    )
    router = RouterR1(config=config)
    
    queries = [
        {"query": "What is quantum computing?"},
        {"query": "Explain neural networks"},
        {"query": "How do transformers differ from RNNs?"},
    ]
    
    results = router.route_batch(queries, task_name="ml_questions")
    
    for i, result in enumerate(results, 1):
        print(f"\n{'='*60}")
        print(f"Query {i}: {result['query']}")
        print(f"Iterations: {result['iterations']}")
        print(f"Response: {result['response'][:200]}...")
        print(f"Total Tokens: {sum(result['total_tokens'].values())}")


def example_with_routing_details():
    """Detailed example showing reasoning trace."""
    
    config = RouterR1Config(
        model_id='Qwen/Qwen2.5-7B-Instruct',
        api_base=os.getenv('API_BASE'),
        api_key=os.getenv('API_KEYS'),
        max_iterations=3,
    )
    router = RouterR1(config=config)
    
    query = "What are the main differences between supervised and unsupervised learning?"
    result = router.route_single(query, return_details=True)
    
    print(f"Query: {query}\n")
    
    # Show reasoning trace
    for trace in result['reasoning_trace']:
        print(f"\n--- Iteration {trace.iteration} ---")
        print(f"Generation:\n{trace.generation[:300]}...")
        
        if trace.search_query:
            print(f"\nSearch Query: {trace.search_query}")
            if trace.route_info:
                print(f"Retrieved Info: {trace.route_info.get('result', '')[:200]}...")
        
        print(f"\nTokens - Prompt: {trace.tokens_used.get('prompt_tokens', 0)}, "
              f"Completion: {trace.tokens_used.get('completion_tokens', 0)}")
    
    print(f"\n\nFinal Answer:\n{result['response']}")


def example_route_only():
    """Example using route-only mode (without calling LLM)."""
    
    config = RouterR1Config(
        model_id='Qwen/Qwen2.5-7B-Instruct',
        api_base=os.getenv('API_BASE'),
        api_key=os.getenv('API_KEYS'),
    )
    router = RouterR1(config=config)
    
    query = "What is quantum computing?"
    
    # Route without performing full LLM generation
    result = router.route_single(query, route_only=True)
    
    print(f"Query: {query}")
    print(f"Model Routed To: {result['model_name']}")
    print(f"Iterations: {result['iterations']}")


def example_from_yaml():
    """Load configuration from YAML file."""
    
    # Example YAML content
    yaml_content = """
data_path:
  query_data_test: 'data/example_data/query_data/default_query_test.jsonl'

hparam:
  model_id: "Qwen/Qwen2.5-7B-Instruct"
  api_base: "https://api.openai.com/v1"
  api_key: "${ROUTING_API_KEY}"
  max_iterations: 5
  temperature: 1.0
  max_tokens: 512
"""
    
    # Save to temporary file
    with open('router_r1_config.yaml', 'w') as f:
        f.write(yaml_content)
    
    try:
        router = RouterR1(yaml_path='router_r1_config.yaml')
        query = "Explain machine learning"
        result = router.route_single(query)
        print(f"Response: {result['response']}")
    finally:
        os.remove('router_r1_config.yaml')


if __name__ == "__main__":
    print("Router-R1 Examples")
    print("=" * 60)
    
    # Uncomment to run specific examples
    # Note: Requires GPU and proper API configuration
    
    print("\n1. Basic Usage:")
    # example_basic_usage()
    
    print("\n2. Batch Routing:")
    # example_batch_routing()
    
    print("\n3. With Routing Details:")
    # example_with_routing_details()
    
    print("\n4. Route Only Mode:")
    # example_route_only()
    
    print("\n5. From YAML Configuration:")
    # example_from_yaml()
    
    print("\nExamples are commented out - uncomment to run with proper GPU and API setup")
