# Router-R1: Multi-Round Reasoning Router

## Overview

Router-R1 is an advanced agentic routing system that combines iterative reasoning with dynamic routing to external knowledge sources. Unlike traditional routers that make a single routing decision, Router-R1 uses a reasoning loop where the model generates search queries, retrieves information from specialized routing pools, and iteratively refines its answer through multiple reasoning steps.

### Key Features

- **Multi-Round Reasoning**: Iteratively reasons through complex queries (up to 5 iterations)
- **Dynamic Routing**: Can query external routing pools multiple times during reasoning
- **Agentic Behavior**: Uses XML tags (`<search>`, `<answer>`, `<information>`) to control flow
- **Explainable**: Full reasoning trace shows the decision-making process
- **Token Tracking**: Comprehensive tracking of vLLM and routing API tokens
- **Zero-Shot**: No training required - works out of the box with appropriate prompting
- **Model Agnostic**: Supports any vLLM-compatible chat model

## Architecture

```
Query → vLLM Generation → [Check for tags] → Reasoning Loop
                                ↓
                          [<search> tag?]
                                ↓
                          Routing Pool API
                                ↓
                          [<information>] appended
                                ↓
                          Continue iteration
                                ↓
                          [<answer> tag?]
                                ↓
                          Return Final Answer
```

## Installation

### Dependencies

Router-R1 requires:

```bash
pip install vllm==0.6.3 torch==2.4.0 openai>=1.0 pyyaml
```

### GPU Requirements

- CUDA capable GPU (8GB+ VRAM recommended)
- vLLM requires GPU - no CPU support
- Multi-GPU support via tensor parallelism

## Configuration

### YAML Configuration

Create a `router_r1.yaml` configuration file:

```yaml
data_path:
  query_data_test: 'data/example_data/query_data/default_query_test.jsonl'

hparam:
  model_id: "Qwen/Qwen2.5-7B-Instruct"          # vLLM model ID (HuggingFace)
  api_base: "https://api.openai.com/v1"         # Routing pool API base URL
  api_key: "${OPENROUTER_API_KEY}"              # Use environment variable
  max_iterations: 5                              # Max reasoning iterations
  temperature: 1.0                               # Generation temperature
  max_tokens: 512                                # Max tokens per generation
  top_p: 1.0                                     # Top-p sampling
  tensor_parallel_size: null                     # Auto-detect GPU count
  gpu_memory_utilization: 0.9                    # GPU memory usage ratio
```

### Environment Variables

Set API credentials via environment:

```bash
export OPENROUTER_API_KEY="your-api-key"
export API_BASE="https://api.openai.com/v1"
```

Or use generic names (same as other routers):

```bash
export API_KEYS="your-api-key"
export API_BASE="https://api.openai.com/v1"
```

## Usage

### Python API - Basic Usage

```python
from llmrouter import RouterR1, RouterR1Config

# Initialize from YAML
router = RouterR1(yaml_path="configs/router_r1.yaml")

# Or from config object
config = RouterR1Config(
    model_id="Qwen/Qwen2.5-7B-Instruct",
    api_base="https://api.openai.com/v1",
    api_key="your-key"
)
router = RouterR1(config=config)

# Route a single query
result = router.route_single(
    query="Explain transformers in machine learning",
    return_details=True
)

print(f"Model: {result['model_name']}")
print(f"Response: {result['response']}")
print(f"Total Tokens: {result['total_tokens']}")
print(f"Iterations: {result['iterations']}")
```

### Python API - Batch Routing

```python
# Route multiple queries
queries = [
    {"query": "What is quantum computing?"},
    {"query": "Explain neural networks"},
    {"query": "How do transformers work?"},
]

results = router.route_batch(queries, task_name="ml_questions")

for result in results:
    print(f"Query: {result['query']}")
    print(f"Response: {result['response']}")
    print(f"Tokens: {sum(result['total_tokens'].values())}")
    print("-" * 60)
```

### Python API - Detailed Reasoning Trace

```python
result = router.route_single(
    query="Explain machine learning",
    return_details=True
)

# Access reasoning trace
for trace in result['reasoning_trace']:
    print(f"Iteration {trace.iteration}:")
    print(f"  Generation: {trace.generation[:100]}...")
    if trace.search_query:
        print(f"  Search Query: {trace.search_query}")
    print(f"  Tokens - Prompt: {trace.tokens_used['prompt_tokens']}, "
          f"Completion: {trace.tokens_used['completion_tokens']}")
```

### CLI Usage

Note: CLI integration requires the main LLMRouter package setup. See main README for CLI examples.

## How It Works

### Routing Mechanism

1. **Initial Prompt**: Format the user query with system instructions
2. **Iterative Loop** (up to 5 iterations):
   - **Generate**: Model generates text including optional `<search>query</search>`
   - **Route**: If search tag found, query the routing pool API
   - **Augment**: Append retrieved info as `<information>results</information>`
   - **Continue**: Feed augmented text back for next iteration
3. **Termination**: Loop ends when model outputs `<answer>final answer</answer>`
4. **Output**: Return complete reasoning trace with final answer

### Token Tracking

Router-R1 tracks three types of tokens:

- **prompt_tokens**: vLLM input tokens (summed across all iterations)
- **completion_tokens**: vLLM output tokens (summed across all iterations)
- **route_tokens**: External routing API tokens

```python
stats = router.get_stats()
print(f"Prompt tokens: {stats['prompt_tokens']}")
print(f"Completion tokens: {stats['completion_tokens']}")
print(f"Route tokens: {stats['route_tokens']}")
print(f"Total: {stats['total_tokens']}")
```

## Response Structure

### Single Query Response

```python
{
    'query': 'original query string',
    'response': 'final answer',
    'model_name': 'model id used',
    'total_tokens': {
        'prompt_tokens': 234,
        'completion_tokens': 456,
        'route_tokens': 0,  # If no routing pool used
    },
    'prompt_tokens': 234,
    'completion_tokens': 456,
    'route_tokens': 0,
    'reasoning_trace': [
        {
            'iteration': 0,
            'generation': 'generated text',
            'search_query': 'optional search query',
            'route_info': {...},  # Routing pool response
            'tokens_used': {...}
        },
        ...
    ],
    'iterations': 2
}
```

## Example Reasoning Trace

```
[Iteration 0] 
Generation: "To answer this question, I need to understand transformers.
<search>What are transformers in deep learning?</search>"

[Iteration 1]
Generation: "Based on the retrieved information, transformers use:
1. Self-attention mechanisms
2. Multi-head attention
3. Feed-forward networks

<answer>Transformers are neural architectures using self-attention to process 
sequential data, allowing parallel processing of input tokens.</answer>"

Final Answer: "Transformers are neural architectures using self-attention..."
```

## Supported Models

Router-R1 supports any vLLM-compatible chat model:

### Qwen Models (Recommended)
- `Qwen/Qwen2.5-7B-Instruct`
- `Qwen/Qwen2.5-14B-Instruct`
- `Qwen/Qwen2-72B-Instruct`

### Llama Models
- `meta-llama/Llama-3.1-8B-Instruct`
- `meta-llama/Llama-3-70B-Instruct`

### Other Models
- Any model compatible with vLLM and instruction templates

## Performance Tips

### Model Selection
- Use 7B models for balance of quality and speed
- Use 14B+ models only when highest quality needed
- Qwen models often excel at reasoning tasks

### API Configuration
- Ensure routing pool API has low latency (<500ms)
- Use environment variables for API keys (better security)
- Monitor routing pool costs separately

### Prompt Engineering
- Customize system prompts in `PromptPool` class if needed
- Experiment with temperature (default: 1.0 for reasoning)
- Adjust max_iterations based on query complexity

### Resource Management
- vLLM auto-detects `tensor_parallel_size` based on visible GPUs
- Override with `LLMROUTER_TENSOR_PARALLEL_SIZE` if needed
- Monitor VRAM usage - adjust model size if needed

### Token Optimization
- Limit max_iterations for faster responses
- Use shorter max_tokens for faster generation
- Monitor total token cost (prompt + completion + route)

## Advantages

✅ **Agentic Reasoning**: Iteratively refines answers through reasoning loops
✅ **Dynamic Routing**: Can query routing pools multiple times per query
✅ **Explainable**: Full reasoning trace shows decision-making
✅ **Flexible**: Works with any vLLM-compatible model
✅ **Token Tracking**: Comprehensive cost tracking
✅ **No Training**: Zero-shot system, no training data needed

## Limitations

❌ **GPU Required**: vLLM requires CUDA, no CPU support
❌ **High Latency**: Multiple iterations increase response time
❌ **High Cost**: Multiple API calls are expensive
❌ **Complex**: Harder to debug than simple routers
❌ **Routing Pool Dependency**: Requires external routing pool API
❌ **Variable Behavior**: Non-deterministic reasoning may vary
❌ **Resource Intensive**: Requires 8GB+ VRAM

## When to Use Router-R1

### Good Use Cases
- Complex queries requiring multi-step reasoning
- Need explainable routing decisions with full traces
- Have access to quality routing pool API
- GPU resources available (8GB+ VRAM)
- Quality more important than latency/cost
- Research or advanced applications

### Consider Alternatives When
- Simple routing tasks → Use KNN/SVM/MLP Router
- No GPU available → Use API-based routers
- Cost/latency sensitive → Use simpler routers
- No routing pool API → Use other routing methods
- Need deterministic behavior → Use trained classifiers

## Configuration Examples

### Development/Testing

```yaml
hparam:
  model_id: "Qwen/Qwen2.5-7B-Instruct"
  api_base: "https://api.openai.com/v1"
  api_key: "${OPENROUTER_API_KEY}"
  max_iterations: 3          # Fewer iterations for testing
  temperature: 0.7           # Lower for consistency
  max_tokens: 256            # Shorter responses
  gpu_memory_utilization: 0.8
```

### Production/High Quality

```yaml
hparam:
  model_id: "Qwen/Qwen2.5-14B-Instruct"
  api_base: "https://api.openai.com/v1"
  api_key: "${OPENROUTER_API_KEY}"
  max_iterations: 5          # Full reasoning
  temperature: 1.0           # Better exploration
  max_tokens: 512            # Longer responses
  gpu_memory_utilization: 0.95
```

## API Reference

### RouterR1 Class

#### Methods

- `__init__(yaml_path, config)`: Initialize router
- `route_single(query, return_details, route_only)`: Route single query
- `route_batch(queries, task_name, return_details)`: Route multiple queries
- `save_result(result, output_path)`: Save result to JSON
- `get_stats()`: Get token usage statistics

### RouterR1Config Class

Configuration dataclass with fields:
- `model_id`: HuggingFace model ID
- `api_base`: Routing pool API base URL
- `api_key`: API authentication key
- `max_iterations`: Maximum reasoning iterations
- `temperature`: Generation temperature
- `max_tokens`: Maximum generation tokens
- `top_p`, `top_k`: Sampling parameters
- `tensor_parallel_size`: GPU parallelism
- `gpu_memory_utilization`: GPU memory usage ratio

## Troubleshooting

### GPU Out of Memory
- Reduce `max_tokens`
- Use smaller model (7B instead of 14B)
- Reduce `gpu_memory_utilization`
- Check CUDA_VISIBLE_DEVICES for correct GPU

### Routing Pool Timeouts
- Check API base URL is correct
- Verify API key has proper permissions
- Monitor routing pool service status
- Reduce query complexity

### Model Not Loading
- Verify HuggingFace model ID is correct
- Check internet connection for model download
- Ensure sufficient disk space for model weights
- Try with smaller model first

## Examples

See [examples/router_r1_examples.py](../examples/router_r1_examples.py) for:
- Basic single query routing
- Batch processing
- Detailed reasoning traces
- Route-only mode
- YAML configuration loading

## References

- Paper: [Router-R1: Teaching LLMs Multi-Round Routing and Aggregation via Reinforcement Learning](https://arxiv.org/abs/2506.09033)
- vLLM: https://docs.vllm.ai/
- OpenAI SDK: https://github.com/openai/openai-python

## License

Same as LLMRouter main project
