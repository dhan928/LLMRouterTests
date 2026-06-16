# Router Modularization and Router-R1 Implementation

## Summary of Changes

This document summarizes the complete refactoring of the router system and implementation of the new Router-R1 (multi-round reasoning router).

## File Structure

### Before (Original)
```
llmrouter/
  routers.py          # All routers in single file
  __init__.py
  openrouter_client.py
  defaults.py
```

### After (Refactored)
```
llmrouter/
  base_router.py              # Base router class and common interfaces
  knn_router.py               # KNN router implementation
  svm_router.py               # SVM router implementation  
  mlp_router.py               # MLP router implementation
  router_manager.py           # Unified router manager
  router_r1.py                # Router-R1 multi-round reasoning implementation
  routers.py                  # Compatibility wrapper (imports all routers)
  ROUTER_R1_README.md         # Detailed Router-R1 documentation
  __init__.py                 # Updated exports
  openrouter_client.py        # (unchanged)
  defaults.py                 # (unchanged)

examples/
  router_r1_examples.py       # Router-R1 usage examples
```

## New Files Created

### 1. base_router.py
**Purpose**: Contains the base router class and shared data structures

**Classes**:
- `RouteDecision`: Decision made by router (dataclass)
- `RouterResponse`: Complete response from router (dataclass)
- `BaseRouter`: Abstract base class for scikit-learn routers
- Type alias: `ModelCatalog`

**Features**:
- Text vectorization with TF-IDF
- Fitting with embeddings or raw text
- Routing methods for embeddings and text
- Integration with OpenRouter API

### 2. knn_router.py
**Purpose**: K-Nearest Neighbors router implementation

**Classes**:
- `KNNRouter`: KNN-based routing

**Features**:
- Configurable n_neighbors parameter
- Multiple distance metrics (euclidean, manhattan, etc.)
- Inherits all BaseRouter functionality

### 3. svm_router.py
**Purpose**: Support Vector Machine router implementation

**Classes**:
- `SVMRouter`: SVM-based routing

**Features**:
- Configurable kernel types (linear, rbf, poly, sigmoid)
- Regularization parameter C
- Kernel coefficient gamma
- Inherits all BaseRouter functionality

### 4. mlp_router.py
**Purpose**: Multi-Layer Perceptron router implementation

**Classes**:
- `MLPRouter`: Neural network-based routing

**Features**:
- Configurable hidden layer sizes
- Maximum iterations setting
- Random state for reproducibility
- Inherits all BaseRouter functionality

### 5. router_manager.py
**Purpose**: Unified manager for all supervised routers

**Classes**:
- `RouterManager`: Factory and manager for all three routers
- Type alias: `RouterType` = Literal["knn", "svm", "mlp"]

**Key Methods**:
- `create_router()`: Create specific router type
- `get_router()`: Retrieve previously created router
- `create_all()`: Create all three routers at once
- `list_available()`: Show all router types
- `list_created()`: Show created routers

### 6. router_r1.py
**Purpose**: Advanced multi-round reasoning router implementation

**Classes**:
- `RouterR1Config`: Configuration dataclass
- `IterationTrace`: Single iteration trace dataclass
- `RouterR1Response`: Complete response dataclass
- `PromptPool`: System prompt templates by model
- `RouteService`: External routing pool API client
- `RouterR1`: Main Router-R1 implementation

**Key Features**:
- Multi-round iterative reasoning (up to 5 iterations)
- Special XML tags for control: `<search>`, `<answer>`, `<information>`
- vLLM integration for local inference
- OpenAI SDK for routing pool API calls
- Automatic model family detection and prompt selection
- Token tracking (prompt, completion, route tokens)
- Batch processing support
- Detailed reasoning trace output
- Route-only mode for testing

**Key Methods**:
- `route_single()`: Route single query
- `route_batch()`: Route multiple queries
- `get_stats()`: Get token usage statistics
- `save_result()`: Save results to JSON
- `_generate()`: Internal vLLM generation
- `_init_vllm()`: Initialize vLLM instance

**Supported Models**:
- Qwen: Qwen2.5-7B/14B/72B
- Llama: Llama-3.1-8B/70B
- Any vLLM-compatible model

### 7. Updated routers.py
**Changes**:
- Now imports from individual router modules
- Re-exports all classes for backward compatibility
- Acts as compatibility wrapper
- No breaking changes to existing code

**Key Changes**:
```python
# Before: All implementations in one file
# After: Imports from separate modules
from .base_router import BaseRouter, RouteDecision, RouterResponse, ModelCatalog
from .knn_router import KNNRouter
from .svm_router import SVMRouter
from .mlp_router import MLPRouter
from .router_manager import RouterManager, RouterType
```

### 8. Updated __init__.py
**Changes**:
- Added Router-R1 imports
- Expanded __all__ with new classes
- Organized exports by category

**New Exports**:
- `RouterR1`
- `RouterR1Config`
- `RouterR1Response`
- `PromptPool`
- `RouteService`

### 9. examples/router_r1_examples.py
**Purpose**: Demonstrates Router-R1 usage patterns

**Examples**:
1. Basic single query routing
2. Batch routing with multiple queries
3. Detailed reasoning traces
4. Route-only mode (no LLM generation)
5. Loading from YAML configuration

### 10. llmrouter/ROUTER_R1_README.md
**Purpose**: Comprehensive Router-R1 documentation

**Sections**:
- Overview and key features
- Architecture diagram
- Installation instructions
- Configuration (YAML and environment variables)
- Python API examples
- CLI usage
- How it works (mechanism details)
- Response structure
- Supported models
- Performance tips
- Advantages and limitations
- When to use Router-R1
- Troubleshooting guide
- Full API reference

## Backward Compatibility

✅ **No breaking changes**: All existing code using routers continues to work
✅ **Old imports work**: `from llmrouter import KNNRouter, SVMRouter, MLPRouter`
✅ **New structure transparent**: Users can import from submodules or main package

## Benefits of Refactoring

### Code Organization
- Each router in its own file
- Clear separation of concerns
- Easier to maintain and extend
- Better discoverability

### Modularity
- Can import specific routers
- Router manager for unified interface
- Easy to add new router types

### Testability
- Each router can be tested independently
- Base router functionality isolated
- Easier to mock dependencies

### Documentation
- Dedicated Router-R1 README
- Comprehensive examples
- Clear API reference
- Configuration guidance

## Usage Examples

### Old Way (Still Works)
```python
from llmrouter import KNNRouter, SVMRouter, MLPRouter, RouterManager
```

### New Way (Preferred)
```python
from llmrouter.knn_router import KNNRouter
from llmrouter.svm_router import SVMRouter
from llmrouter.mlp_router import MLPRouter
from llmrouter.router_manager import RouterManager
```

### Router-R1 Usage
```python
from llmrouter import RouterR1, RouterR1Config

config = RouterR1Config(
    model_id="Qwen/Qwen2.5-7B-Instruct",
    api_base="https://api.openai.com/v1",
    api_key="your-key"
)
router = RouterR1(config=config)
result = router.route_single("What is machine learning?")
```

## Dependencies

### New Dependencies for Router-R1
- `vllm==0.6.3` - Local model inference
- `torch==2.4.0` - Deep learning framework
- `openai>=1.0` - API client for routing pool
- `pyyaml` - YAML configuration (usually already installed)

### Existing Dependencies (Unchanged)
- `scikit-learn` - For KNN, SVM, MLP routers
- `numpy` - For numerical operations
- All other existing dependencies

## Configuration

### Router-R1 YAML Config
```yaml
hparam:
  model_id: "Qwen/Qwen2.5-7B-Instruct"
  api_base: "https://api.openai.com/v1"
  api_key: "${OPENROUTER_API_KEY}"
  max_iterations: 5
  temperature: 1.0
  max_tokens: 512
```

### Environment Variables
```bash
export OPENROUTER_API_KEY="your-key"
export API_BASE="https://api.openai.com/v1"
```

## Migration Guide

### For Existing Code
No changes needed - all existing imports continue to work:

```python
# This still works exactly as before
from llmrouter import KNNRouter, SVMRouter, MLPRouter, RouterManager
```

### To Use Router-R1
```python
# Add to your code
from llmrouter import RouterR1

# Use as shown in examples
router = RouterR1(yaml_path="configs/router_r1.yaml")
result = router.route_single("Your query here")
```

### To Use New Module Structure
```python
# Import from specific modules (optional)
from llmrouter.base_router import BaseRouter
from llmrouter.knn_router import KNNRouter
from llmrouter.router_manager import RouterManager
```

## Testing

The existing test suite (`tests/test_simple_routers.py`) continues to work without modifications due to backward compatibility.

To test Router-R1:
1. Ensure vLLM and dependencies installed
2. Have GPU available with CUDA
3. Set up API credentials
4. See `examples/router_r1_examples.py` for test cases

## Future Enhancements

Possible future improvements:
- Additional router types (gradient boosting, ensemble methods)
- More language model integrations
- Custom prompt templates per domain
- Caching for routing decisions
- Distributed routing across multiple GPUs
- More detailed performance metrics
- Integration with other LLM frameworks

## Summary

The refactoring successfully:
✅ Modularized the existing router implementations
✅ Maintained 100% backward compatibility
✅ Implemented full Router-R1 functionality
✅ Provided comprehensive documentation
✅ Created working examples
✅ Improved code organization and maintainability

All new functionality is available immediately while existing code continues to work unchanged.
