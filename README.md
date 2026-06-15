# LLM Router

An intelligent system that automatically routes text queries to either a weak (fast/cheap) or strong (powerful/expensive) language model using machine learning. Includes three routing algorithms: **KNN**, **SVM**, and **MLP**.

## Features

- **Text-Based Query Routing** - Automatically classifies queries by complexity
- **Three ML Algorithms** - KNN, SVM, and MLP routers for different performance characteristics
- **OpenRouter Integration** - Seamless integration with OpenRouter API
- **Pre-Configured Models** - Weak model: `openai/gpt-4o-mini` | Strong model: `openai/gpt-4.1`
- **Zero Setup** - Pre-built training data included, ready to use immediately

## The Three Routers

### KNN Router (K-Nearest Neighbors)
Fast and simple. Makes routing decisions based on similarity to training examples.
```python
router = KNNRouter(catalog, n_neighbors=3)
```
**Best for:** Quick decisions on balanced datasets

### SVM Router (Support Vector Machine)
Strong generalization with clear decision boundaries. Handles edge cases well.
```python
router = SVMRouter(catalog, kernel='rbf', C=1.0)
```
**Best for:** Robust routing across diverse queries

### MLP Router (Multi-Layer Perceptron)
Neural network that learns complex patterns. Most flexible but requires careful tuning.
```python
router = MLPRouter(catalog, hidden_layer_sizes=(16,), max_iter=500)
```
**Best for:** Complex non-linear relationships

## Installation

```bash
pip install -e ".[dev]"
```

## Quick Start

### 1. Set Up API Key (Optional)

Create a `.env` file in the project root to enable live OpenRouter responses:

```env
OPENROUTER_API_KEY=sk-or-v1-xxxxx
OPENROUTER_WEAK_MODEL=openai/gpt-4o-mini
OPENROUTER_STRONG_MODEL=openai/gpt-4.1
```

### 2. See It In Action

**First, view a quick demo:**
```bash
python examples/demo.py
```

**Then launch the interactive chatbot:**
```bash
python examples/chatbot.py
```

In the chatbot, type questions and watch how they get routed:
- Simple questions → **weak model** (fast, cheap)
- Complex questions → **strong model** (powerful, expensive)

Type `help` in the chatbot for all commands.

## Usage Examples

### Minimal Example (5 lines)

```python
from llmrouter import KNNRouter, default_model_catalog, TRAINING_QUERIES, TRAINING_LABELS

router = KNNRouter(default_model_catalog())
router.fit_texts(TRAINING_QUERIES, TRAINING_LABELS)
decision = router.route_text("Your query")
print(decision.model_name)  # "weak" or "strong"
```

### Modular Router Manager (Recommended)

Use the `RouterManager` to select and switch between routers:

```python
from llmrouter import RouterManager, default_model_catalog, TRAINING_QUERIES, TRAINING_LABELS

manager = RouterManager()
catalog = default_model_catalog()

# Create routers dynamically
knn = manager.create_router("knn", catalog, n_neighbors=3)
svm = manager.create_router("svm", catalog, kernel="rbf")
mlp = manager.create_router("mlp", catalog, hidden_layer_sizes=(12,))

# Train them
for router in [knn, svm, mlp]:
    router.fit_texts(TRAINING_QUERIES, TRAINING_LABELS)

# Use any router
decision = manager.get_router("knn").route_text("Your query")
print(decision.model_name)
```

### Create All Routers at Once

```python
from llmrouter import RouterManager, default_model_catalog, TRAINING_QUERIES, TRAINING_LABELS

manager = RouterManager()
routers = manager.create_all(default_model_catalog())

# Train all routers
for router in routers.values():
    router.fit_texts(TRAINING_QUERIES, TRAINING_LABELS)

# Compare decisions
query = "Design a database"
for router_type in manager.list_created():
    router = manager.get_router(router_type)
    decision = router.route_text(query)
    print(f"{router_type}: {decision.model_name}")
```

## How It Works

1. **Train** - Routers learn from 12 pre-built training queries (6 simple/weak, 6 complex/strong)
2. **Feature Extract** - TF-IDF vectorizes incoming text queries
3. **Classify** - ML model predicts "weak" or "strong" category
4. **Route** - Query is sent to the selected model via OpenRouter API
5. **Respond** - Response is returned to the user

## Testing

All three routers are fully tested:

```bash
pytest tests/ -v
```

**Test Results:**
```
✓ KNNRouter can fit and route text queries
✓ SVMRouter can fit and route text queries  
✓ MLPRouter can fit and route text queries
✓ Routers correctly call selected OpenRouter models
✓ Routers validate training before routing

5 passed in 5.47s
```

## API Reference

### Router Manager (Unified Interface)

**`RouterManager`** - Central hub for selecting and managing all three routers.

**Methods:**
- `create_router(router_type, catalog, **kwargs)` - Create a specific router
  - `router_type`: `"knn"`, `"svm"`, or `"mlp"`
  - Returns: Router instance
- `get_router(router_type)` - Retrieve a created router
- `create_all(catalog, ...)` - Create all three routers at once
- `list_available()` - Show all available router types
- `list_created()` - Show all created routers

### Router Classes

**`KNNRouter`** - K-Nearest Neighbors
```python
KNNRouter(catalog, n_neighbors=3, metric="euclidean")
```

**`SVMRouter`** - Support Vector Machine
```python
SVMRouter(catalog, kernel="rbf", C=1.0, gamma="scale")
```

**`MLPRouter`** - Multi-Layer Perceptron
```python
MLPRouter(catalog, hidden_layer_sizes=(16,), max_iter=500, random_state=7)
```

### Router Methods (All Routers)

**`fit_texts(training_queries, best_model_names)`**
Train the router on text queries.

**`route_text(query)`**
Returns `RouteDecision` containing the selected model name and model info.

**`answer(query, openrouter_client, temperature=0.2, max_tokens=512)`**
Routes the query and calls OpenRouter API. Returns `RouterResponse` with the model response.

### Data Classes

**`RouteDecision`**
```python
decision.query              # Original query
decision.model_name         # "weak" or "strong"
decision.model_info         # Model configuration
```

**`RouterResponse`**
```python
response.decision           # RouteDecision object
response.response           # LLM response text
response.raw               # Full OpenRouter API response
```

## Examples

The `examples/` directory provides different ways to test the router:

- **`chatbot.py`** ⭐ - **Interactive terminal chatbot** - Type questions and see real-time routing
- **`modular_routers.py`** - Modular system example using `RouterManager`
- **`demo.py`** - Quick demo with sample queries showing routing behavior
- **`basic_usage.py`** - Route a single query from the command line
- **`detailed_demo.py`** - Comprehensive demonstration of all features

**Recommended way to get started:**
```bash
# 1. See modular router system in action
python examples/modular_routers.py

# 2. Try the interactive chatbot
python examples/chatbot.py
```

## Configuration

Set these environment variables in `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENROUTER_API_KEY` | Required | Your OpenRouter API key |
| `OPENROUTER_WEAK_MODEL` | `openai/gpt-4o-mini` | Model for simple queries |
| `OPENROUTER_STRONG_MODEL` | `openai/gpt-4.1` | Model for complex queries |
| `OPENROUTER_SITE_URL` | Optional | Your application URL |
| `OPENROUTER_APP_NAME` | `LLMRouterTests` | Your application name |

## Project Structure

```
llmrouter/
├── routers.py              # KNN, SVM, MLP implementations
├── openrouter_client.py    # OpenRouter API client
├── defaults.py             # Training data & model catalog
└── __init__.py             # Public API

examples/
├── basic_usage.py          # Simple example
└── detailed_demo.py        # Comprehensive demo

tests/
└── test_simple_routers.py  # Full test coverage
```

## License

See LICENSE file
