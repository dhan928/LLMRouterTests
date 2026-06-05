# Simple LLM Router

This repo includes only three routers:

- `KNNRouter`: K-Nearest Neighbors based routing
- `SVMRouter`: Support Vector Machine based routing
- `MLPRouter`: Multi-Layer Perceptron based routing

Each router trains on example query text labeled as either `weak` or `strong`.
When you pass a real query, the router embeds the text with TF-IDF, predicts
`weak` or `strong`, and maps that choice to an OpenRouter model.

## Setup

```powershell
python -m pip install -e ".[dev]"
```

Put your OpenRouter API key in `.env`:

```text
OPENROUTER_API_KEY=your_key_here
OPENROUTER_WEAK_MODEL=openai/gpt-4o-mini
OPENROUTER_STRONG_MODEL=openai/gpt-4.1
OPENROUTER_SITE_URL=http://localhost
OPENROUTER_APP_NAME=LLMRouterTests
```

The default weak/strong models can be changed in `.env`.

## Route Without Calling The API

```powershell
python examples/basic_usage.py "Explain this Python traceback and suggest a fix."
```

This prints the decision from all three routers:

```text
knnrouter: strong -> openai/gpt-4.1
svmrouter: strong -> openai/gpt-4.1
mlprouter: strong -> openai/gpt-4.1
```

## Route And Call OpenRouter

After adding your key to `.env`:

```powershell
python examples/basic_usage.py "Explain this Python traceback and suggest a fix." --call-api
```

Each router will:

1. Train on the bundled weak/strong examples.
2. Route your real query to `weak` or `strong`.
3. Call the selected OpenRouter model.
4. Print the model response.

## Run Tests

```powershell
python -m pytest
```

The tests do not call OpenRouter. They use a fake client to verify the selected
model would be sent to the API.

## How The Routers Decide

Training examples labeled `weak` are simple tasks like greetings, summaries,
and short factual questions.

Training examples labeled `strong` are harder tasks like debugging, architecture,
math reasoning, legal comparison, and complex codebase planning.

The routers learn that split differently:

- KNN compares your query to nearby training examples.
- SVM learns a boundary between weak-style and strong-style queries.
- MLP learns a small neural classifier over the text features.

OpenRouter API reference used for the client:

- [Chat completions](https://openrouter.ai/docs/api-reference/chat-completion)
- [Authentication](https://openrouter.ai/docs/api-keys)
