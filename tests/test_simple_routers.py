import pytest

from llmrouter import KNNRouter, MLPRouter, SVMRouter


MODEL_CATALOG = {
    "weak": {"openrouter_model": "openai/gpt-4o-mini"},
    "strong": {"openrouter_model": "openai/gpt-4.1"},
}

TRAINING_QUERIES = [
    "Say hello.",
    "Summarize a short note.",
    "Give me a simple recipe.",
    "Explain quantum physics with equations.",
    "Debug a difficult production outage.",
    "Design a distributed database architecture.",
]

TRAINING_LABELS = [
    "weak",
    "weak",
    "weak",
    "strong",
    "strong",
    "strong",
]


class FakeOpenRouterClient:
    def __init__(self):
        self.calls = []

    def chat(self, *, model, query, temperature, max_tokens):
        self.calls.append(
            {
                "model": model,
                "query": query,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        return {"choices": [{"message": {"content": f"answered by {model}"}}]}


@pytest.mark.parametrize(
    "router",
    [
        KNNRouter(MODEL_CATALOG, n_neighbors=1),
        SVMRouter(MODEL_CATALOG),
        MLPRouter(MODEL_CATALOG, hidden_layer_sizes=(8,), max_iter=1000),
    ],
)
def test_router_can_fit_real_query_text(router):
    router.fit_texts(TRAINING_QUERIES, TRAINING_LABELS)

    decision = router.route_text("Please debug this complex Python service failure.")

    assert decision.model_name in {"weak", "strong"}
    assert decision.model_info == MODEL_CATALOG[decision.model_name]


def test_router_answer_calls_selected_openrouter_model():
    router = KNNRouter(MODEL_CATALOG, n_neighbors=1)
    router.fit_texts(TRAINING_QUERIES, TRAINING_LABELS)
    client = FakeOpenRouterClient()

    result = router.answer("Debug a difficult production outage.", client)

    assert result.decision.model_name == "strong"
    assert client.calls[0]["model"] == "openai/gpt-4.1"
    assert result.response == "answered by openai/gpt-4.1"


def test_router_requires_fit_before_route_text():
    router = KNNRouter(MODEL_CATALOG, n_neighbors=1)

    with pytest.raises(RuntimeError, match="must be fit"):
        router.route_text("hello")
