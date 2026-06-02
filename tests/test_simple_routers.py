import pytest

from llmrouter import KNNRouter, MLPRouter, SVMRouter


MODEL_CATALOG = {
    "fast-small": {"cost": "low"},
    "balanced": {"cost": "medium"},
    "strong-large": {"cost": "high"},
}

TRAIN_EMBEDDINGS = [
    [0.0, 0.0],
    [0.2, 0.1],
    [1.0, 1.0],
    [1.2, 1.1],
    [2.0, 2.0],
    [2.2, 2.1],
]

BEST_MODELS = [
    "fast-small",
    "fast-small",
    "balanced",
    "balanced",
    "strong-large",
    "strong-large",
]


@pytest.mark.parametrize(
    "router",
    [
        KNNRouter(MODEL_CATALOG, n_neighbors=1),
        SVMRouter(MODEL_CATALOG),
        MLPRouter(MODEL_CATALOG, hidden_layer_sizes=(8,), max_iter=1000),
    ],
)
def test_router_can_fit_and_route(router):
    router.fit(TRAIN_EMBEDDINGS, BEST_MODELS)

    decision = router.route(
        query="Solve a complex reasoning task.",
        query_embedding=[2.1, 2.0],
    )

    assert decision.model_name in MODEL_CATALOG
    assert decision.model_info == MODEL_CATALOG[decision.model_name]


def test_router_requires_fit_before_route():
    router = KNNRouter(MODEL_CATALOG, n_neighbors=1)

    with pytest.raises(RuntimeError, match="must be fit"):
        router.route("hello", [0.0, 0.0])
