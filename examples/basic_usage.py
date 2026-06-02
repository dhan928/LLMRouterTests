from llmrouter import KNNRouter, MLPRouter, SVMRouter


MODEL_CATALOG = {
    "fast-small": {"provider": "demo", "cost": "low", "best_for": "simple tasks"},
    "balanced": {"provider": "demo", "cost": "medium", "best_for": "general tasks"},
    "strong-large": {"provider": "demo", "cost": "high", "best_for": "hard tasks"},
}

# Toy embeddings. In a real project these would come from an embedding model.
TRAIN_EMBEDDINGS = [
    [0.0, 0.1],
    [0.1, 0.0],
    [1.0, 1.1],
    [1.1, 1.0],
    [2.0, 2.1],
    [2.1, 2.0],
]

BEST_MODELS = [
    "fast-small",
    "fast-small",
    "balanced",
    "balanced",
    "strong-large",
    "strong-large",
]


def main() -> None:
    routers = {
        "knn": KNNRouter(MODEL_CATALOG, n_neighbors=1),
        "svm": SVMRouter(MODEL_CATALOG),
        "mlp": MLPRouter(MODEL_CATALOG, hidden_layer_sizes=(8,), max_iter=1000),
    }

    query = "Explain a difficult proof."
    query_embedding = [2.05, 2.0]

    for name, router in routers.items():
        router.fit(TRAIN_EMBEDDINGS, BEST_MODELS)
        decision = router.route(query, query_embedding)
        print(f"{name}: {decision.model_name} -> {dict(decision.model_info)}")


if __name__ == "__main__":
    main()
