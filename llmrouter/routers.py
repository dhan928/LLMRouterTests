from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping, Sequence

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer


ModelCatalog = Mapping[str, Mapping[str, object]]
RouterType = Literal["knn", "svm", "mlp"]


@dataclass(frozen=True)
class RouteDecision:
    query: str
    model_name: str
    model_info: Mapping[str, object]


@dataclass(frozen=True)
class RouterResponse:
    decision: RouteDecision
    response: str
    raw: Mapping[str, object]


class BaseRouter:
    """Small shared wrapper around a scikit-learn classifier."""

    def __init__(self, classifier, model_catalog: ModelCatalog):
        self.classifier = classifier
        self.model_catalog = model_catalog
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True)
        self._is_fit = False

    def fit(
        self,
        query_embeddings: Sequence[Sequence[float]],
        best_model_names: Sequence[str],
    ) -> "BaseRouter":
        self.classifier.fit(np.asarray(query_embeddings), list(best_model_names))
        self._is_fit = True
        return self

    def fit_texts(
        self,
        training_queries: Sequence[str],
        best_model_names: Sequence[str],
    ) -> "BaseRouter":
        query_embeddings = self.vectorizer.fit_transform(training_queries)
        self.classifier.fit(query_embeddings, list(best_model_names))
        self._is_fit = True
        return self

    def route(self, query: str, query_embedding: Sequence[float]) -> RouteDecision:
        if not self._is_fit:
            raise RuntimeError("Router must be fit before calling route().")

        embedding = np.asarray(query_embedding).reshape(1, -1)
        model_name = str(self.classifier.predict(embedding)[0])
        return RouteDecision(
            query=query,
            model_name=model_name,
            model_info=self.model_catalog.get(model_name, {}),
        )

    def route_text(self, query: str) -> RouteDecision:
        if not self._is_fit:
            raise RuntimeError("Router must be fit before calling route_text().")

        query_embedding = self.vectorizer.transform([query])
        model_name = str(self.classifier.predict(query_embedding)[0])
        return RouteDecision(
            query=query,
            model_name=model_name,
            model_info=self.model_catalog.get(model_name, {}),
        )

    def answer(
        self,
        query: str,
        openrouter_client,
        *,
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> RouterResponse:
        decision = self.route_text(query)
        openrouter_model = decision.model_info.get("openrouter_model")
        if not openrouter_model:
            raise ValueError(f"No OpenRouter model configured for {decision.model_name}.")

        raw = openrouter_client.chat(
            model=str(openrouter_model),
            query=query,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        response = (
            raw.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        return RouterResponse(decision=decision, response=response, raw=raw)


class KNNRouter(BaseRouter):
    """K-Nearest Neighbors based routing."""

    def __init__(
        self,
        model_catalog: ModelCatalog,
        n_neighbors: int = 3,
        metric: str = "euclidean",
    ):
        super().__init__(
            KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric),
            model_catalog,
        )


class SVMRouter(BaseRouter):
    """Support Vector Machine based routing."""

    def __init__(
        self,
        model_catalog: ModelCatalog,
        kernel: str = "rbf",
        C: float = 1.0,
        gamma: str = "scale",
    ):
        super().__init__(
            SVC(kernel=kernel, C=C, gamma=gamma),
            model_catalog,
        )


class MLPRouter(BaseRouter):
    """Multi-Layer Perceptron based routing."""

    def __init__(
        self,
        model_catalog: ModelCatalog,
        hidden_layer_sizes: tuple[int, ...] = (16,),
        max_iter: int = 500,
        random_state: int = 7,
    ):
        super().__init__(
            MLPClassifier(
                hidden_layer_sizes=hidden_layer_sizes,
                max_iter=max_iter,
                random_state=random_state,
            ),
            model_catalog,
        )


class RouterManager:
    """Unified interface for managing all three routers (KNN, SVM, MLP).
    
    Provides a modular way to select, configure, and use any of the three
    routing algorithms from a single source.
    """

    def __init__(self):
        self.routers: dict[str, BaseRouter] = {}

    def create_router(
        self,
        router_type: RouterType,
        model_catalog: ModelCatalog,
        **kwargs,
    ) -> BaseRouter:
        """Create and register a router.
        
        Args:
            router_type: Type of router ('knn', 'svm', or 'mlp')
            model_catalog: Catalog mapping model names to configurations
            **kwargs: Router-specific parameters
        
        Returns:
            The created router instance
            
        Raises:
            ValueError: If router_type is not valid
        """
        if router_type == "knn":
            router = KNNRouter(model_catalog, **kwargs)
        elif router_type == "svm":
            router = SVMRouter(model_catalog, **kwargs)
        elif router_type == "mlp":
            router = MLPRouter(model_catalog, **kwargs)
        else:
            raise ValueError(f"Unknown router type: {router_type}. Use 'knn', 'svm', or 'mlp'")
        
        self.routers[router_type] = router
        return router

    def get_router(self, router_type: RouterType) -> BaseRouter:
        """Get a previously created router.
        
        Args:
            router_type: Type of router to retrieve
            
        Returns:
            The router instance
            
        Raises:
            KeyError: If router hasn't been created yet
        """
        if router_type not in self.routers:
            raise KeyError(f"Router '{router_type}' not created yet. Call create_router() first.")
        return self.routers[router_type]

    def create_all(
        self,
        model_catalog: ModelCatalog,
        knn_kwargs: dict | None = None,
        svm_kwargs: dict | None = None,
        mlp_kwargs: dict | None = None,
    ) -> dict[RouterType, BaseRouter]:
        """Create all three routers at once with custom parameters.
        
        Args:
            model_catalog: Catalog mapping model names to configurations
            knn_kwargs: Parameters for KNN router
            svm_kwargs: Parameters for SVM router
            mlp_kwargs: Parameters for MLP router
            
        Returns:
            Dictionary of all three routers
        """
        self.create_router("knn", model_catalog, **(knn_kwargs or {}))
        self.create_router("svm", model_catalog, **(svm_kwargs or {}))
        self.create_router("mlp", model_catalog, **(mlp_kwargs or {}))
        return self.routers

    def list_available(self) -> list[RouterType]:
        """List all available router types."""
        return ["knn", "svm", "mlp"]

    def list_created(self) -> list[RouterType]:
        """List all created routers."""
        return list(self.routers.keys())
