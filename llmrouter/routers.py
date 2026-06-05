from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer


ModelCatalog = Mapping[str, Mapping[str, object]]


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
