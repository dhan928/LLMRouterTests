from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC


ModelCatalog = Mapping[str, Mapping[str, object]]


@dataclass(frozen=True)
class RouteDecision:
    query: str
    model_name: str
    model_info: Mapping[str, object]


class BaseRouter:
    """Small shared wrapper around a scikit-learn classifier."""

    def __init__(self, classifier, model_catalog: ModelCatalog):
        self.classifier = classifier
        self.model_catalog = model_catalog
        self._is_fit = False

    def fit(
        self,
        query_embeddings: Sequence[Sequence[float]],
        best_model_names: Sequence[str],
    ) -> "BaseRouter":
        self.classifier.fit(np.asarray(query_embeddings), list(best_model_names))
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
