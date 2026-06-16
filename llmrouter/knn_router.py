"""K-Nearest Neighbors based router."""

from __future__ import annotations

from sklearn.neighbors import KNeighborsClassifier

from .base_router import BaseRouter, ModelCatalog


class KNNRouter(BaseRouter):
    """K-Nearest Neighbors based routing.
    
    Uses KNN classification to determine which model should handle a query
    based on similar training examples.
    """

    def __init__(
        self,
        model_catalog: ModelCatalog,
        n_neighbors: int = 3,
        metric: str = "euclidean",
    ):
        """Initialize the KNN router.
        
        Args:
            model_catalog: Mapping of model names to their configurations
            n_neighbors: Number of neighbors to consider
            metric: Distance metric to use (e.g., 'euclidean', 'manhattan')
        """
        super().__init__(
            KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric),
            model_catalog,
        )
