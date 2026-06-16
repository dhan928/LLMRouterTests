"""Support Vector Machine based router."""

from __future__ import annotations

from sklearn.svm import SVC

from .base_router import BaseRouter, ModelCatalog


class SVMRouter(BaseRouter):
    """Support Vector Machine based routing.
    
    Uses SVM classification to determine which model should handle a query
    by finding optimal decision boundaries in the feature space.
    """

    def __init__(
        self,
        model_catalog: ModelCatalog,
        kernel: str = "rbf",
        C: float = 1.0,
        gamma: str = "scale",
    ):
        """Initialize the SVM router.
        
        Args:
            model_catalog: Mapping of model names to their configurations
            kernel: Kernel type to use ('linear', 'rbf', 'poly', 'sigmoid')
            C: Regularization parameter
            gamma: Kernel coefficient ('scale' or 'auto')
        """
        super().__init__(
            SVC(kernel=kernel, C=C, gamma=gamma),
            model_catalog,
        )
