"""Multi-Layer Perceptron based router."""

from __future__ import annotations

from sklearn.neural_network import MLPClassifier

from .base_router import BaseRouter, ModelCatalog


class MLPRouter(BaseRouter):
    """Multi-Layer Perceptron based routing.
    
    Uses a neural network classifier to determine which model should handle
    a query based on learned non-linear decision boundaries.
    """

    def __init__(
        self,
        model_catalog: ModelCatalog,
        hidden_layer_sizes: tuple[int, ...] = (16,),
        max_iter: int = 500,
        random_state: int = 7,
    ):
        """Initialize the MLP router.
        
        Args:
            model_catalog: Mapping of model names to their configurations
            hidden_layer_sizes: Sizes of hidden layers
            max_iter: Maximum number of iterations
            random_state: Random seed for reproducibility
        """
        super().__init__(
            MLPClassifier(
                hidden_layer_sizes=hidden_layer_sizes,
                max_iter=max_iter,
                random_state=random_state,
            ),
            model_catalog,
        )
