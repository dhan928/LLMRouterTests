"""Unified router manager for all supervised routers."""

from __future__ import annotations

from typing import Literal

from .base_router import BaseRouter, ModelCatalog
from .knn_router import KNNRouter
from .svm_router import SVMRouter
from .mlp_router import MLPRouter

RouterType = Literal["knn", "svm", "mlp"]


class RouterManager:
    """Unified interface for managing all supervised routers (KNN, SVM, MLP).
    
    Provides a modular way to select, configure, and use any of the three
    routing algorithms from a single manager instance.
    """

    def __init__(self):
        """Initialize the router manager."""
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
        """List all available router types.
        
        Returns:
            List of router type strings
        """
        return ["knn", "svm", "mlp"]

    def list_created(self) -> list[RouterType]:
        """List all currently created routers.
        
        Returns:
            List of created router type strings
        """
        return list(self.routers.keys())
