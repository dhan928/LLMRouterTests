"""Routers module - imports from individual router implementations.

This module provides a unified interface to all routing implementations.
Individual routers are defined in separate modules for better modularity.
"""

from __future__ import annotations

# Import from base router module
from .base_router import (
    BaseRouter,
    RouteDecision,
    RouterResponse,
    ModelCatalog,
)

# Import individual routers
from .knn_router import KNNRouter
from .svm_router import SVMRouter
from .mlp_router import MLPRouter

# Import router manager
from .router_manager import RouterManager, RouterType

# Re-export for backward compatibility
__all__ = [
    "BaseRouter",
    "RouteDecision",
    "RouterResponse",
    "ModelCatalog",
    "KNNRouter",
    "SVMRouter",
    "MLPRouter",
    "RouterManager",
    "RouterType",
]
