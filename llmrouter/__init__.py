from .defaults import TRAINING_LABELS, TRAINING_QUERIES, default_model_catalog
from .openrouter_client import OpenRouterClient, load_env
from .routers import BaseRouter, KNNRouter, MLPRouter, SVMRouter, RouterManager, RouterType, RouteDecision, RouterResponse
from .router_r1 import RouterR1, RouterR1Config, RouterR1Response, PromptPool, RouteService

__all__ = [
    # Base classes
    "BaseRouter",
    "RouteDecision",
    "RouterResponse",
    
    # Supervised routers
    "KNNRouter",
    "SVMRouter",
    "MLPRouter",
    "RouterManager",
    "RouterType",
    
    # Router-R1 (Multi-round reasoning router)
    "RouterR1",
    "RouterR1Config",
    "RouterR1Response",
    "PromptPool",
    "RouteService",
    
    # Client and utilities
    "OpenRouterClient",
    "load_env",
    "default_model_catalog",
    "TRAINING_QUERIES",
    "TRAINING_LABELS",
]

