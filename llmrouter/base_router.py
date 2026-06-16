"""Base router class and common data structures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


ModelCatalog = Mapping[str, Mapping[str, object]]


@dataclass(frozen=True)
class RouteDecision:
    """Decision made by the router about which model to use."""
    query: str
    model_name: str
    model_info: Mapping[str, object]


@dataclass(frozen=True)
class RouterResponse:
    """Complete response from a router including the routing decision and answer."""
    decision: RouteDecision
    response: str
    raw: Mapping[str, object]


class BaseRouter:
    """Base class for all scikit-learn based routers (KNN, SVM, MLP).
    
    Provides common functionality for fitting and routing using text vectorization
    and classification algorithms.
    """

    def __init__(self, classifier, model_catalog: ModelCatalog):
        """Initialize the router.
        
        Args:
            classifier: A scikit-learn classifier instance
            model_catalog: Mapping of model names to their configurations
        """
        self.classifier = classifier
        self.model_catalog = model_catalog
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True)
        self._is_fit = False

    def fit(
        self,
        query_embeddings: Sequence[Sequence[float]],
        best_model_names: Sequence[str],
    ) -> BaseRouter:
        """Fit the router using pre-computed embeddings.
        
        Args:
            query_embeddings: Sequence of embedding vectors
            best_model_names: Sequence of model names (labels)
            
        Returns:
            Self for method chaining
        """
        self.classifier.fit(np.asarray(query_embeddings), list(best_model_names))
        self._is_fit = True
        return self

    def fit_texts(
        self,
        training_queries: Sequence[str],
        best_model_names: Sequence[str],
    ) -> BaseRouter:
        """Fit the router using raw text queries.
        
        Automatically vectorizes the text using TF-IDF.
        
        Args:
            training_queries: Sequence of query texts
            best_model_names: Sequence of model names (labels)
            
        Returns:
            Self for method chaining
        """
        query_embeddings = self.vectorizer.fit_transform(training_queries)
        self.classifier.fit(query_embeddings, list(best_model_names))
        self._is_fit = True
        return self

    def route(self, query: str, query_embedding: Sequence[float]) -> RouteDecision:
        """Route using a pre-computed embedding.
        
        Args:
            query: The query text
            query_embedding: The embedding vector for the query
            
        Returns:
            RouteDecision with the selected model
            
        Raises:
            RuntimeError: If router hasn't been fit yet
        """
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
        """Route using raw text (auto-vectorized).
        
        Args:
            query: The query text
            
        Returns:
            RouteDecision with the selected model
            
        Raises:
            RuntimeError: If router hasn't been fit yet
        """
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
        """Route a query and get an answer from the selected model.
        
        Args:
            query: The query text
            openrouter_client: Client for calling the OpenRouter API
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            
        Returns:
            RouterResponse with decision, response, and raw API output
        """
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
