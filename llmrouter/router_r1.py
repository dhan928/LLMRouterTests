"""Router-R1: Advanced multi-round reasoning router with dynamic routing.

This router implements the Router-R1 approach as described in:
"Router-R1: Teaching LLMs Multi-Round Routing and Aggregation via Reinforcement Learning"
(Zhang, H., Feng, T., & You, J. (2025). arXiv:2506.09033)
"""

from __future__ import annotations

import os
import re
import yaml
import json
from typing import Any, Dict, List, Optional, Mapping
from dataclasses import dataclass, field
from datetime import datetime
import logging

try:
    from openai import OpenAI
except ImportError:
    raise ImportError("openai is required for Router-R1. Install with: pip install openai>=1.0")

_VLLM_AVAILABLE = False
try:
    from vllm import LLM, SamplingParams

    _VLLM_AVAILABLE = True
except ImportError:
    LLM = None  # type: ignore[misc, assignment]
    SamplingParams = None  # type: ignore[misc, assignment]


logger = logging.getLogger(__name__)


@dataclass
class RouterR1Config:
    """Configuration for Router-R1."""
    
    model_id: str
    api_base: Optional[str] = None
    api_key: Optional[str] = None
    backend: str = "auto"  # "auto", "vllm", or "api"
    route_model: str = "openai/gpt-4o-mini"
    route_models: Optional[List[str]] = None
    coordinator_mode: bool = False
    max_iterations: int = 5
    temperature: float = 1.0
    max_tokens: int = 512
    top_p: float = 1.0
    top_k: int = -1
    tensor_parallel_size: Optional[int] = None
    gpu_memory_utilization: float = 0.9
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> RouterR1Config:
        """Load configuration from YAML file.
        
        Args:
            yaml_path: Path to YAML configuration file
            
        Returns:
            RouterR1Config instance
        """
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        hparam = config_dict.get('hparam', {})
        return cls(**hparam)


@dataclass
class IterationTrace:
    """Trace of a single iteration in the reasoning loop."""
    
    iteration: int
    generation: str
    search_query: Optional[str] = None
    route_info: Optional[Dict[str, Any]] = None
    tokens_used: Dict[str, int] = field(default_factory=dict)


@dataclass
class RouterR1Response:
    """Response from Router-R1 routing."""
    
    query: str
    final_answer: str
    reasoning_trace: List[IterationTrace]
    model_used: str
    total_tokens: Dict[str, int]
    iteration_count: int
    success: bool = True
    error: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class RouteModelProfile:
    """Profile for a routed expert model, similar to weak/strong in the ML routers."""

    def __init__(self, model_id: str, name: str, specialty: str, system_prompt: str):
        self.model_id = model_id
        self.name = name
        self.specialty = specialty
        self.system_prompt = system_prompt


class RoutePromptPool:
    """Per-model prompts used when the coordinator routes a <search> to an expert."""

    PROFILES: Dict[str, RouteModelProfile] = {
        "anthropic/claude-3.5-haiku": RouteModelProfile(
            model_id="anthropic/claude-3.5-haiku",
            name="Claude Haiku",
            specialty="Nuanced analysis, comparisons, and trade-off evaluation",
            system_prompt="""You are Claude Haiku, an analysis expert in a multi-model routing pool.

Your role: provide careful, balanced analysis when the coordinator routes a search to you.

Strengths:
- Compare approaches, methods, or options with clear trade-offs
- Explain why something works, not just what it is
- Flag caveats, limitations, and edge cases

Respond in 2-4 concise paragraphs. Be precise. Do not mention that you are part of a routing system.""",
        ),
        "google/gemini-2.0-flash-001": RouteModelProfile(
            model_id="google/gemini-2.0-flash-001",
            name="Gemini Flash",
            specialty="Fast factual retrieval, definitions, and summaries",
            system_prompt="""You are Gemini Flash, a factual retrieval expert in a multi-model routing pool.

Your role: deliver quick, accurate facts when the coordinator routes a search to you.

Strengths:
- Clear definitions and terminology
- Short factual summaries
- Listing key concepts, steps, or components

Respond in bullet points or 1-2 short paragraphs. Stick to established facts. Do not mention that you are part of a routing system.""",
        ),
        "openai/gpt-4.1": RouteModelProfile(
            model_id="openai/gpt-4.1",
            name="GPT-4.1",
            specialty="Deep reasoning, synthesis, coding, and complex problem-solving",
            system_prompt="""You are GPT-4.1, a deep-reasoning expert in a multi-model routing pool.

Your role: handle the hardest routed searches — synthesis, architecture, debugging, and multi-step logic.

Strengths:
- Connecting ideas across subtopics into a coherent explanation
- Code, algorithms, and system design
- Step-by-step reasoning for complex questions

Respond with structured, thorough but concise answers. Do not mention that you are part of a routing system.""",
        ),
        "openai/gpt-4o-mini": RouteModelProfile(
            model_id="openai/gpt-4o-mini",
            name="GPT-4o Mini",
            specialty="Quick answers for simple, low-complexity questions",
            system_prompt="""You are GPT-4o Mini, a fast lightweight expert in a multi-model routing pool.

Your role: answer simple routed searches quickly and cheaply.

Strengths:
- Straightforward definitions
- Brief rewrites and classifications
- Simple factual lookups

Keep answers under 150 words. Do not mention that you are part of a routing system.""",
        ),
        "meta-llama/llama-3.1-8b-instruct": RouteModelProfile(
            model_id="meta-llama/llama-3.1-8b-instruct",
            name="Llama 3.1 8B",
            specialty="General-purpose explanations at moderate depth",
            system_prompt="""You are Llama 3.1 8B, a general expert in a multi-model routing pool.

Your role: provide clear, accessible explanations for routed searches.

Strengths:
- Plain-language explanations for technical topics
- Moderate-depth overviews
- Practical examples

Respond in 1-3 short paragraphs. Do not mention that you are part of a routing system.""",
        ),
    }

    DEFAULT_PROFILE = RouteModelProfile(
        model_id="default",
        name="General Expert",
        specialty="General-purpose information retrieval",
        system_prompt="""You are an expert assistant in a multi-model routing pool.

Provide concise, accurate information for the coordinator's search query.
Focus on facts relevant to the question. Keep the response under 200 words.""",
    )

    @classmethod
    def get_profile(cls, model_id: str) -> RouteModelProfile:
        if model_id in cls.PROFILES:
            return cls.PROFILES[model_id]
        lowered = model_id.lower()
        for key, profile in cls.PROFILES.items():
            if key.lower() in lowered or lowered in key.lower():
                return profile
        if "claude" in lowered or "anthropic" in lowered:
            return cls.PROFILES["anthropic/claude-3.5-haiku"]
        if "gemini" in lowered or "google" in lowered:
            return cls.PROFILES["google/gemini-2.0-flash-001"]
        if "gpt-4" in lowered or "openai" in lowered:
            return cls.PROFILES["openai/gpt-4.1"]
        if "llama" in lowered or "meta" in lowered:
            return cls.PROFILES["meta-llama/llama-3.1-8b-instruct"]
        return cls.DEFAULT_PROFILE

    @classmethod
    def get_system_prompt(cls, model_id: str) -> str:
        return cls.get_profile(model_id).system_prompt

    @classmethod
    def format_user_prompt(cls, search_query: str, profile: RouteModelProfile) -> str:
        return (
            f"The coordinator has routed this search to you ({profile.name}) "
            f"because you specialize in: {profile.specialty}\n\n"
            f"Search query: {search_query}"
        )

    @classmethod
    def expert_catalog(cls, model_ids: List[str]) -> str:
        lines = ["Expert routing pool (one expert is assigned per <search>):"]
        for model_id in model_ids:
            profile = cls.get_profile(model_id)
            lines.append(f"- {profile.name} ({model_id}): {profile.specialty}")
        return "\n".join(lines)


class PromptPool:
    """Manages prompt templates for different models."""
    
    QWEN_SYSTEM_PROMPT = """You are an intelligent AI assistant that can reason through complex problems step by step.
    
You can use the following special tags:
- <search>query</search>: To search for information in the routing pool
- <information>data</information>: This contains results from your search
- <answer>final answer</answer>: To provide your final answer

Instructions:
1. Think step-by-step about the query
2. If you need information, use <search>query</search> tags
3. When you have enough information to answer, provide your final answer in <answer>tags</answer>
4. Be concise and direct in your reasoning"""

    COORDINATOR_PROMPT = """You are a weak coordinator model. You do NOT answer from your own knowledge.

Expert models in the routing pool handle all factual work. Your only job is to:
1. Read the user's question
2. Issue <search>specific sub-question</search> tags to consult experts
3. Read the <information>...</information> they return
4. When you have enough expert input, synthesize a final response in <answer>...</answer>

{expert_catalog}

When writing <search> queries, tailor them to the expert's specialty:
- Factual definitions or quick facts → phrase as a lookup question
- Comparisons or trade-offs → ask for analysis of options
- Complex synthesis, code, or architecture → ask for step-by-step reasoning

Rules:
- NEVER skip searching — always start with <search> before any <answer>
- Break complex questions into separate <search> calls (one topic per search)
- Do not invent facts; only use information returned by experts
- Keep your own text minimal between tags"""

    LLAMA_SYSTEM_PROMPT = """You are a helpful AI assistant capable of multi-step reasoning.

Use these special tags when needed:
- <search>query</search>: Search for information
- <information>data</information>: Contains search results  
- <answer>final answer</answer>: Your final response

Guide:
1. Analyze the question carefully
2. Search for information if needed using the tags above
3. Provide your final answer wrapped in <answer>tags</answer>"""

    @classmethod
    def get_system_prompt(
        cls,
        model_id: str,
        *,
        coordinator_mode: bool = False,
        route_models: Optional[List[str]] = None,
    ) -> str:
        """Get appropriate system prompt for a model."""
        if coordinator_mode:
            catalog = RoutePromptPool.expert_catalog(route_models or [])
            return cls.COORDINATOR_PROMPT.format(expert_catalog=catalog)
        if "qwen" in model_id.lower():
            return cls.QWEN_SYSTEM_PROMPT
        if "llama" in model_id.lower():
            return cls.LLAMA_SYSTEM_PROMPT
        return cls.QWEN_SYSTEM_PROMPT

    @classmethod
    def format_user_prompt(cls, query: str, previous_exchanges: List[str] = None) -> str:
        """Format user prompt for multi-round interaction.
        
        Args:
            query: User query
            previous_exchanges: Previous exchanges in conversation
            
        Returns:
            Formatted prompt
        """
        if not previous_exchanges:
            return f"Answer this question step by step:\n\n{query}"
        
        prompt = "Continue reasoning with the new information:\n\n"
        for exchange in previous_exchanges:
            prompt += f"{exchange}\n\n"
        prompt += f"Query: {query}"
        return prompt


class RouteService:
    """Manages API calls to the external routing pool."""
    
    def __init__(self, api_base: str, api_key: str):
        """Initialize routing service.
        
        Args:
            api_base: Base URL for routing pool API
            api_key: API key for authentication
        """
        self.api_base = api_base
        self.api_key = api_key
        self.client = OpenAI(api_key=api_key, base_url=api_base)
        self.total_tokens = 0
    
    def search(self, query: str, model: str = "openai/gpt-4o-mini") -> Dict[str, Any]:
        """Query the routing pool for information."""
        profile = RoutePromptPool.get_profile(model)
        system_prompt = profile.system_prompt
        user_prompt = RoutePromptPool.format_user_prompt(query, profile)

        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.7,
                max_tokens=256,
            )
            
            result = response.choices[0].message.content
            tokens = response.usage.completion_tokens
            self.total_tokens += tokens
            
            return {
                "success": True,
                "result": result,
                "tokens": tokens,
                "model": model,
                "expert_name": profile.name,
                "specialty": profile.specialty,
                "system_prompt": system_prompt,
            }
        except Exception as e:
            logger.error(f"Routing pool query failed: {e}")
            return {
                "success": False,
                "result": f"Failed to retrieve information: {str(e)}",
                "tokens": 0,
                "model": model,
                "expert_name": profile.name,
                "specialty": profile.specialty,
                "system_prompt": system_prompt,
            }


class RouterR1:
    """Advanced multi-round reasoning router with dynamic routing.
    
    Combines iterative reasoning with dynamic routing to external knowledge sources.
    Uses vLLM for efficient local inference and OpenAI-compatible APIs for routing pools.
    """
    
    # Regex patterns for extracting special tags
    SEARCH_PATTERN = re.compile(r'<search>(.*?)</search>', re.DOTALL)
    ANSWER_PATTERN = re.compile(r'<answer>(.*?)</answer>', re.DOTALL)
    
    def __init__(self, yaml_path: Optional[str] = None, config: Optional[RouterR1Config] = None):
        """Initialize Router-R1.
        
        Args:
            yaml_path: Path to YAML configuration file
            config: RouterR1Config instance (alternative to yaml_path)
            
        Raises:
            ValueError: If neither yaml_path nor config is provided
        """
        if yaml_path:
            self.config = RouterR1Config.from_yaml(yaml_path)
        elif config:
            self.config = config
        else:
            raise ValueError("Either yaml_path or config must be provided")
        
        # Get API credentials from environment or config
        self.config.api_key = self.config.api_key or os.getenv("OPENROUTER_API_KEY") or os.getenv("API_KEYS")
        self.config.api_base = (
            self.config.api_base
            or os.getenv("API_BASE")
            or os.getenv("OPENROUTER_BASE_URL")
            or "https://openrouter.ai/api/v1"
        )

        self.backend = self._resolve_backend()
        self.llm = None
        self.api_client = None

        if self.backend == "vllm":
            self._init_vllm()
        else:
            self._init_api_client()

        # Initialize routing service
        if self.config.api_base and self.config.api_key:
            self.route_service = RouteService(self.config.api_base, self.config.api_key)
        else:
            logger.warning("Routing pool not configured. Routing will be disabled.")
            self.route_service = None
        
        # Token tracking
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_route_tokens = 0

        if self.config.route_models:
            self.route_models = list(self.config.route_models)
        else:
            self.route_models = [self.config.route_model]
        self._route_model_index = 0

    def _next_route_model(self) -> str:
        model = self.route_models[self._route_model_index % len(self.route_models)]
        self._route_model_index += 1
        return model

    def _resolve_backend(self) -> str:
        backend = (self.config.backend or "auto").lower()
        if backend == "auto":
            return "vllm" if _VLLM_AVAILABLE else "api"
        if backend == "vllm" and not _VLLM_AVAILABLE:
            raise ImportError(
                "vLLM backend requested but vllm is not installed. "
                "Install with: pip install vllm torch  (Linux/WSL + CUDA), "
                "or set backend: api in configs/router_r1.yaml"
            )
        if backend not in {"vllm", "api"}:
            raise ValueError(f"Unknown backend '{backend}'. Use 'auto', 'vllm', or 'api'.")
        return backend

    def _init_api_client(self) -> None:
        if not self.config.api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is required for the API backend. "
                "Add it to your .env file."
            )
        logger.info(f"Using API backend with model: {self.config.model_id}")
        self.api_client = OpenAI(api_key=self.config.api_key, base_url=self.config.api_base)
    
    def _init_vllm(self):
        """Initialize vLLM instance for local inference."""
        logger.info(f"Loading model: {self.config.model_id}")
        
        # Auto-detect tensor parallel size if not specified
        if self.config.tensor_parallel_size is None:
            import torch
            self.config.tensor_parallel_size = torch.cuda.device_count()
        
        self.llm = LLM(
            model=self.config.model_id,
            tensor_parallel_size=self.config.tensor_parallel_size,
            gpu_memory_utilization=self.config.gpu_memory_utilization,
            dtype="auto",
        )
        logger.info(f"Model loaded successfully with tensor_parallel_size={self.config.tensor_parallel_size}")
    
    def _perform_route(
        self, search_query: str, generation: str, *, forced: bool = False
    ) -> tuple[str, Dict[str, Any]]:
        route_model = self._next_route_model()
        route_info = self.route_service.search(search_query, model=route_model)
        if forced:
            route_info["forced"] = True
        self.total_route_tokens += route_info.get("tokens", 0)
        information = route_info.get("result", "No information found")
        current_input = f"{generation}\n\n<information>\n{information}\n</information>"
        return current_input, route_info

    def route_single(
        self,
        query: str,
        return_details: bool = True,
        route_only: bool = False,
    ) -> Dict[str, Any]:
        """Route a single query with multi-round reasoning.
        
        Args:
            query: The query to route
            return_details: Whether to return detailed reasoning trace
            route_only: If True, only perform routing without LLM generation
            
        Returns:
            Dictionary with response and metadata
        """
        traces = []
        current_input = query
        model_name = self.config.model_id
        self._route_model_index = 0
        
        for iteration in range(self.config.max_iterations):
            logger.info(f"Iteration {iteration}: Processing query")
            
            # Generate response from vLLM
            generation, tokens = self._generate(current_input, iteration)
            logger.debug(f"Generated output: {generation[:200]}...")
            
            # Check for answer
            answer_match = self.ANSWER_PATTERN.search(generation)
            if answer_match:
                if (
                    self.config.coordinator_mode
                    and self.total_route_tokens == 0
                    and self.route_service
                    and not route_only
                ):
                    logger.info("Coordinator mode: ignoring premature answer, routing to experts first")
                    search_query = query
                    current_input, route_info = self._perform_route(
                        search_query, generation, forced=True
                    )
                    traces.append(IterationTrace(
                        iteration=iteration,
                        generation=generation,
                        search_query=search_query,
                        route_info=route_info,
                        tokens_used=tokens,
                    ))
                    continue

                final_answer = answer_match.group(1).strip()
                traces.append(IterationTrace(
                    iteration=iteration,
                    generation=generation,
                    tokens_used=tokens,
                ))
                logger.info(f"Answer found at iteration {iteration}")
                break
            
            # Check for search query
            search_match = self.SEARCH_PATTERN.search(generation)
            search_query = None
            route_info = None
            
            if search_match and not route_only:
                search_query = search_match.group(1).strip()
                logger.info(f"Search query: {search_query}")

                if self.route_service:
                    current_input, route_info = self._perform_route(search_query, generation)
                else:
                    logger.warning("Routing service not configured, skipping search")
                    current_input = generation
            elif self.config.coordinator_mode and not route_only and self.route_service:
                search_query = (
                    query if iteration == 0 else f"Additional expert input needed for: {query}"
                )
                logger.info(f"Coordinator mode: forcing route — {search_query}")
                current_input, route_info = self._perform_route(
                    search_query, generation, forced=True
                )
            else:
                current_input = generation
            
            traces.append(IterationTrace(
                iteration=iteration,
                generation=generation,
                search_query=search_query,
                route_info=route_info,
                tokens_used=tokens,
            ))
        else:
            # No answer found after max iterations
            final_answer = f"[Incomplete reasoning after {self.config.max_iterations} iterations]\n\n{current_input}"
        
        # Compile response
        response = RouterR1Response(
            query=query,
            final_answer=final_answer,
            reasoning_trace=traces if return_details else [],
            model_used=model_name,
            total_tokens={
                "prompt_tokens": self.total_prompt_tokens,
                "completion_tokens": self.total_completion_tokens,
                "route_tokens": self.total_route_tokens,
            },
            iteration_count=len(traces),
        )
        
        return {
            "query": query,
            "response": final_answer,
            "model_name": model_name,
            "route_models": self.route_models,
            "route_model": self.route_models[0],
            "route_profiles": [
                {
                    "model_id": m,
                    "name": RoutePromptPool.get_profile(m).name,
                    "specialty": RoutePromptPool.get_profile(m).specialty,
                }
                for m in self.route_models
            ],
            "coordinator_mode": self.config.coordinator_mode,
            "backend": self.backend,
            "total_tokens": response.total_tokens,
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "route_tokens": self.total_route_tokens,
            "reasoning_trace": traces if return_details else [],
            "iterations": len(traces),
        }
    
    def route_batch(
        self,
        queries: List[Dict[str, Any]],
        task_name: str = "batch",
        return_details: bool = True,
    ) -> List[Dict[str, Any]]:
        """Route multiple queries.
        
        Args:
            queries: List of query dictionaries with 'query' key
            task_name: Name of the task for logging
            return_details: Whether to return detailed reasoning traces
            
        Returns:
            List of routing responses
        """
        logger.info(f"Starting batch routing for {task_name} with {len(queries)} queries")
        results = []
        
        for i, query_dict in enumerate(queries):
            query = query_dict.get("query", "")
            logger.info(f"Processing query {i+1}/{len(queries)}")
            
            result = self.route_single(query, return_details=return_details)
            result["task_name"] = task_name
            result["query_index"] = i
            results.append(result)
        
        logger.info(f"Batch routing completed for {task_name}")
        return results
    
    def _generate(self, prompt: str, iteration: int) -> tuple[str, Dict[str, int]]:
        """Generate text using the configured backend."""
        if self.backend == "vllm":
            return self._generate_vllm(prompt)
        return self._generate_api(prompt)

    def _generate_vllm(self, prompt: str) -> tuple[str, Dict[str, int]]:
        system_prompt = PromptPool.get_system_prompt(
            self.config.model_id,
            coordinator_mode=self.config.coordinator_mode,
            route_models=self.route_models,
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        formatted_prompt = self.llm.get_tokenizer().apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        sampling_params = SamplingParams(
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
            max_tokens=self.config.max_tokens,
            stop=["</search>", "</answer>"],
        )
        outputs = self.llm.generate([formatted_prompt], sampling_params)
        generated_text = outputs[0].outputs[0].text
        prompt_tokens = len(outputs[0].prompt_token_ids)
        completion_tokens = len(outputs[0].outputs[0].token_ids)
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        return generated_text, {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }

    def _generate_api(self, prompt: str) -> tuple[str, Dict[str, int]]:
        system_prompt = PromptPool.get_system_prompt(
            self.config.model_id,
            coordinator_mode=self.config.coordinator_mode,
            route_models=self.route_models,
        )
        response = self.api_client.chat.completions.create(
            model=self.config.model_id,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_tokens=self.config.max_tokens,
        )
        generated_text = response.choices[0].message.content or ""
        usage = response.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        return generated_text, {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }
    
    def save_result(self, result: Dict[str, Any], output_path: str):
        """Save routing result to file.
        
        Args:
            result: Routing result dictionary
            output_path: Path to save the result
        """
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        logger.info(f"Result saved to {output_path}")
    
    def get_stats(self) -> Dict[str, int]:
        """Get token usage statistics.
        
        Returns:
            Dictionary with token counts
        """
        return {
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "route_tokens": self.total_route_tokens,
            "total_tokens": self.total_prompt_tokens + self.total_completion_tokens + self.total_route_tokens,
        }
