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
    from vllm import LLM, SamplingParams
except ImportError:
    raise ImportError("vllm is required for Router-R1. Install with: pip install vllm==0.6.3 torch==2.4.0")

try:
    from openai import OpenAI
except ImportError:
    raise ImportError("openai is required for Router-R1. Install with: pip install openai>=1.0")


logger = logging.getLogger(__name__)


@dataclass
class RouterR1Config:
    """Configuration for Router-R1."""
    
    model_id: str
    api_base: Optional[str] = None
    api_key: Optional[str] = None
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
    def get_system_prompt(cls, model_id: str) -> str:
        """Get appropriate system prompt for a model.
        
        Args:
            model_id: HuggingFace model ID
            
        Returns:
            System prompt string
        """
        if "qwen" in model_id.lower():
            return cls.QWEN_SYSTEM_PROMPT
        elif "llama" in model_id.lower():
            return cls.LLAMA_SYSTEM_PROMPT
        else:
            return cls.QWEN_SYSTEM_PROMPT  # Default fallback

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
    
    def search(self, query: str, model: str = "gpt-3.5-turbo") -> Dict[str, Any]:
        """Query the routing pool for information.
        
        Args:
            query: Search query
            model: Model to use for routing pool
            
        Returns:
            Dictionary with results and token usage
        """
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful information retrieval assistant. Provide concise, relevant information."
                    },
                    {
                        "role": "user",
                        "content": f"Provide information about: {query}"
                    }
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
            }
        except Exception as e:
            logger.error(f"Routing pool query failed: {e}")
            return {
                "success": False,
                "result": f"Failed to retrieve information: {str(e)}",
                "tokens": 0,
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
        self.config.api_base = self.config.api_base or os.getenv("API_BASE") or os.getenv("OPENROUTER_BASE_URL")
        
        # Initialize vLLM
        self._init_vllm()
        
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
        
        for iteration in range(self.config.max_iterations):
            logger.info(f"Iteration {iteration}: Processing query")
            
            # Generate response from vLLM
            generation, tokens = self._generate(current_input, iteration)
            logger.debug(f"Generated output: {generation[:200]}...")
            
            # Check for answer
            answer_match = self.ANSWER_PATTERN.search(generation)
            if answer_match:
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
                
                # Call routing pool
                if self.route_service:
                    route_info = self.route_service.search(search_query)
                    self.total_route_tokens += route_info.get("tokens", 0)
                    
                    # Augment generation with retrieved information
                    information = route_info.get("result", "No information found")
                    augmented = f"{generation}\n\n<information>\n{information}\n</information>"
                    current_input = augmented
                else:
                    logger.warning("Routing service not configured, skipping search")
                    current_input = generation
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
        """Generate text using vLLM.
        
        Args:
            prompt: Input prompt
            iteration: Current iteration number
            
        Returns:
            Tuple of (generated_text, token_counts)
        """
        system_prompt = PromptPool.get_system_prompt(self.config.model_id)
        
        # Format messages for chat template
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        
        # Apply chat template
        formatted_prompt = self.llm.get_tokenizer().apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        # Generate
        sampling_params = SamplingParams(
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
            max_tokens=self.config.max_tokens,
            stop=["</search>", "</answer>"],
        )
        
        outputs = self.llm.generate([formatted_prompt], sampling_params)
        
        # Extract generated text and token counts
        generated_text = outputs[0].outputs[0].text
        prompt_tokens = len(outputs[0].prompt_token_ids)
        completion_tokens = len(outputs[0].outputs[0].token_ids)
        
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
