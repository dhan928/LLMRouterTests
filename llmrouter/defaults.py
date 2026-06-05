from __future__ import annotations

import os


def default_model_catalog() -> dict[str, dict[str, str]]:
    return {
        "weak": {
            "openrouter_model": os.getenv("OPENROUTER_WEAK_MODEL", "openai/gpt-4o-mini"),
            "description": "Lower-cost model for simple questions.",
        },
        "strong": {
            "openrouter_model": os.getenv("OPENROUTER_STRONG_MODEL", "openai/gpt-4.1"),
            "description": "Stronger model for harder reasoning, coding, and analysis.",
        },
    }


TRAINING_QUERIES = [
    "Say hello in a friendly way.",
    "Summarize this short paragraph in one sentence.",
    "What is the capital of France?",
    "Rewrite this sentence to sound more professional.",
    "Give me three quick dinner ideas.",
    "Classify this message as positive or negative.",
    "Explain quantum entanglement with equations and caveats.",
    "Debug this Python function and explain the root cause.",
    "Design a scalable architecture for a production API.",
    "Compare two legal contract clauses and identify risks.",
    "Solve this multi-step algebra proof carefully.",
    "Write a detailed migration plan for a complex codebase.",
]

TRAINING_LABELS = [
    "weak",
    "weak",
    "weak",
    "weak",
    "weak",
    "weak",
    "strong",
    "strong",
    "strong",
    "strong",
    "strong",
    "strong",
]
