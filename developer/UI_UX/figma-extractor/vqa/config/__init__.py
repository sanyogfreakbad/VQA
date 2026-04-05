"""VQA Configuration - All tunable parameters in one place."""

from .thresholds import Thresholds, THRESHOLDS
from .llm_config import LLMConfig, LLM_CONFIG

__all__ = [
    "Thresholds",
    "THRESHOLDS",
    "LLMConfig",
    "LLM_CONFIG",
]
