"""LLM configuration for Gemini API."""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class LLMConfig:
    """Configuration for Gemini LLM calls."""
    
    # Model settings
    model: str = "gemini-2.0-flash"
    temperature: float = 0.2
    max_output_tokens: int = 4096
    
    # Rate limiting
    max_retries: int = 3
    retry_delay: float = 1.0
    concurrent_requests: int = 5
    
    # Token limits per request
    max_image_tokens: int = 258  # Gemini image token cost
    max_context_tokens: int = 8192
    
    # Batching
    batch_size: int = 5
    batch_delay: float = 0.5
    
    # API settings
    api_key: Optional[str] = None
    
    def __post_init__(self):
        """Load API key from environment if not provided."""
        if self.api_key is None:
            self.api_key = os.getenv("GEMINI_API_KEY")
    
    @property
    def is_configured(self) -> bool:
        """Check if API key is available."""
        return self.api_key is not None and len(self.api_key) > 0
    
    def to_dict(self) -> dict:
        """Convert to dictionary (excluding API key)."""
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
            "max_retries": self.max_retries,
            "retry_delay": self.retry_delay,
            "concurrent_requests": self.concurrent_requests,
            "batch_size": self.batch_size,
            "is_configured": self.is_configured,
        }


# Default LLM config instance
LLM_CONFIG = LLMConfig()
