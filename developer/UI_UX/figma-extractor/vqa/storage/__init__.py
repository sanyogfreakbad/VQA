"""Storage module for VQA pipeline.

Provides persistent caching and feedback storage using SQLite.
"""

from .database import VQADatabase, get_database
from .cache import ComparisonCache, get_cache

__all__ = [
    "VQADatabase",
    "get_database",
    "ComparisonCache",
    "get_cache",
]
