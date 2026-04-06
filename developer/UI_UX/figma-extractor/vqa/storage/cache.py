"""Caching layer for VQA pipeline.

Provides high-level caching interface for comparison results and screenshots.
"""

import logging
import os
from typing import Optional, Dict, Any

from .database import VQADatabase, get_database

logger = logging.getLogger(__name__)


class ComparisonCache:
    """High-level cache for VQA comparison results.
    
    Usage:
        cache = ComparisonCache()
        
        # Check for cached result
        result = cache.get(figma_url, web_url, node_id)
        if result:
            return result  # Cache hit
        
        # Run comparison...
        result = run_comparison(...)
        
        # Store in cache
        cache.store(figma_url, web_url, node_id, result)
    """
    
    def __init__(
        self,
        db: VQADatabase = None,
        enabled: bool = None,
    ):
        """Initialize the cache.
        
        Args:
            db: Database instance (uses global if not provided)
            enabled: Whether caching is enabled (from env if not provided)
        """
        self._db = db
        self._enabled = enabled
        
        if self._enabled is None:
            self._enabled = os.getenv("VQA_CACHE_ENABLED", "true").lower() == "true"
    
    @property
    def db(self) -> VQADatabase:
        """Get the database instance (lazy initialization)."""
        if self._db is None:
            self._db = get_database()
        return self._db
    
    @property
    def enabled(self) -> bool:
        """Check if caching is enabled."""
        return self._enabled
    
    def get(
        self,
        figma_url: str,
        web_url: str,
        node_id: str = None,
    ) -> Optional[Dict[str, Any]]:
        """Get cached comparison result.
        
        Args:
            figma_url: Figma file URL
            web_url: Web page URL
            node_id: Optional Figma node ID
        
        Returns:
            Cached result dict with '_cached': True, or None if not cached
        """
        if not self.enabled:
            return None
        
        try:
            return self.db.get_comparison(figma_url, web_url, node_id)
        except Exception as e:
            logger.warning(f"Cache get error: {e}")
            return None
    
    def store(
        self,
        figma_url: str,
        web_url: str,
        results: Dict[str, Any],
        node_id: str = None,
    ) -> Optional[str]:
        """Store comparison result in cache.
        
        Args:
            figma_url: Figma file URL
            web_url: Web page URL
            results: Comparison results dict
            node_id: Optional Figma node ID
        
        Returns:
            Comparison ID if stored, None if caching disabled
        """
        if not self.enabled:
            return None
        
        try:
            return self.db.store_comparison(figma_url, web_url, results, node_id)
        except Exception as e:
            logger.warning(f"Cache store error: {e}")
            return None
    
    def invalidate(
        self,
        figma_url: str = None,
        web_url: str = None,
    ) -> int:
        """Invalidate cached comparisons.
        
        Args:
            figma_url: Invalidate all with this Figma URL
            web_url: Invalidate all with this web URL
            If both None, invalidates all cached comparisons
        
        Returns:
            Number of entries invalidated
        """
        if not self.enabled:
            return 0
        
        try:
            return self.db.invalidate_comparison(figma_url, web_url)
        except Exception as e:
            logger.warning(f"Cache invalidate error: {e}")
            return 0
    
    def store_screenshot(
        self,
        data: bytes,
        figma_url: str = None,
        web_url: str = None,
        node_id: str = None,
        source: str = "unknown",
    ) -> Optional[str]:
        """Store a screenshot in the cache.
        
        Args:
            data: Screenshot bytes
            figma_url: Associated Figma URL
            web_url: Associated web URL
            node_id: Figma node ID
            source: 'figma' or 'web'
        
        Returns:
            Screenshot ID if stored
        """
        if not self.enabled:
            return None
        
        try:
            return self.db.store_screenshot(data, figma_url, web_url, node_id, source)
        except Exception as e:
            logger.warning(f"Screenshot store error: {e}")
            return None
    
    def get_screenshot(self, screenshot_id: str) -> Optional[bytes]:
        """Get a screenshot from cache.
        
        Args:
            screenshot_id: Screenshot ID
        
        Returns:
            Screenshot bytes or None
        """
        if not self.enabled:
            return None
        
        try:
            return self.db.get_screenshot(screenshot_id)
        except Exception as e:
            logger.warning(f"Screenshot get error: {e}")
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dict with cache statistics
        """
        try:
            stats = self.db.get_stats()
            stats["enabled"] = self.enabled
            return stats
        except Exception as e:
            logger.warning(f"Stats error: {e}")
            return {"enabled": self.enabled, "error": str(e)}
    
    def cleanup(self) -> Dict[str, int]:
        """Remove expired entries.
        
        Returns:
            Dict with cleanup counts
        """
        try:
            return self.db.cleanup_expired()
        except Exception as e:
            logger.warning(f"Cleanup error: {e}")
            return {"error": str(e)}


# Global cache instance
_global_cache: Optional[ComparisonCache] = None


def get_cache() -> ComparisonCache:
    """Get the global cache instance.
    
    Returns:
        ComparisonCache instance
    """
    global _global_cache
    
    if _global_cache is None:
        _global_cache = ComparisonCache()
    
    return _global_cache
