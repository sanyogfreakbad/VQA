"""SQLite database for VQA pipeline persistence.

Tables:
- screenshots: Stores screenshot blobs with TTL
- comparisons: Stores comparison results
- feedback: Stores user feedback on findings
"""

import json
import logging
import os
import sqlite3
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List
from contextlib import contextmanager
from threading import Lock

logger = logging.getLogger(__name__)


class VQADatabase:
    """SQLite database for VQA pipeline storage."""
    
    DEFAULT_DB_PATH = "data/vqa_cache.db"
    DEFAULT_TTL_HOURS = 24
    
    def __init__(
        self,
        db_path: Optional[str] = None,
        ttl_hours: int = None,
    ):
        """Initialize the database.
        
        Args:
            db_path: Path to SQLite database file
            ttl_hours: Time-to-live for cached items in hours
        """
        self.db_path = Path(db_path or os.getenv("VQA_SQLITE_PATH", self.DEFAULT_DB_PATH))
        self.ttl_hours = ttl_hours or int(os.getenv("VQA_CACHE_TTL_HOURS", self.DEFAULT_TTL_HOURS))
        self._lock = Lock()
        self._init_db()
    
    def _init_db(self) -> None:
        """Initialize database schema."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        with self._get_connection() as conn:
            # Enable WAL mode for better concurrent access
            conn.execute("PRAGMA journal_mode=WAL")
            
            # Screenshots table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS screenshots (
                    id TEXT PRIMARY KEY,
                    hash TEXT UNIQUE NOT NULL,
                    data BLOB NOT NULL,
                    figma_url TEXT,
                    web_url TEXT,
                    node_id TEXT,
                    source TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP NOT NULL
                )
            """)
            
            # Comparisons table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS comparisons (
                    id TEXT PRIMARY KEY,
                    cache_key TEXT UNIQUE NOT NULL,
                    figma_url TEXT NOT NULL,
                    web_url TEXT NOT NULL,
                    figma_node_id TEXT,
                    results_json TEXT NOT NULL,
                    quality_score REAL,
                    findings_count INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP NOT NULL
                )
            """)
            
            # Feedback table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    id TEXT PRIMARY KEY,
                    comparison_id TEXT NOT NULL,
                    finding_id TEXT NOT NULL,
                    verdict TEXT NOT NULL CHECK(verdict IN ('confirmed', 'rejected')),
                    reasoning TEXT,
                    user_id TEXT,
                    figma_crop_hash TEXT,
                    web_crop_hash TEXT,
                    category TEXT,
                    diff_type TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (comparison_id) REFERENCES comparisons(id)
                )
            """)
            
            # Indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_screenshots_hash ON screenshots(hash)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_screenshots_expires ON screenshots(expires_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_comparisons_cache_key ON comparisons(cache_key)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_comparisons_expires ON comparisons(expires_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_feedback_comparison ON feedback(comparison_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_feedback_category ON feedback(category)")
            
            conn.commit()
    
    @contextmanager
    def _get_connection(self):
        """Get a database connection with context management."""
        conn = sqlite3.connect(
            str(self.db_path),
            timeout=30.0,
            check_same_thread=False,
        )
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def _generate_id(self, prefix: str = "") -> str:
        """Generate a unique ID."""
        import uuid
        return f"{prefix}{uuid.uuid4().hex[:12]}"
    
    def _compute_cache_key(self, figma_url: str, web_url: str, node_id: str = None) -> str:
        """Compute cache key from URLs."""
        key_str = f"{figma_url}|{web_url}|{node_id or ''}"
        return hashlib.sha256(key_str.encode()).hexdigest()
    
    def _expires_at(self, hours: int = None) -> str:
        """Calculate expiration timestamp."""
        hours = hours or self.ttl_hours
        expires = datetime.now() + timedelta(hours=hours)
        return expires.isoformat()
    
    # =========================================================================
    # Screenshot Methods
    # =========================================================================
    
    def store_screenshot(
        self,
        data: bytes,
        figma_url: str = None,
        web_url: str = None,
        node_id: str = None,
        source: str = "unknown",
    ) -> str:
        """Store a screenshot and return its ID.
        
        Args:
            data: Screenshot bytes (PNG)
            figma_url: Associated Figma URL
            web_url: Associated web URL
            node_id: Figma node ID
            source: Source identifier ('figma' or 'web')
        
        Returns:
            Screenshot ID (hash-based)
        """
        screenshot_hash = hashlib.sha256(data).hexdigest()
        screenshot_id = f"ss_{screenshot_hash[:16]}"
        
        with self._lock:
            with self._get_connection() as conn:
                # Check if already exists
                existing = conn.execute(
                    "SELECT id FROM screenshots WHERE hash = ?",
                    (screenshot_hash,)
                ).fetchone()
                
                if existing:
                    # Update expiration
                    conn.execute(
                        "UPDATE screenshots SET expires_at = ? WHERE hash = ?",
                        (self._expires_at(), screenshot_hash)
                    )
                    conn.commit()
                    return existing["id"]
                
                # Insert new screenshot
                conn.execute("""
                    INSERT INTO screenshots (id, hash, data, figma_url, web_url, node_id, source, expires_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    screenshot_id,
                    screenshot_hash,
                    data,
                    figma_url,
                    web_url,
                    node_id,
                    source,
                    self._expires_at(),
                ))
                conn.commit()
        
        logger.debug(f"Stored screenshot: {screenshot_id}")
        return screenshot_id
    
    def get_screenshot(self, screenshot_id: str) -> Optional[bytes]:
        """Retrieve a screenshot by ID.
        
        Args:
            screenshot_id: Screenshot ID
        
        Returns:
            Screenshot bytes or None if not found/expired
        """
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT data FROM screenshots WHERE id = ? AND expires_at > ?",
                (screenshot_id, datetime.now().isoformat())
            ).fetchone()
            
            if row:
                return row["data"]
        return None
    
    def get_screenshot_by_hash(self, hash_value: str) -> Optional[bytes]:
        """Retrieve a screenshot by its hash."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT data FROM screenshots WHERE hash = ? AND expires_at > ?",
                (hash_value, datetime.now().isoformat())
            ).fetchone()
            
            if row:
                return row["data"]
        return None
    
    # =========================================================================
    # Comparison Methods
    # =========================================================================
    
    def store_comparison(
        self,
        figma_url: str,
        web_url: str,
        results: Dict[str, Any],
        figma_node_id: str = None,
    ) -> str:
        """Store comparison results.
        
        Args:
            figma_url: Figma file URL
            web_url: Web page URL
            results: Comparison results dict
            figma_node_id: Optional Figma node ID
        
        Returns:
            Comparison ID
        """
        cache_key = self._compute_cache_key(figma_url, web_url, figma_node_id)
        comparison_id = self._generate_id("cmp_")
        
        quality_score = results.get("quality_score", {}).get("score")
        findings = results.get("findings", [])
        findings_count = len(findings)
        
        with self._lock:
            with self._get_connection() as conn:
                # Delete existing entry with same cache key
                conn.execute("DELETE FROM comparisons WHERE cache_key = ?", (cache_key,))
                
                # Insert new comparison
                conn.execute("""
                    INSERT INTO comparisons 
                    (id, cache_key, figma_url, web_url, figma_node_id, results_json, quality_score, findings_count, expires_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    comparison_id,
                    cache_key,
                    figma_url,
                    web_url,
                    figma_node_id,
                    json.dumps(results),
                    quality_score,
                    findings_count,
                    self._expires_at(),
                ))
                conn.commit()
        
        logger.info(f"Stored comparison: {comparison_id} (score={quality_score}, findings={findings_count})")
        return comparison_id
    
    def get_comparison(
        self,
        figma_url: str,
        web_url: str,
        figma_node_id: str = None,
    ) -> Optional[Dict[str, Any]]:
        """Retrieve cached comparison results.
        
        Args:
            figma_url: Figma file URL
            web_url: Web page URL
            figma_node_id: Optional Figma node ID
        
        Returns:
            Comparison results dict or None if not cached/expired
        """
        cache_key = self._compute_cache_key(figma_url, web_url, figma_node_id)
        
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT id, results_json FROM comparisons WHERE cache_key = ? AND expires_at > ?",
                (cache_key, datetime.now().isoformat())
            ).fetchone()
            
            if row:
                results = json.loads(row["results_json"])
                results["_cached"] = True
                results["_comparison_id"] = row["id"]
                logger.info(f"Cache hit for comparison: {row['id']}")
                return results
        
        return None
    
    def get_comparison_by_id(self, comparison_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve comparison by ID."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT results_json FROM comparisons WHERE id = ?",
                (comparison_id,)
            ).fetchone()
            
            if row:
                return json.loads(row["results_json"])
        return None
    
    def invalidate_comparison(
        self,
        figma_url: str = None,
        web_url: str = None,
    ) -> int:
        """Invalidate cached comparisons.
        
        Args:
            figma_url: Invalidate all with this Figma URL
            web_url: Invalidate all with this web URL
        
        Returns:
            Number of entries invalidated
        """
        with self._lock:
            with self._get_connection() as conn:
                conditions = []
                params = []
                
                if figma_url:
                    conditions.append("figma_url = ?")
                    params.append(figma_url)
                if web_url:
                    conditions.append("web_url = ?")
                    params.append(web_url)
                
                if not conditions:
                    # Invalidate all
                    result = conn.execute("DELETE FROM comparisons")
                else:
                    query = f"DELETE FROM comparisons WHERE {' OR '.join(conditions)}"
                    result = conn.execute(query, params)
                
                conn.commit()
                deleted = result.rowcount
        
        logger.info(f"Invalidated {deleted} cached comparisons")
        return deleted
    
    # =========================================================================
    # Feedback Methods
    # =========================================================================
    
    def store_feedback(
        self,
        comparison_id: str,
        finding_id: str,
        verdict: str,
        reasoning: str = None,
        user_id: str = None,
        figma_crop_hash: str = None,
        web_crop_hash: str = None,
        category: str = None,
        diff_type: str = None,
    ) -> str:
        """Store user feedback on a finding.
        
        Args:
            comparison_id: ID of the comparison
            finding_id: ID of the finding
            verdict: 'confirmed' or 'rejected'
            reasoning: User's reasoning
            user_id: Optional user identifier
            figma_crop_hash: Hash of Figma crop image
            web_crop_hash: Hash of web crop image
            category: Finding category
            diff_type: Type of difference
        
        Returns:
            Feedback ID
        """
        if verdict not in ("confirmed", "rejected"):
            raise ValueError("verdict must be 'confirmed' or 'rejected'")
        
        feedback_id = self._generate_id("fb_")
        
        with self._lock:
            with self._get_connection() as conn:
                conn.execute("""
                    INSERT INTO feedback 
                    (id, comparison_id, finding_id, verdict, reasoning, user_id, 
                     figma_crop_hash, web_crop_hash, category, diff_type)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    feedback_id,
                    comparison_id,
                    finding_id,
                    verdict,
                    reasoning,
                    user_id,
                    figma_crop_hash,
                    web_crop_hash,
                    category,
                    diff_type,
                ))
                conn.commit()
        
        logger.info(f"Stored feedback: {feedback_id} ({verdict})")
        return feedback_id
    
    def get_feedback_stats(self) -> Dict[str, Any]:
        """Get statistics about collected feedback.
        
        Returns:
            Dict with feedback statistics by category
        """
        with self._get_connection() as conn:
            # Overall stats
            total = conn.execute("SELECT COUNT(*) as count FROM feedback").fetchone()["count"]
            confirmed = conn.execute(
                "SELECT COUNT(*) as count FROM feedback WHERE verdict = 'confirmed'"
            ).fetchone()["count"]
            rejected = conn.execute(
                "SELECT COUNT(*) as count FROM feedback WHERE verdict = 'rejected'"
            ).fetchone()["count"]
            
            # Stats by category
            category_stats = {}
            rows = conn.execute("""
                SELECT category, verdict, COUNT(*) as count 
                FROM feedback 
                WHERE category IS NOT NULL
                GROUP BY category, verdict
            """).fetchall()
            
            for row in rows:
                cat = row["category"]
                if cat not in category_stats:
                    category_stats[cat] = {"confirmed": 0, "rejected": 0}
                category_stats[cat][row["verdict"]] = row["count"]
        
        return {
            "total": total,
            "confirmed": confirmed,
            "rejected": rejected,
            "by_category": category_stats,
        }
    
    def get_feedback_for_category(
        self,
        category: str,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get feedback entries for a specific category.
        
        Args:
            category: Category to filter by
            limit: Maximum entries to return
        
        Returns:
            List of feedback entries
        """
        with self._get_connection() as conn:
            rows = conn.execute("""
                SELECT * FROM feedback 
                WHERE category = ? 
                ORDER BY created_at DESC 
                LIMIT ?
            """, (category, limit)).fetchall()
            
            return [dict(row) for row in rows]
    
    # =========================================================================
    # Maintenance Methods
    # =========================================================================
    
    def cleanup_expired(self) -> Dict[str, int]:
        """Remove expired entries from all tables.
        
        Returns:
            Dict with count of deleted entries per table
        """
        now = datetime.now().isoformat()
        deleted = {}
        
        with self._lock:
            with self._get_connection() as conn:
                # Cleanup screenshots
                result = conn.execute(
                    "DELETE FROM screenshots WHERE expires_at < ?", (now,)
                )
                deleted["screenshots"] = result.rowcount
                
                # Cleanup comparisons
                result = conn.execute(
                    "DELETE FROM comparisons WHERE expires_at < ?", (now,)
                )
                deleted["comparisons"] = result.rowcount
                
                conn.commit()
        
        logger.info(f"Cleanup completed: {deleted}")
        return deleted
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics.
        
        Returns:
            Dict with table counts and sizes
        """
        with self._get_connection() as conn:
            stats = {
                "screenshots_count": conn.execute(
                    "SELECT COUNT(*) as c FROM screenshots"
                ).fetchone()["c"],
                "comparisons_count": conn.execute(
                    "SELECT COUNT(*) as c FROM comparisons"
                ).fetchone()["c"],
                "feedback_count": conn.execute(
                    "SELECT COUNT(*) as c FROM feedback"
                ).fetchone()["c"],
                "db_size_bytes": self.db_path.stat().st_size if self.db_path.exists() else 0,
            }
        
        return stats


# Global database instance
_global_db: Optional[VQADatabase] = None


def get_database(db_path: str = None) -> VQADatabase:
    """Get the global database instance.
    
    Args:
        db_path: Optional path to override default
    
    Returns:
        VQADatabase instance
    """
    global _global_db
    
    if _global_db is None or db_path:
        _global_db = VQADatabase(db_path=db_path)
    
    return _global_db
