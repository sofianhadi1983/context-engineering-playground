import sqlite3
import json
import hashlib
import time
import threading
from typing import Optional, Dict, Any
from pathlib import Path


class ResponseCache:
    """
    SQLite-based caching system for API responses with TTL support.
    """

    def __init__(self, db_path: str = "cache.db", ttl: int = 3600):
        """
        Initialize the response cache.

        Args:
            db_path: Path to SQLite database file.
            ttl: Time-to-live in seconds for cached responses.
        """
        self.db_path = Path(db_path)
        self.ttl = ttl
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0
        self._init_database()

    def _init_database(self) -> None:
        """
        Initialize the SQLite database and create tables with proper schema.

        Raises:
            Exception: If database initialization fails.
        """
        with self._lock:
            conn = None
            try:
                conn = sqlite3.connect(str(self.db_path))
                cursor = conn.cursor()

                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS cache (
                        cache_key TEXT PRIMARY KEY,
                        prompt TEXT NOT NULL,
                        model TEXT NOT NULL,
                        params TEXT NOT NULL,
                        response TEXT NOT NULL,
                        created_at REAL NOT NULL,
                        accessed_at REAL NOT NULL,
                        access_count INTEGER DEFAULT 1
                    )
                """)

                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_created_at ON cache(created_at)
                """)

                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_prompt ON cache(prompt)
                """)

                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_model ON cache(model)
                """)

                conn.commit()

            except sqlite3.Error as e:
                raise Exception(f"Failed to initialize cache database: {e}")
            finally:
                if conn:
                    conn.close()

    def _generate_key(self, prompt: str, model: str, params: Dict[str, Any]) -> str:
        """
        Generate a cache key from prompt, model, and parameters.

        Args:
            prompt: The prompt text.
            model: The model name.
            params: Dictionary of parameters.

        Returns:
            SHA256 hash as cache key.

        Raises:
            ValueError: If inputs are invalid.
        """
        if not prompt or not isinstance(prompt, str):
            raise ValueError("Prompt must be a non-empty string")
        if not model or not isinstance(model, str):
            raise ValueError("Model must be a non-empty string")
        if not isinstance(params, dict):
            raise ValueError("Params must be a dictionary")

        try:
            params_sorted = json.dumps(params, sort_keys=True)
            key_string = f"{prompt}|{model}|{params_sorted}"
            return hashlib.sha256(key_string.encode()).hexdigest()
        except (TypeError, ValueError) as e:
            raise ValueError(f"Failed to generate cache key: {e}")

    def get(
        self,
        prompt: str,
        model: str,
        params: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve a cached response if available and not expired.

        Args:
            prompt: The prompt text.
            model: The model name.
            params: Dictionary of parameters.

        Returns:
            Cached response dictionary or None if not found/expired.

        Raises:
            Exception: If database operation fails.
        """
        try:
            cache_key = self._generate_key(prompt, model, params)
        except ValueError:
            self._misses += 1
            return None

        with self._lock:
            conn = None
            try:
                conn = sqlite3.connect(str(self.db_path))
                cursor = conn.cursor()

                cursor.execute(
                    "SELECT response, created_at, access_count FROM cache WHERE cache_key = ?",
                    (cache_key,)
                )

                result = cursor.fetchone()

                if result is None:
                    self._misses += 1
                    return None

                response_json, created_at, access_count = result
                current_time = time.time()

                if current_time - created_at > self.ttl:
                    cursor.execute("DELETE FROM cache WHERE cache_key = ?", (cache_key,))
                    conn.commit()
                    self._misses += 1
                    return None

                cursor.execute(
                    "UPDATE cache SET accessed_at = ?, access_count = ? WHERE cache_key = ?",
                    (current_time, access_count + 1, cache_key)
                )
                conn.commit()

                self._hits += 1
                return json.loads(response_json)

            except sqlite3.Error as e:
                raise Exception(f"Cache get operation failed: {e}")
            except json.JSONDecodeError as e:
                raise Exception(f"Failed to decode cached response: {e}")
            finally:
                if conn:
                    conn.close()

    def set(
        self,
        prompt: str,
        model: str,
        params: Dict[str, Any],
        response: Dict[str, Any]
    ) -> None:
        """
        Store a response in the cache.

        Args:
            prompt: The prompt text.
            model: The model name.
            params: Dictionary of parameters.
            response: Response dictionary to cache.

        Raises:
            ValueError: If inputs are invalid.
            Exception: If database operation fails.
        """
        if not isinstance(response, dict):
            raise ValueError("Response must be a dictionary")

        cache_key = self._generate_key(prompt, model, params)
        current_time = time.time()

        try:
            params_json = json.dumps(params, sort_keys=True)
            response_json = json.dumps(response)
        except (TypeError, ValueError) as e:
            raise ValueError(f"Failed to serialize data: {e}")

        with self._lock:
            conn = None
            try:
                conn = sqlite3.connect(str(self.db_path))
                cursor = conn.cursor()

                cursor.execute("""
                    INSERT OR REPLACE INTO cache
                    (cache_key, prompt, model, params, response, created_at, accessed_at, access_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?, 1)
                """, (cache_key, prompt, model, params_json, response_json, current_time, current_time))

                conn.commit()

            except sqlite3.Error as e:
                raise Exception(f"Cache set operation failed: {e}")
            finally:
                if conn:
                    conn.close()

    def invalidate(self, prompt: str) -> int:
        """
        Remove specific cached responses matching the prompt.

        Args:
            prompt: The prompt text to invalidate.

        Returns:
            Number of entries removed.

        Raises:
            Exception: If database operation fails.
        """
        if not prompt or not isinstance(prompt, str):
            raise ValueError("Prompt must be a non-empty string")

        with self._lock:
            conn = None
            try:
                conn = sqlite3.connect(str(self.db_path))
                cursor = conn.cursor()

                cursor.execute("DELETE FROM cache WHERE prompt = ?", (prompt,))
                deleted_count = cursor.rowcount

                conn.commit()
                return deleted_count

            except sqlite3.Error as e:
                raise Exception(f"Cache invalidate operation failed: {e}")
            finally:
                if conn:
                    conn.close()

    def clear_expired(self) -> int:
        """
        Remove all expired cache entries based on TTL.

        Returns:
            Number of expired entries removed.

        Raises:
            Exception: If database operation fails.
        """
        current_time = time.time()
        expiry_threshold = current_time - self.ttl

        with self._lock:
            conn = None
            try:
                conn = sqlite3.connect(str(self.db_path))
                cursor = conn.cursor()

                cursor.execute(
                    "DELETE FROM cache WHERE created_at < ?",
                    (expiry_threshold,)
                )
                deleted_count = cursor.rowcount

                conn.commit()
                return deleted_count

            except sqlite3.Error as e:
                raise Exception(f"Cache clear_expired operation failed: {e}")
            finally:
                if conn:
                    conn.close()

    def clear_all(self) -> int:
        """
        Clear the entire cache and reset statistics.

        Returns:
            Number of entries removed.

        Raises:
            Exception: If database operation fails.
        """
        with self._lock:
            conn = None
            try:
                conn = sqlite3.connect(str(self.db_path))
                cursor = conn.cursor()

                cursor.execute("SELECT COUNT(*) FROM cache")
                count = cursor.fetchone()[0]

                cursor.execute("DELETE FROM cache")

                conn.commit()
                conn.close()

                conn = sqlite3.connect(str(self.db_path))
                conn.isolation_level = None
                cursor = conn.cursor()
                cursor.execute("VACUUM")

                self._hits = 0
                self._misses = 0

                return count

            except sqlite3.Error as e:
                raise Exception(f"Cache clear_all operation failed: {e}")
            finally:
                if conn:
                    conn.close()

    def get_stats(self) -> Dict[str, Any]:
        """
        Return comprehensive cache statistics.

        Returns:
            Dictionary with cache statistics including entries, hits, misses, and database size.

        Raises:
            Exception: If database operation fails.
        """
        with self._lock:
            conn = None
            try:
                conn = sqlite3.connect(str(self.db_path))
                cursor = conn.cursor()

                cursor.execute("SELECT COUNT(*) FROM cache")
                total_entries = cursor.fetchone()[0]

                cursor.execute("SELECT SUM(access_count) FROM cache")
                total_accesses_result = cursor.fetchone()[0]
                total_accesses = total_accesses_result if total_accesses_result else 0

                current_time = time.time()
                expiry_threshold = current_time - self.ttl

                cursor.execute(
                    "SELECT COUNT(*) FROM cache WHERE created_at >= ?",
                    (expiry_threshold,)
                )
                valid_entries = cursor.fetchone()[0]

                cursor.execute(
                    "SELECT COUNT(*) FROM cache WHERE created_at < ?",
                    (expiry_threshold,)
                )
                expired_entries = cursor.fetchone()[0]

                cursor.execute("SELECT AVG(access_count) FROM cache")
                avg_access_result = cursor.fetchone()[0]
                avg_accesses = round(avg_access_result, 2) if avg_access_result else 0

                total_requests = self._hits + self._misses
                hit_rate = (self._hits / total_requests * 100) if total_requests > 0 else 0

                db_size = self.db_path.stat().st_size if self.db_path.exists() else 0

                return {
                    "total_entries": total_entries,
                    "valid_entries": valid_entries,
                    "expired_entries": expired_entries,
                    "total_accesses": total_accesses,
                    "avg_accesses_per_entry": avg_accesses,
                    "cache_hits": self._hits,
                    "cache_misses": self._misses,
                    "total_requests": total_requests,
                    "hit_rate_percent": round(hit_rate, 2),
                    "db_size_bytes": db_size,
                    "db_size_kb": round(db_size / 1024, 2),
                    "ttl_seconds": self.ttl,
                }

            except sqlite3.Error as e:
                raise Exception(f"Failed to retrieve cache statistics: {e}")
            except OSError as e:
                raise Exception(f"Failed to get database file size: {e}")
            finally:
                if conn:
                    conn.close()

    def __enter__(self) -> "ResponseCache":
        """
        Context manager entry.

        Returns:
            Self for use in with statement.
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """
        Context manager exit with automatic cleanup of expired entries.

        Args:
            exc_type: Exception type if an exception occurred.
            exc_val: Exception value if an exception occurred.
            exc_tb: Exception traceback if an exception occurred.

        Returns:
            False to propagate any exception that occurred.
        """
        try:
            self.clear_expired()
        except Exception:
            pass
        return False

    def __del__(self):
        """
        Destructor to clean up expired entries on object deletion.
        """
        try:
            if hasattr(self, 'db_path') and self.db_path.exists():
                self.clear_expired()
        except Exception:
            pass
