"""SQLite cache for raw isolated-mode candidates and coefficient results."""

from __future__ import annotations

import hashlib
import json
import pickle
import sqlite3
import time
from contextlib import closing
from pathlib import Path
from typing import Any


def stable_key(payload: dict[str, Any]) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


class SQLiteCache:
    def __init__(self, path: str | Path, *, progress=None):
        self.path = Path(path)
        self.progress = progress
        self.stats: dict[str, dict[str, int]] = {}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute(
                """
                CREATE TABLE IF NOT EXISTS coefficient_cache (
                    cache_key TEXT PRIMARY KEY,
                    payload BLOB NOT NULL,
                    created_utc TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

    def _stats_for(self, kind: str) -> dict[str, int]:
        return self.stats.setdefault(
            kind, {"lookups": 0, "hits": 0, "misses": 0, "writes": 0}
        )

    def get(self, key: str, *, kind: str = "other") -> Any | None:
        started = time.perf_counter()
        with closing(sqlite3.connect(self.path)) as db, db:
            row = db.execute(
                "SELECT payload FROM coefficient_cache WHERE cache_key = ?", (key,)
            ).fetchone()
        hit = row is not None
        stats = self._stats_for(kind)
        stats["lookups"] += 1
        stats["hits" if hit else "misses"] += 1
        if self.progress:
            self.progress.cache_event(kind, "hit" if hit else "miss")
        result = None if row is None else pickle.loads(row[0])
        if self.progress:
            self.progress.record_timing("cache_lookup", time.perf_counter() - started)
        return result

    def put(self, key: str, value: Any, *, kind: str = "other") -> None:
        started = time.perf_counter()
        payload = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute(
                """
                INSERT INTO coefficient_cache(cache_key, payload)
                VALUES(?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET payload=excluded.payload
                """,
                (key, payload),
            )
        self._stats_for(kind)["writes"] += 1
        if self.progress:
            self.progress.cache_event(kind, "write")
            self.progress.record_timing("cache_write", time.perf_counter() - started)

    def delete(self, key: str) -> None:
        """Remove an unusable entry without disturbing other cached solves."""
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("DELETE FROM coefficient_cache WHERE cache_key = ?", (key,))

    def count(self) -> int:
        with closing(sqlite3.connect(self.path)) as db, db:
            return int(db.execute("SELECT COUNT(*) FROM coefficient_cache").fetchone()[0])
