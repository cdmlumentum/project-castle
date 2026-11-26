"""SQLite cache for raw isolated-mode candidates and coefficient results."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from zipfile import BadZipFile
import time
from contextlib import closing
from pathlib import Path
from typing import Any
from .checkpoints import encode, decode


def stable_key(payload: dict[str, Any]) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


class SQLiteCache:
    def __init__(self, path: str | Path, *, progress=None, max_bytes=0):
        self.path = Path(path)
        self.progress = progress
        self.max_bytes = max_bytes
        self.stats: dict[str, dict[str, int]] = {}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute(
                """
                CREATE TABLE IF NOT EXISTS cache_v2 (
                    cache_key TEXT PRIMARY KEY,
                    payload BLOB NOT NULL,
                    sha256 TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    accessed REAL NOT NULL,
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
                "SELECT payload,sha256 FROM cache_v2 WHERE cache_key = ?", (key,)
            ).fetchone()
            if row is not None:
                db.execute('UPDATE cache_v2 SET accessed=? WHERE cache_key=?', (time.time(), key))
        result = None
        if row is not None:
            try:
                if hashlib.sha256(row[0]).hexdigest() != row[1]:
                    raise ValueError('Cache checksum mismatch')
                result = decode(row[0])
            except (ValueError, KeyError, TypeError, OSError, BadZipFile):
                self.delete(key)
                row = None
        hit = row is not None
        stats = self._stats_for(kind)
        stats["lookups"] += 1
        stats["hits" if hit else "misses"] += 1
        if self.progress:
            self.progress.cache_event(kind, "hit" if hit else "miss")
        if self.progress:
            self.progress.record_timing("cache_lookup", time.perf_counter() - started)
        return result

    def put(self, key: str, value: Any, *, kind: str = "other") -> None:
        started = time.perf_counter()
        payload = encode(value)
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute(
                """
                INSERT INTO cache_v2(cache_key, payload,sha256,kind,accessed)
                VALUES(?, ?, ?, ?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET payload=excluded.payload,sha256=excluded.sha256,kind=excluded.kind,accessed=excluded.accessed
                """,
                (key, payload, hashlib.sha256(payload).hexdigest(), kind, time.time()),
            )
        if self.max_bytes:
            self.prune(self.max_bytes)
        self._stats_for(kind)["writes"] += 1
        if self.progress:
            self.progress.cache_event(kind, "write")
            self.progress.record_timing("cache_write", time.perf_counter() - started)

    def delete(self, key: str) -> None:
        """Remove an unusable entry without disturbing other cached solves."""
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("DELETE FROM cache_v2 WHERE cache_key = ?", (key,))

    def count(self) -> int:
        with closing(sqlite3.connect(self.path)) as db, db:
            return int(db.execute("SELECT COUNT(*) FROM cache_v2").fetchone()[0])

    def inspect(self):
        with closing(sqlite3.connect(self.path)) as db:
            rows = db.execute('SELECT kind,COUNT(*),SUM(LENGTH(payload)) FROM cache_v2 GROUP BY kind').fetchall()
        return {'path': str(self.path), 'format': 'checksummed_npz_v2',
                'entries': sum(r[1] for r in rows), 'payload_bytes': sum(r[2] for r in rows), 'disk_bytes': self.path.stat().st_size,
                'by_kind': {r[0]: {'entries': r[1], 'bytes': r[2]} for r in rows}}

    def prune(self, max_bytes):
        if max_bytes < 0:
            raise ValueError('Cache byte limit must be nonnegative')
        removed = 0
        with closing(sqlite3.connect(self.path)) as db, db:
            rows = db.execute('SELECT cache_key,LENGTH(payload) FROM cache_v2 ORDER BY accessed DESC').fetchall()
            retained = 0
            for key, size in rows:
                if retained+size <= max_bytes:
                    retained += size
                else:
                    db.execute('DELETE FROM cache_v2 WHERE cache_key=?', (key,))
                    removed += 1
        return removed

    def vacuum(self):
        """Explicit maintenance operation; never run on the optical hot path."""
        with closing(sqlite3.connect(self.path)) as db:
            db.execute('VACUUM')
