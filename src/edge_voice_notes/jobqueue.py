"""Persistent job queue on SQLite (WAL, synchronous=FULL).

The queue only records *which* recordings are pending, done or failed and how
many times each was attempted. Fine-grained progress (which chunks and sections
are finished) lives in atomically written files next to the recording, so the
worker can resume a 60-minute transcription at the last finished chunk.

The attempt counter is incremented and committed *before* the work starts. A
recording that crashes the device every time (OOM, driver bug) is therefore
parked as failed after `max_attempts` boots instead of causing a reboot loop.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id         TEXT PRIMARY KEY,
    state      TEXT NOT NULL CHECK (state IN ('pending', 'done', 'failed')),
    attempts   INTEGER NOT NULL DEFAULT 0,
    last_error TEXT,
    created    REAL NOT NULL,
    updated    REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS jobs_pending ON jobs (state, created);
"""


@dataclass(frozen=True)
class Job:
    id: str
    state: str
    attempts: int
    last_error: str | None
    created: float
    updated: float


class JobQueue:
    def __init__(self, path: Path) -> None:
        self.db = sqlite3.connect(str(path), isolation_level=None, timeout=30)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript(SCHEMA)

    def close(self) -> None:
        self.db.close()

    def _row(self, r) -> Job | None:
        return Job(*r) if r else None

    def add(self, job_id: str) -> bool:
        now = time.time()
        cur = self.db.execute(
            "INSERT OR IGNORE INTO jobs (id, state, attempts, created, updated) VALUES (?, 'pending', 0, ?, ?)",
            (job_id, now, now))
        return cur.rowcount == 1

    def get(self, job_id: str) -> Job | None:
        return self._row(self.db.execute(
            "SELECT id, state, attempts, last_error, created, updated FROM jobs WHERE id = ?",
            (job_id,)).fetchone())

    def all(self) -> list[Job]:
        return [Job(*r) for r in self.db.execute(
            "SELECT id, state, attempts, last_error, created, updated FROM jobs ORDER BY created, id")]

    def next_pending(self) -> Job | None:
        return self._row(self.db.execute(
            "SELECT id, state, attempts, last_error, created, updated FROM jobs "
            "WHERE state = 'pending' ORDER BY created, id LIMIT 1").fetchone())

    def begin_attempt(self, job_id: str) -> int:
        self.db.execute("UPDATE jobs SET attempts = attempts + 1, updated = ? WHERE id = ?",
                        (time.time(), job_id))
        return self.get(job_id).attempts

    def progress(self, job_id: str) -> None:
        """A checkpoint was written: this attempt is not a crash loop, reset the counter."""
        self.db.execute("UPDATE jobs SET attempts = 0, updated = ? WHERE id = ?", (time.time(), job_id))

    def set_error(self, job_id: str, error: str) -> None:
        self.db.execute("UPDATE jobs SET last_error = ?, updated = ? WHERE id = ?",
                        (error[:2000], time.time(), job_id))

    def mark_done(self, job_id: str) -> None:
        self.db.execute("UPDATE jobs SET state = 'done', last_error = NULL, updated = ? WHERE id = ?",
                        (time.time(), job_id))

    def mark_failed(self, job_id: str, error: str) -> None:
        self.db.execute("UPDATE jobs SET state = 'failed', last_error = ?, updated = ? WHERE id = ?",
                        (error[:2000], time.time(), job_id))

    def retry(self, job_id: str) -> bool:
        cur = self.db.execute(
            "UPDATE jobs SET state = 'pending', attempts = 0, last_error = NULL, updated = ? "
            "WHERE id = ? AND state = 'failed'", (time.time(), job_id))
        return cur.rowcount == 1
