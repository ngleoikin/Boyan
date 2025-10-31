"""SQLite helpers used by both the gateway and the brain services."""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from .config import cfg, project_root

_DEFAULT_DB = project_root() / "services" / "mcp" / "storage" / "assist.db"

_DB_PATH = Path(str(cfg("ASSIST_DB", str(_DEFAULT_DB))))
_DB_PATH = _DB_PATH.expanduser()
_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

_SCHEMA = [
    """
    CREATE TABLE IF NOT EXISTS notes(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        text TEXT NOT NULL,
        tags TEXT,
        created_ts TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS reminders(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        text TEXT NOT NULL,
        due_ts TEXT NOT NULL,
        created_ts TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS ocr_logs(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        device_id TEXT,
        cam TEXT,
        text TEXT,
        lang TEXT,
        ts TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS faces(
        id TEXT PRIMARY KEY,
        name TEXT,
        vec BLOB,
        created_ts TEXT NOT NULL
    )
    """,
]


def connect() -> sqlite3.Connection:
    """Return a configured SQLite connection."""

    conn = sqlite3.connect(_DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


@contextmanager
def get_conn() -> Iterator[sqlite3.Connection]:
    """Context manager yielding a SQLite connection."""

    conn = connect()
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    """Ensure that all required tables are present."""

    with get_conn() as conn:
        cur = conn.cursor()
        for stmt in _SCHEMA:
            cur.execute(stmt)
