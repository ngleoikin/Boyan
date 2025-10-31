"""Configuration helpers for Boyan services.

This module centralises .env loading and provides a small helper to
retrieve settings in a type-safe manner. The loader follows the layout
spelled out in the project README – the .env file lives in the
repository root while services run from ``services/mcp``.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Callable, Optional, TypeVar

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parents[2]
_ENV_FILE = _ROOT / ".env"

if _ENV_FILE.exists():  # Load eagerly so all modules see the same view.
    load_dotenv(_ENV_FILE)

T = TypeVar("T")


def project_root() -> Path:
    """Return the absolute path to the repository root."""

    return _ROOT


def cfg(
    key: str,
    default: Optional[T] = None,
    cast: Optional[Callable[[str], T]] = None,
) -> Optional[T]:
    """Fetch a configuration value from the environment.

    Parameters
    ----------
    key:
        Environment variable name.
    default:
        Default value returned when the key is absent or empty.
    cast:
        Optional callable used to coerce the raw string into a concrete
        type. Errors raised by ``cast`` fall back to ``default`` to keep
        configuration resilient.
    """

    raw = os.getenv(key)
    if raw is None or raw == "":
        return default

    if cast is None:
        return raw  # type: ignore[return-value]

    try:
        return cast(raw)
    except Exception:
        return default


@lru_cache(maxsize=None)
def local_timezone_name() -> str:
    """Return the configured local timezone name (defaults to UTC)."""

    return str(cfg("LOCAL_TZ", "UTC"))
