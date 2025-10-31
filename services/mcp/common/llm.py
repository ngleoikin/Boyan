"""Helpers for interacting with large language model endpoints."""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, Iterable, List, Optional

import httpx

from . import config

LOGGER = logging.getLogger(__name__)


def build_text_message(role: str, text: str) -> Dict[str, Any]:
    """Create a DeepSeek-compatible text message payload."""

    return {"role": role, "content": [{"type": "input_text", "text": text}]}


async def call_responses(
    messages: Iterable[Dict[str, Any]],
    *,
    model: Optional[str] = None,
    temperature: float = 0.2,
) -> Dict[str, Any]:
    """Send a conversation to the configured responses endpoint."""

    base_url = str(config.cfg("OPENAI_BASE", "https://api.deepseek.com")).rstrip("/")
    api_key = config.cfg("OPENAI_API_KEY")
    model_id = model or str(config.cfg("MODEL_ID", "deepseek-chat"))
    url = f"{base_url}/v1/responses"

    payload = {"model": model_id, "input": list(messages), "temperature": temperature}
    headers = {"content-type": "application/json"}
    if api_key:
        headers["authorization"] = f"Bearer {api_key}"

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
    return resp.json()


def extract_first_text(payload: Any) -> str:
    """Traverse a DeepSeek responses payload and return the first text string."""

    candidates: List[str] = []

    def _collect(value: Any) -> None:
        if isinstance(value, str):
            stripped = value.strip()
            if stripped:
                candidates.append(stripped)
        elif isinstance(value, list):
            for item in value:
                _collect(item)
        elif isinstance(value, dict):
            for key in ("text", "output_text", "content", "message", "messages"):
                if key in value:
                    _collect(value[key])
            if not value:
                return

    _collect(payload)
    if candidates:
        return candidates[0]
    raise ValueError("无法从模型响应中解析文本")


def format_event_for_prompt(event: Dict[str, Any]) -> str:
    """Create a concise JSON representation for prompt injection."""

    return json.dumps(event, ensure_ascii=False, indent=2)
