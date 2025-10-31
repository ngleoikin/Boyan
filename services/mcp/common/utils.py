"""Utility helpers shared across Boyan services."""
from __future__ import annotations

import base64
import io
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Iterable, Mapping, Optional, Sequence, Tuple
from uuid import uuid4

import dateparser
import numpy as np
from dateutil import tz

from .config import local_timezone_name

LOGGER = logging.getLogger(__name__)

_REL_MINUTES = re.compile(r"(\d+)\s*分钟后")
_REL_HOURS = re.compile(r"(\d+)\s*小时后")
_ABS_TOMORROW = re.compile(r"明天\s*(\d{1,2})[:：](\d{2})")
_ABS_HM = re.compile(r"(?<!\d)(\d{1,2})[:：](\d{2})(?!\d)")
_ABS_YMDHM = re.compile(r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})[ T](\d{1,2})[:：](\d{2})")


def _local_tz() -> tz.tzfile:
    name = local_timezone_name()
    return tz.gettz(name) or tz.UTC


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def local_now() -> datetime:
    return datetime.now(_local_tz())


def ensure_tzaware(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=_local_tz())
    return dt


def parse_natural_datetime(text: str, reference: Optional[datetime] = None) -> datetime:
    """Parse Chinese (and general natural language) time expressions."""

    if reference is None:
        reference = local_now()

    settings = {
        "PREFER_DATES_FROM": "future",
        "RETURN_AS_TIMEZONE_AWARE": True,
        "RELATIVE_BASE": ensure_tzaware(reference),
        "TIMEZONE": local_timezone_name(),
        "TO_TIMEZONE": "UTC",
    }

    parsed = dateparser.parse(text, settings=settings)
    if parsed:
        return parsed.astimezone(timezone.utc)

    lower = text.lower()

    if m := _REL_MINUTES.search(lower):
        delta = timedelta(minutes=int(m.group(1)))
        return (reference + delta).astimezone(timezone.utc)

    if m := _REL_HOURS.search(lower):
        delta = timedelta(hours=int(m.group(1)))
        return (reference + delta).astimezone(timezone.utc)

    if m := _ABS_TOMORROW.search(text):
        hh, mm = int(m.group(1)), int(m.group(2))
        tomorrow = ensure_tzaware(reference).replace(hour=hh, minute=mm, second=0, microsecond=0) + timedelta(days=1)
        return tomorrow.astimezone(timezone.utc)

    if m := _ABS_YMDHM.search(text):
        y, M, d, hh, mm = map(int, m.groups())
        local_dt = datetime(y, M, d, hh, mm, tzinfo=_local_tz())
        return local_dt.astimezone(timezone.utc)

    if m := _ABS_HM.search(text):
        hh, mm = map(int, m.groups())
        candidate = ensure_tzaware(reference).replace(hour=hh, minute=mm, second=0, microsecond=0)
        if candidate <= ensure_tzaware(reference):
            candidate += timedelta(days=1)
        return candidate.astimezone(timezone.utc)

    return (reference + timedelta(minutes=10)).astimezone(timezone.utc)


def isoformat_z(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def generate_uid(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:10]}"


def parse_data_uri(uri: str) -> Tuple[str, bytes]:
    if not uri.startswith("data:"):
        raise ValueError("Not a data URI")
    header, payload = uri.split(",", 1)
    mime = header.split(";")[0][5:] or "application/octet-stream"
    if ";base64" in header:
        return mime, base64.b64decode(payload)
    return mime, payload.encode()


def image_bytes_to_array(data: bytes) -> np.ndarray:
    try:
        import cv2  # type: ignore
    except Exception:
        cv2 = None

    arr = np.frombuffer(data, dtype=np.uint8)
    if cv2 is not None:
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is not None:
            return img

    try:
        from PIL import Image
    except Exception as exc:  # pragma: no cover - depends on pillow
        raise ValueError("无法解析图像数据") from exc

    with Image.open(io.BytesIO(data)) as im:
        rgb = im.convert("RGB")
        arr = np.array(rgb)
        # Convert to BGR to match OpenCV conventions used elsewhere.
        return arr[:, :, ::-1]


def encode_vector(vec: np.ndarray) -> bytes:
    return np.asarray(vec, dtype=np.float32).tobytes()


def decode_vector(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def generate_ics(reminders: Sequence[Mapping[str, str]]) -> str:
    lines = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//Boyan//Brain//CN",
    ]
    for reminder in reminders:
        due = reminder["due_ts"].replace("-", "").replace(":", "")
        due = due.replace("Z", "Z").replace("T", "T")
        summary = (
            reminder["text"]
            .replace("\\", "\\\\")
            .replace(",", "\\,")
            .replace(";", "\\;")
            .replace("\n", "\\n")
        )
        uid = reminder.get("id") or generate_uid("reminder")
        lines.extend(
            [
                "BEGIN:VEVENT",
                f"UID:{uid}",
                f"DTSTART:{due}",
                f"SUMMARY:{summary}",
                "END:VEVENT",
            ]
        )
    lines.append("END:VCALENDAR")
    return "\r\n".join(lines) + "\r\n"


def decode_base64_image(data: str) -> np.ndarray:
    if data.startswith("data:"):
        _, payload = data.split(",", 1)
    else:
        payload = data
    return image_bytes_to_array(base64.b64decode(payload))
