"""Pydantic schemas shared between the services."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class EdgeEvent(BaseModel):
    type: str = Field(..., description="Event type from the device")
    device_id: str
    ts: datetime
    cam: str
    topic: str
    data: Dict[str, Any] = Field(default_factory=dict)

    model_config = {
        "populate_by_name": True,
        "json_encoders": {datetime: lambda d: d.isoformat().replace("+00:00", "Z")},
    }


class NoteIn(BaseModel):
    text: str
    tags: Optional[List[str]] = None


class NoteOut(BaseModel):
    id: int
    text: str
    tags: List[str]
    created_ts: str


class ReminderIn(BaseModel):
    utterance: str
    text: Optional[str] = None


class ReminderOut(BaseModel):
    id: int
    text: str
    due_ts: str
    created_ts: str


class OCRDataURIRequest(BaseModel):
    data_uri: str
    prompt: str = "只输出原文"


class OCROut(BaseModel):
    text: str
    raw: Dict[str, Any]


class FaceRegisterRequest(BaseModel):
    name: str
    image_b64: str


class FaceRegisterResponse(BaseModel):
    ok: bool
    id: str
    appended: bool


class FaceVerifyRequest(BaseModel):
    image_b64: str
    top_k: int = 3
    threshold: Optional[float] = None


class FaceCandidate(BaseModel):
    person_id: str
    name: str
    score: float


class FaceVerifyResponse(BaseModel):
    ok: bool
    candidates: List[FaceCandidate]
    best: Optional[FaceCandidate] = None
