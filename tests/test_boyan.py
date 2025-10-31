from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict
from unittest.mock import patch

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "services" / "mcp"))

# Prepare isolated storage before importing project modules.
tmp_db = tempfile.NamedTemporaryFile(prefix="boyan-test-", suffix=".db", delete=False)
os.environ["ASSIST_DB"] = tmp_db.name
os.environ["LOCAL_TZ"] = "Asia/Taipei"
os.environ["FACE_COSINE_OK"] = "0.0"

from services.mcp import brain_server, mcp_gateway_face  # noqa: E402
from services.mcp.common import db

_SAMPLE_IMAGE_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAYAAABytg0kAAAAFklEQVR42mNkYGD4z0AEYBxVSFIAAAJbAAH0SDQ8AAAAAElFTkSuQmCC"
)


def _make_image_b64() -> str:
    return _SAMPLE_IMAGE_B64


def test_gateway_ocr_and_face() -> None:
    db.init_db()
    client = TestClient(mcp_gateway_face.app)

    async def fake_probe() -> bool:
        return True

    async def fake_ocr(image_b64: str, mime: str, prompt: str) -> tuple[str, Dict[str, Any]]:
        return "测试 OCR", {"mock": True, "prompt": prompt}

    with patch("services.mcp.mcp_gateway_face._probe_ocr", new=fake_probe), patch(
        "services.mcp.mcp_gateway_face._call_ocr", new=fake_ocr
    ):
        health = client.get("/health").json()
        assert health["ok"] is True
        assert health["ocr"] is True

        data_uri = "data:image/png;base64," + _make_image_b64()
        ocr_resp = client.post("/ocr/from_data_uri", json={"data_uri": data_uri, "prompt": "只输出原文"}).json()
        assert ocr_resp["text"] == "测试 OCR"
        assert ocr_resp["raw"]["mock"] is True

    face_payload = {"name": "tester", "image_b64": _make_image_b64()}
    register = client.post("/face/register", json=face_payload).json()
    assert register["ok"] is True
    assert register["id"].startswith("p_")

    verify = client.post("/face/verify", json={"image_b64": face_payload["image_b64"]}).json()
    assert verify["ok"] is True
    assert verify["candidates"], "candidates should not be empty"
    best = verify["best"]
    assert best is not None and best["name"] == "tester"


def test_brain_workflow() -> None:
    db.init_db()
    client = TestClient(brain_server.app)

    health = client.get("/health").json()
    assert health == {"ok": True}

    note = client.post(
        "/notes/add",
        json={"text": "明早联系王工要合同", "tags": ["autotest"]},
    ).json()
    assert note["ok"] is True

    notes = client.get("/notes/list", params={"q": "合同"}).json()
    assert any(item["id"] == note["id"] for item in notes)

    reminder = client.post(
        "/reminders/parse_create",
        json={"utterance": "10分钟后提醒我喝水", "text": "喝水"},
    ).json()
    assert reminder["ok"] is True

    reminders = client.get("/reminders/list").json()
    assert any(item["id"] == reminder["id"] for item in reminders)

    ics_resp = client.get("/reminders/export/ics")
    assert ics_resp.status_code == 200
    assert "BEGIN:VCALENDAR" in ics_resp.text

    event_payload = {
        "type": "detected",
        "device_id": "dev-1",
        "ts": "2025-10-31T10:00:00Z",
        "cam": "cam-1",
        "topic": "vision/person",
        "data": {"confidence": 0.95},
    }
    action_person = client.post("/edge/event", json=event_payload).json()
    assert action_person["actions"][0]["action"] == "notify"

    event_face = event_payload | {"topic": "vision/face/verified", "data": {"name": "tester", "score": 0.9}}
    action_face = client.post("/edge/event", json=event_face).json()
    assert action_face["actions"][0]["action"] == "greet"

    event_ocr = event_payload | {"topic": "vision/ocr", "data": {"text": "测试 OCR"}}
    action_ocr = client.post("/edge/event", json=event_ocr).json()
    assert action_ocr["actions"][0]["action"] == "store_ocr"
