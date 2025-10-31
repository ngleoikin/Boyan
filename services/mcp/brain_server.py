from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Response

from common import db, schemas, utils

app = FastAPI(title="Boyan Brain", version="1.0.0")


@app.on_event("startup")
def _startup() -> None:
    db.init_db()


@app.get("/health")
def health() -> Dict[str, bool]:
    return {"ok": True}


def _insert_note(text: str, tags: Optional[List[str]]) -> Dict[str, Any]:
    created_ts = utils.isoformat_z(utils.utc_now())
    tags_str = json.dumps(tags or [], ensure_ascii=False)
    with db.get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO notes(text, tags, created_ts) VALUES(?,?,?)",
            (text, tags_str, created_ts),
        )
        note_id = cur.lastrowid
    return {"ok": True, "id": note_id, "created_ts": created_ts}


@app.post("/notes/add")
def notes_add(payload: schemas.NoteIn) -> Dict[str, Any]:
    return _insert_note(payload.text, payload.tags)


@app.get("/notes/list", response_model=List[schemas.NoteOut])
def notes_list(q: Optional[str] = None, limit: int = 50) -> List[schemas.NoteOut]:
    limit = max(1, min(limit, 200))
    sql = "SELECT id, text, tags, created_ts FROM notes"
    params: List[Any] = []
    if q:
        sql += " WHERE text LIKE ? OR tags LIKE ?"
        wildcard = f"%{q}%"
        params.extend([wildcard, wildcard])
    sql += " ORDER BY created_ts DESC LIMIT ?"
    params.append(limit)

    items: List[schemas.NoteOut] = []
    with db.get_conn() as conn:
        rows = conn.execute(sql, tuple(params)).fetchall()
        for row in rows:
            tags = json.loads(row["tags"] or "[]")
            items.append(
                schemas.NoteOut(
                    id=row["id"],
                    text=row["text"],
                    tags=tags,
                    created_ts=row["created_ts"],
                )
            )
    return items


@app.post("/reminders/parse_create")
def reminders_parse_create(payload: schemas.ReminderIn) -> Dict[str, Any]:
    due_dt = utils.parse_natural_datetime(payload.utterance)
    due_ts = utils.isoformat_z(due_dt)
    text = payload.text.strip() if payload.text else payload.utterance.strip()
    created_ts = utils.isoformat_z(utils.utc_now())

    with db.get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO reminders(text, due_ts, created_ts) VALUES(?,?,?)",
            (text, due_ts, created_ts),
        )
        reminder_id = cur.lastrowid

    return {"ok": True, "id": reminder_id, "text": text, "due_ts": due_ts}


@app.get("/reminders/list", response_model=List[schemas.ReminderOut])
def reminders_list(limit: int = 100) -> List[schemas.ReminderOut]:
    limit = max(1, min(limit, 300))
    with db.get_conn() as conn:
        rows = conn.execute(
            "SELECT id, text, due_ts, created_ts FROM reminders ORDER BY due_ts ASC LIMIT ?",
            (limit,),
        ).fetchall()
    return [
        schemas.ReminderOut(
            id=row["id"],
            text=row["text"],
            due_ts=row["due_ts"],
            created_ts=row["created_ts"],
        )
        for row in rows
    ]


@app.get("/reminders/export/ics")
def reminders_export_ics() -> Response:
    with db.get_conn() as conn:
        rows = conn.execute("SELECT id, text, due_ts FROM reminders ORDER BY due_ts ASC").fetchall()
    reminders = [
        {
            "id": f"rem_{row['id']}",
            "text": row["text"],
            "due_ts": row["due_ts"],
        }
        for row in rows
    ]
    ics = utils.generate_ics(reminders)
    return Response(content=ics, media_type="text/calendar; charset=utf-8")


@app.post("/edge/event")
def edge_event(event: schemas.EdgeEvent) -> Dict[str, Any]:
    actions: List[Dict[str, Any]] = []
    skipped = True

    ts_value = event.ts
    if isinstance(ts_value, datetime):
        ts = utils.isoformat_z(ts_value)
    else:
        ts = str(ts_value)

    if event.topic == "vision/person":
        confidence = float(event.data.get("confidence", 0))
        if confidence >= 0.8:
            actions.append(
                {
                    "action": "notify",
                    "title": "检测到人员",
                    "text": f"cam={event.cam}, confidence={confidence:.2f}",
                    "device_id": event.device_id,
                    "ts": ts,
                }
            )
            skipped = False
    elif event.topic == "vision/face/verified":
        name = str(event.data.get("name", "")) or "访客"
        score = event.data.get("score")
        text = f"欢迎 {name}"
        if score is not None:
            text += f" (score={score})"
        actions.append(
            {
                "action": "greet",
                "title": "人脸识别",
                "text": text,
                "device_id": event.device_id,
                "ts": ts,
            }
        )
        skipped = False
    elif event.topic == "vision/ocr":
        text = str(event.data.get("text", ""))
        lang = event.data.get("lang")
        if text.strip():
            actions.append(
                {
                    "action": "store_ocr",
                    "title": "OCR 文本入库",
                    "text": text,
                    "device_id": event.device_id,
                    "ts": ts,
                }
            )
            skipped = False
            with db.get_conn() as conn:
                conn.execute(
                    "INSERT INTO ocr_logs(device_id, cam, text, lang, ts) VALUES(?,?,?,?,?)",
                    (
                        event.device_id,
                        event.cam,
                        text,
                        lang,
                        ts,
                    ),
                )
    return {"ok": True, "skipped": skipped, "actions": actions}
