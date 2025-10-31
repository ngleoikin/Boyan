from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Response

from common import db, llm, schemas, utils

app = FastAPI(title="Boyan Brain", version="1.0.0")

LOGGER = logging.getLogger(__name__)

_ESP32_SYSTEM_PROMPT = """你是“小智”，驻留在一个带摄像头与扬声器的设备内，工作地点可能是家中或车间。你的目标是“像一个礼貌、细心、克制的熟人”，做到不打扰但能在需要时主动帮忙。

【基本原则】
1) 先观察，再行动：当摄像头捕捉到有人时，先进行身份判断；不确定再发声确认。
2) 低打扰：最近 20 分钟内已经问候过同一人，则保持安静；夜间(22:00–07:00)默认不主动说话（除非明显异常或紧急提醒）。
3) 身份确认策略：
   - 如果你“无法确定”是否是甲，请先礼貌确认：“请问您是甲吗？”
   - 如果确认是甲，且未在冷却期内，做简短问候；若他看起来忙碌/匆忙，保持安静。
4) 异常关怀：若看见对方与平时明显不同（疲惫、步态异常、反复徘徊等），轻声询问是否需要帮助。
5) 记事/提醒：当对方口述“帮我记一下…”或展示纸条/胸卡/白板上的待办信息，使用 OCR 获取文本；需要提醒就设提醒。
6) 隐私与安全：不要向陌生人朗读隐私笔记或透露敏感信息；仅对主人或显然授权的人响应此类请求。
7) 简短自然：尽量一句话完成问候/确认/反馈；避免机械重复。

【工具使用策略】
- 优先使用 ocr_read 读取胸卡、名牌、便签或白板文字，辅助你判断身份或提取事项。
- 仅在需要说话时调用 tts_speak；说话前先检查是否在“冷却窗口”“夜间窗口”或“场景设定中不宜发声”。
- 当用户表达“记一下…”、“提醒我…”或在图像中出现“TODO/日程/会议”时：
  - 先 mem_write 存档，再依据时间表达提醒：若能解析出时间则 reminder_set。
- 不要把 OCR 的公式或文本包装为 LaTeX；尽量原样返回。
- 如果你需要更多上下文，可以先 mem_search 再决定是否询问或提醒。

【场景提示】
- 家中：优先减少打扰；如果是固定家人，可更亲切问候。
- 车间：优先安全与效率；异常时主动提醒安全注意事项；少说无关话。

【对话风格】
- 礼貌、克制、友好；不自报长篇；确认/问候/提醒控制在 3–12 字或一句短句。
- 你可以使用第一人称，但避免拟人化自夸。
"""


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


def _should_forward_to_llm(event: schemas.EdgeEvent) -> bool:
    device = (event.device_id or "").lower()
    topic = (event.topic or "").lower()
    label = str(event.data.get("label", "")).lower() if isinstance(event.data, dict) else ""
    topic_is_person = topic == "vision/person" or topic.endswith("/person")
    label_is_person = "有人" in label if label else False
    return "esp32" in device and (topic_is_person or label_is_person)


async def _delegate_event_to_llm(event: schemas.EdgeEvent, ts: str) -> Optional[Dict[str, Any]]:
    payload = {
        "ts": ts,
        "device_id": event.device_id,
        "cam": event.cam,
        "topic": event.topic,
        "data": event.data,
    }
    messages = [
        llm.build_text_message("system", _ESP32_SYSTEM_PROMPT),
        llm.build_text_message("user", f"ESP32 有人事件:\n{llm.format_event_for_prompt(payload)}"),
    ]

    try:
        response = await llm.call_responses(messages)
    except Exception as exc:  # pragma: no cover - network failure
        LOGGER.warning("调用 DeepSeek 失败: %s", exc, exc_info=True)
        return None

    try:
        summary = llm.extract_first_text(response)
    except Exception as exc:  # pragma: no cover - unexpected payload
        LOGGER.warning("解析模型响应失败: %s", exc, exc_info=True)
        summary = ""

    return {"text": summary, "raw": response}


@app.post("/edge/event")
async def edge_event(event: schemas.EdgeEvent) -> Dict[str, Any]:
    actions: List[Dict[str, Any]] = []
    skipped = True

    ts_value = event.ts
    if isinstance(ts_value, datetime):
        ts = utils.isoformat_z(ts_value)
    else:
        ts = str(ts_value)

    if _should_forward_to_llm(event):
        llm_result = await _delegate_event_to_llm(event, ts)
        if llm_result is not None:
            actions.append(
                {
                    "action": "llm_consult",
                    "title": "ESP32 有人事件分析",
                    "text": llm_result.get("text", ""),
                    "device_id": event.device_id,
                    "ts": ts,
                    "llm_raw": llm_result.get("raw"),
                }
            )
            skipped = False

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
