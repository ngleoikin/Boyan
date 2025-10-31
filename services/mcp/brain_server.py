from __future__ import annotations
import os, re, sqlite3, time, uuid
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Response
from pydantic import BaseModel, Field

app = FastAPI()

# -----------------------
# 时区/DB 基础设施
# -----------------------
LOCAL_TZ = os.getenv("LOCAL_TZ", "UTC")
try:
    TZ = ZoneInfo(LOCAL_TZ)
except Exception:
    TZ = ZoneInfo("UTC")

def _now_utc() -> datetime:
    return datetime.now(timezone.utc)

def _now_local() -> datetime:
    return datetime.now(TZ)

def _iso(dt: datetime) -> str:
    # 统一输出 ISO8601 串（UTC 用 Z 结尾）
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def _db_path() -> str:
    p = os.getenv("ASSIST_DB")
    if not p:
        os.makedirs("./storage", exist_ok=True)
        p = "./storage/assist.db"
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p

def _db() -> sqlite3.Connection:
    conn = sqlite3.connect(_db_path())
    conn.row_factory = sqlite3.Row
    return conn

def _ensure_schema():
    with _db() as conn:
        c = conn.cursor()
        c.execute("""
        CREATE TABLE IF NOT EXISTS notes (
            id TEXT PRIMARY KEY,
            text TEXT NOT NULL,
            tags TEXT DEFAULT '',
            created_at TEXT NOT NULL
        )""")
        c.execute("""
        CREATE TABLE IF NOT EXISTS reminders (
            id TEXT PRIMARY KEY,
            text TEXT NOT NULL,
            scheduled_at TEXT NOT NULL,   -- ISO8601 UTC
            created_at TEXT NOT NULL
        )""")
        conn.commit()

_ensure_schema()

# -----------------------
# 事件联动（保留）
# -----------------------
class EdgeEvent(BaseModel):
    type: str = Field(..., description="事件类型字符串，如 detected/alert 等")
    device_id: str
    ts: str             # 建议 ISO8601 字符串
    cam: str
    topic: str          # vision/person | vision/face/verified | vision/ocr
    data: Dict[str, Any] = Field(default_factory=dict)

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/edge/event")
def edge_event(evt: EdgeEvent):
    """
    最小演示策略：
    - vision/person：confidence >= 0.8 触发 notify
    - vision/face/verified：无条件触发 greet
    - vision/ocr：text 非空触发 store_ocr
    """
    actions: List[Dict[str, Any]] = []
    skipped = True

    if evt.topic == "vision/person":
        conf = float(evt.data.get("confidence", 0.0))
        if conf >= 0.8:
            actions.append({
                "action": "notify",
                "title": "检测到人员",
                "text": f"cam={evt.cam}, confidence={conf:.2f}",
                "device_id": evt.device_id,
                "ts": evt.ts
            })
            skipped = False

    elif evt.topic == "vision/face/verified":
        name = str(evt.data.get("name", "未知"))
        score = evt.data.get("score", None)
        actions.append({
            "action": "greet",
            "title": "已识别人脸",
            "text": f"欢迎 {name}" + (f" (score={score})" if score is not None else ""),
            "device_id": evt.device_id,
            "ts": evt.ts
        })
        skipped = False

    elif evt.topic == "vision/ocr":
        text = str(evt.data.get("text", "") or "")
        if text.strip():
            actions.append({
                "action": "store_ocr",
                "title": "OCR文本入库",
                "len": len(text),
                "preview": text[:60],
                "device_id": evt.device_id,
                "ts": evt.ts
            })
            skipped = False

    return {"ok": True, "skipped": skipped, "actions": actions}

# -----------------------
# Notes API
# -----------------------
class NoteAdd(BaseModel):
    text: str
    tags: Optional[List[str]] = None

@app.post("/notes/add")
def notes_add(n: NoteAdd):
    _ensure_schema()
    nid = "n_" + str(int(time.time()*1000))
    created = _iso(_now_utc())
    tags = ",".join(n.tags) if n.tags else ""
    with _db() as conn:
        conn.execute("INSERT INTO notes(id, text, tags, created_at) VALUES(?,?,?,?)",
                     (nid, n.text, tags, created))
        conn.commit()
    return {"ok": True, "id": nid, "created_at": created}

@app.get("/notes/list")
def notes_list(q: Optional[str] = None, limit: int = 50):
    _ensure_schema()
    limit = max(1, min(limit, 200))
    with _db() as conn:
        if q:
            cur = conn.execute(
                "SELECT id, text, tags, created_at FROM notes "
                "WHERE text LIKE ? OR tags LIKE ? ORDER BY created_at DESC LIMIT ?",
                (f"%{q}%", f"%{q}%", limit))
        else:
            cur = conn.execute(
                "SELECT id, text, tags, created_at FROM notes "
                "ORDER BY created_at DESC LIMIT ?", (limit,))
        rows = [dict(r) for r in cur.fetchall()]
    return {"ok": True, "items": rows}

# -----------------------
# Reminders API
# -----------------------
class ReminderParseCreate(BaseModel):
    utterance: str
    text: Optional[str] = None  # 不传则从 utterance 中抽出

REL_MIN = re.compile(r"(\d+)\s*分钟后")
REL_HOUR = re.compile(r"(\d+)\s*小时后")
ABS_HM = re.compile(r"(?<!\d)(\d{1,2})[:：](\d{2})(?!\d)")              # HH:MM
ABS_YMDHM = re.compile(r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})[ T](\d{1,2})[:：](\d{2})")
TOMORROW_HM = re.compile(r"明天\s*(\d{1,2})[:：](\d{2})")

def parse_when(utt: str) -> datetime:
    now = _now_local()

    m = REL_MIN.search(utt)
    if m:
        return (now + timedelta(minutes=int(m.group(1)))).astimezone(timezone.utc)

    m = REL_HOUR.search(utt)
    if m:
        return (now + timedelta(hours=int(m.group(1)))).astimezone(timezone.utc)

    m = TOMORROW_HM.search(utt)
    if m:
        hh, mm = int(m.group(1)), int(m.group(2))
        target = datetime(now.year, now.month, now.day, hh, mm, tzinfo=TZ) + timedelta(days=1)
        return target.astimezone(timezone.utc)

    m = ABS_YMDHM.search(utt)
    if m:
        y, M, d, hh, mm = map(int, m.groups())
        target = datetime(y, M, d, hh, mm, tzinfo=TZ)
        return target.astimezone(timezone.utc)

    m = ABS_HM.search(utt)
    if m:
        hh, mm = int(m.group(1)), int(m.group(2))
        target = datetime(now.year, now.month, now.day, hh, mm, tzinfo=TZ)
        if target < now:
            target += timedelta(days=1)  # 今天已过，则顺延到明天
        return target.astimezone(timezone.utc)

    # 兜底：10 分钟后
    return (now + timedelta(minutes=10)).astimezone(timezone.utc)

@app.post("/reminders/parse_create")
def reminders_parse_create(r: ReminderParseCreate):
    _ensure_schema()
    text = (r.text or "").strip()
    when_utc = parse_when(r.utterance)
    if not text:
        # 简单从话术里拿末尾名词短语（非常保守）
        text = r.utterance.strip()

    rid = "rem_" + str(int(time.time()*1000))
    created = _iso(_now_utc())
    scheduled = _iso(when_utc)
    with _db() as conn:
        conn.execute("INSERT INTO reminders(id, text, scheduled_at, created_at) VALUES(?,?,?,?)",
                     (rid, text, scheduled, created))
        conn.commit()
    return {"ok": True, "id": rid, "text": text, "scheduled_at": scheduled, "created_at": created}

@app.get("/reminders/list")
def reminders_list(limit: int = 100):
    _ensure_schema()
    limit = max(1, min(limit, 300))
    with _db() as conn:
        cur = conn.execute(
            "SELECT id, text, scheduled_at, created_at FROM reminders "
            "ORDER BY scheduled_at ASC LIMIT ?", (limit,))
        rows = [dict(r) for r in cur.fetchall()]
    return {"ok": True, "items": rows}

@app.get("/reminders/export/ics")
def reminders_export_ics():
    _ensure_schema()
    with _db() as conn:
        cur = conn.execute(
            "SELECT id, text, scheduled_at FROM reminders ORDER BY scheduled_at ASC")
        rows = cur.fetchall()

    def ics_escape(s: str) -> str:
        return s.replace("\\", "\\\\").replace(",", "\\,").replace(";", "\\;").replace("\n", "\\n")

    lines = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//Boyan//Brain//CN"
    ]
    for r in rows:
        dt = r["scheduled_at"].replace("-", "").replace(":", "").replace("Z", "Z").replace("T", "T")
        uid = r["id"] + "@boyan"
        summary = ics_escape(r["text"])
        lines += [
            "BEGIN:VEVENT",
            f"UID:{uid}",
            f"DTSTART:{dt}",
            f"SUMMARY:{summary}",
            "END:VEVENT"
        ]
    lines.append("END:VCALENDAR")
    ics = "\r\n".join(lines) + "\r\n"
    return Response(content=ics, media_type="text/calendar; charset=utf-8")
