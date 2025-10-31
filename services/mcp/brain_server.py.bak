import os
import json
import time
import base64
import re
import sqlite3
import threading
from collections import deque, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger
import dateparser
from tzlocal import get_localzone

OPENAI_BASE = os.environ.get("OPENAI_BASE", "https://api.openai.com/v1")
OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "YOUR_KEY")
MODEL_ID = os.environ.get("MODEL_ID", "deepseek-v3.1")
MCP_GATE = os.environ.get("MCP_GATE", "http://127.0.0.1:8787")
LISTEN_PORT = int(os.environ.get("BRAIN_PORT", "8768"))
FACE_SEARCH_URL = os.environ.get("FACE_SEARCH_URL", "http://127.0.0.1:8787/face/search")
COOLDOWN_S = int(os.environ.get("COOLDOWN_S", "1200"))
NIGHT_START = int(os.environ.get("NIGHT_START", "22"))
NIGHT_END = int(os.environ.get("NIGHT_END", "7"))
FACE_GREETING_THR = float(os.environ.get("FACE_GREETING_THR", "0.55"))
PARSE_TIME_URL = os.environ.get("PARSE_TIME_URL", "http://127.0.0.1:8787/nlp/parse_time")
INGEST_NOTES_URL = os.environ.get("INGEST_NOTES_URL", "http://127.0.0.1:8787/notes/ingest")
JOURNAL_APPEND_URL = os.environ.get("JOURNAL_APPEND_URL", "http://127.0.0.1:8787/journal/append")
WHITELIST = set(
    s.strip() for s in os.environ.get("FACE_WHITELIST", "").split(",") if s.strip()
)
BLACKLIST = set(
    s.strip() for s in os.environ.get("FACE_BLACKLIST", "").split(",") if s.strip()
)

PROFILES = {
    "home": {
        "fall": {"beta": 0.65, "still_sec": 12, "cooldown": 120},
        "posture": {"sit_max_min": 90, "stand_max_min": 240},
        "audio": {"bang_db": 72, "bang_min_gap": 20},
        "ppe": {"enabled": False},
        "tone": {"greeting_soft": True},
    },
    "factory": {
        "fall": {"beta": 0.6, "still_sec": 8, "cooldown": 90},
        "posture": {"sit_max_min": 60, "stand_max_min": 180},
        "audio": {"bang_db": 78, "bang_min_gap": 30},
        "ppe": {"enabled": True},
        "tone": {"greeting_soft": False},
    },
}
CURRENT = {"profile": os.environ.get("DEFAULT_PROFILE", "home"), "cfg": None}
if CURRENT["profile"] not in PROFILES:
    CURRENT["profile"] = "home"
CURRENT["cfg"] = PROFILES[CURRENT["profile"]]

INTENT_COOLDOWN = {
    "greet": int(os.environ.get("CD_GREET_S", "1200")),
    "confirm_identity": int(os.environ.get("CD_CONFIRM_S", "600")),
    "ask_help": int(os.environ.get("CD_ASKHELP_S", "900")),
    "read_note": int(os.environ.get("CD_READNOTE_S", "300")),
}
LINGER_SECONDS = int(os.environ.get("ANOM_LINGER_S", "45"))
MIN_MOVE_PIXELS = int(os.environ.get("ANOM_MIN_MOVE", "40"))
PACER_WINDOW_S = int(os.environ.get("ANOM_PACER_WIN", "20"))
PACER_FLIPS = int(os.environ.get("ANOM_PACER_FLIP", "6"))
TTS_MIN_INTERVAL_S = int(os.environ.get("TTS_MIN_INTERVAL_S", "3"))
TTS_DEDUP_S = int(os.environ.get("TTS_DEDUP_S", "600"))


FUSE_CFG = {
    "anyone_attentive_sec": int(os.environ.get("ANYONE_ATTENTIVE_SEC", "30")),
    "familiar_cooldown": int(os.environ.get("FAMILIAR_COOLDOWN_SEC", "120")),
    "backoff_sec": int(os.environ.get("FUSE_BACKOFF_SEC", "6")),
}
FUSE = {
    "state": "IDLE",
    "attentive_until": 0.0,
    "last_engaged": 0.0,
    "greeted": {},
}
ESCALATE = {"fall_window": 20, "fall_count": 0, "fall_first": 0.0}
QUIET_HOURS = {"start": None, "end": None}


@dataclass
class PersonState:
    key: str
    last_seen_ts: float = 0.0
    positions: deque = field(default_factory=lambda: deque(maxlen=120))
    last_intent_ts: dict = field(default_factory=dict)


_PERSONS: Dict[str, PersonState] = {}
_COOLDOWN: Dict[str, float] = {}
_LAST_TTS_TS: float = 0.0
_TTS_TEXT_CACHE: deque = deque(maxlen=50)

SYSTEM_PROMPT = """你是“小智”，驻留在一个带摄像头与扬声器的设备内，工作地点可能是家中或车间。你的目标是“像一个礼貌、细心、克制的熟人”，做到不打扰但能在需要时主动帮忙。

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
  - 先用 note.create 记录文本；
  - 若语句包含时间或需要提醒，再调用 reminder.create（它会自动解析自然语言并设置提醒）。
- 不要把 OCR 的公式或文本包装为 LaTeX；尽量原样返回。
- 如果你需要更多上下文，可以先 note.search 或 reminder.list，再决定是否询问或提醒。

【场景提示】
- 家中：优先减少打扰；如果是固定家人，可更亲切问候。
- 车间：优先安全与效率；异常时主动提醒安全注意事项；少说无关话。

【对话风格】
- 礼貌、克制、友好；不自报长篇；确认/问候/提醒控制在 3–12 字或一句短句。
- 你可以使用第一人称，但避免拟人化自夸。
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "ocr_read",
            "description": "对给定图片（可选 ROI）做 OCR，返回原始文本。适合胸卡、白板、便签等。",
            "parameters": {
                "type": "object",
                "properties": {
                    "image_b64": {"type": "string"},
                    "mime": {"type": "string", "default": "image/png"},
                    "roi": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "minItems": 4,
                        "maxItems": 4,
                    },
                },
                "required": ["image_b64"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "tts_speak",
            "description": "通过本地 TTS 播报中文/英文。用于问候、确认、提醒。",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "voice": {"type": "string"},
                    "volume": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 100,
                        "default": 80,
                    },
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "mem_write",
            "description": "把用户口述/看到的事项记到记事本（note 或 todo），可带时间。",
            "parameters": {
                "type": "object",
                "properties": {
                    "kind": {"type": "string", "enum": ["note", "todo"], "default": "note"},
                    "text": {"type": "string"},
                    "when": {"type": "string"},
                },
                "required": ["kind", "text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "mem_search",
            "description": "检索记事。",
            "parameters": {
                "type": "object",
                "properties": {"q": {"type": "string"}},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "reminder_set",
            "description": "为某条 note 或文本设提醒。",
            "parameters": {
                "type": "object",
                "properties": {
                    "note_id": {"type": "string"},
                    "text": {"type": "string"},
                    "when": {"type": "string"},
                    "repeat_cron": {"type": "string", "nullable": True},
                },
                "required": ["when"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "reminder_list",
            "description": "列出当前提醒（含已触发/未触发）。",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "reminder_cancel",
            "description": "取消一个提醒。",
            "parameters": {
                "type": "object",
                "properties": {"id": {"type": "string"}},
                "required": ["id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "note.create",
            "description": "写入一条新的记事。",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "可选标签数组",
                    },
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "note.search",
            "description": "按关键词查看最近的记事。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 200, "default": 100},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "note.delete",
            "description": "删除一条记事。",
            "parameters": {
                "type": "object",
                "properties": {"id": {"type": "integer"}},
                "required": ["id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "reminder.create",
            "description": "根据自然语言创建提醒。",
            "parameters": {
                "type": "object",
                "properties": {
                    "utterance": {"type": "string"},
                    "text": {"type": "string", "description": "可选，自定义提醒播报内容"},
                    "tz": {"type": "string", "description": "可选，时区标识"},
                },
                "required": ["utterance"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "reminder.list",
            "description": "列出当前提醒。",
            "parameters": {
                "type": "object",
                "properties": {
                    "status": {"type": "string", "description": "过滤状态，如 scheduled/fired/cancelled"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 200, "default": 100},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "reminder.cancel",
            "description": "取消一个提醒。",
            "parameters": {
                "type": "object",
                "properties": {"id": {"type": "integer"}},
                "required": ["id"],
            },
        },
    },
]

app = FastAPI(title="Brain Orchestrator", version="0.2")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class SetProfileIn(BaseModel):
    name: str


class QuietConfigIn(BaseModel):
    start: Optional[str] = None
    end: Optional[str] = None


@app.get("/policy/profile")
def policy_profile():
    return CURRENT


@app.post("/policy/profile")
def set_policy_profile(inp: SetProfileIn):
    name = inp.name.lower()
    if name not in PROFILES:
        raise HTTPException(status_code=400, detail="unknown profile")
    CURRENT["profile"] = name
    CURRENT["cfg"] = PROFILES[name]
    return CURRENT


@app.post("/policy/quiet")
def set_quiet_hours(inp: QuietConfigIn):
    QUIET_HOURS["start"] = inp.start
    QUIET_HOURS["end"] = inp.end
    return QUIET_HOURS


DB_PATH = os.environ.get("ASSIST_DB", os.path.abspath(os.path.join(os.getcwd(), "assist.db")))
LOCAL_TZ_NAME = os.environ.get("LOCAL_TZ") or str(get_localzone())
_DB_LOCK = threading.Lock()
_SCHED_LOCK = threading.Lock()
_SCHED: Optional[BackgroundScheduler] = None


def _db_connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def _init_db() -> None:
    with _DB_LOCK:
        conn = _db_connect()
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS notes(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                tags TEXT NOT NULL DEFAULT '[]',
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS reminders(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                due_ts_utc REAL,
                cron TEXT,
                tz TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at REAL NOT NULL
            );
            """
        )
        conn.commit()
        conn.close()


def _ensure_db_ready() -> None:
    if not os.path.exists(DB_PATH):
        _init_db()
    else:
        _init_db()


_ensure_db_ready()


def _publish_event(topic: str, data: Dict[str, Any]) -> None:
    try:
        with httpx.Client(timeout=2) as cli:
            cli.post(
                f"http://127.0.0.1:{LISTEN_PORT}/events/publish",
                json={"topic": topic, "data": data},
            )
    except Exception:
        pass


def _schedule_job(reminder_id: int, due_ts_utc: Optional[float], cron: Optional[str], tzname: str) -> None:
    if _SCHED is None:
        return
    job_id = f"rem-{reminder_id}"
    if cron:
        trigger = CronTrigger.from_crontab(cron, timezone=tzname)
    else:
        trigger = DateTrigger(run_date=datetime.fromtimestamp(due_ts_utc, tz=timezone.utc))
    _SCHED.add_job(_on_fire, trigger, args=[reminder_id], id=job_id, replace_existing=True)


def _reload_jobs() -> None:
    if _SCHED is None:
        return
    with _DB_LOCK:
        conn = _db_connect()
        rows = conn.execute(
            "SELECT id, due_ts_utc, cron, tz FROM reminders WHERE status='scheduled'"
        ).fetchall()
        conn.close()
    for row in rows:
        _schedule_job(row["id"], row["due_ts_utc"], row["cron"], row["tz"])


def _catch_up_missed() -> None:
    if _SCHED is None:
        return
    now = time.time()
    with _DB_LOCK:
        conn = _db_connect()
        rows = conn.execute(
            """
            SELECT id, text, due_ts_utc, tz
            FROM reminders
            WHERE status='scheduled' AND cron IS NULL AND due_ts_utc IS NOT NULL AND due_ts_utc<?
            """,
            (now,),
        ).fetchall()
        for row in rows:
            _deliver_reminder(row["id"], row["text"], row["tz"], missed=True)
            conn.execute("UPDATE reminders SET status='fired' WHERE id=?", (row["id"],))
        conn.commit()
        conn.close()


def _ensure_scheduler_started() -> None:
    global _SCHED
    if _SCHED is not None:
        return
    with _SCHED_LOCK:
        if _SCHED is not None:
            return
        _ensure_db_ready()
        scheduler = BackgroundScheduler()
        scheduler.start()
        _SCHED = scheduler
        _reload_jobs()
        _catch_up_missed()


def _deliver_reminder(reminder_id: int, text: str, tzname: str, missed: bool = False) -> None:
    prefix = "（补）提醒：" if missed else "提醒："
    message = f"{prefix}{text}"
    if not speak(message):
        exec_tool_call({"function": {"name": "tts_speak", "arguments": json.dumps({"text": message})}})
    _publish_event(
        "reminder/missed" if missed else "reminder/fired",
        {"id": reminder_id, "text": text, "tz": tzname, "ts": time.time()},
    )


def _on_fire(reminder_id: int) -> None:
    with _DB_LOCK:
        conn = _db_connect()
        row = conn.execute(
            "SELECT id, text, cron, tz FROM reminders WHERE id=?", (reminder_id,)
        ).fetchone()
        if not row:
            conn.close()
            return
        text = row["text"]
        cron = row["cron"]
        tzname = row["tz"]
        if not cron:
            conn.execute("UPDATE reminders SET status='fired' WHERE id=?", (reminder_id,))
            conn.commit()
        conn.close()
    _deliver_reminder(reminder_id, text, tzname)


def parse_when(utterance: str, base_tz: str = LOCAL_TZ_NAME) -> Optional[Dict[str, Any]]:
    text = utterance.strip().lower()
    everyday = re.search(r"每天\s*(\d{1,2})[:：点时](\d{1,2})?", text)
    if everyday:
        hh = int(everyday.group(1))
        mm = int(everyday.group(2) or 0)
        return {"mode": "cron", "ts_utc": None, "cron": f"{mm} {hh} * * *", "tz": base_tz}

    weekday = re.search(r"每周([一二三四五六日天])\s*(\d{1,2})[:：点时](\d{1,2})?", text)
    if weekday:
        wk_map = {"一": "MON", "二": "TUE", "三": "WED", "四": "THU", "五": "FRI", "六": "SAT", "日": "SUN", "天": "SUN"}
        dow = wk_map.get(weekday.group(1), "MON")
        hh = int(weekday.group(2))
        mm = int(weekday.group(3) or 0)
        return {"mode": "cron", "ts_utc": None, "cron": f"{mm} {hh} * * {dow}", "tz": base_tz}

    monthday = re.search(r"每月\s*(\d{1,2})\s*[号日]\s*(\d{1,2})[:：点时](\d{1,2})?", text)
    if monthday:
        day = int(monthday.group(1))
        hh = int(monthday.group(2))
        mm = int(monthday.group(3) or 0)
        return {"mode": "cron", "ts_utc": None, "cron": f"{mm} {hh} {day} * *", "tz": base_tz}

    dt = dateparser.parse(
        utterance,
        settings={
            "TIMEZONE": base_tz,
            "RETURN_AS_TIMEZONE_AWARE": True,
            "PREFER_DATES_FROM": "future",
        },
    )
    if not dt:
        return None
    ts_utc = dt.astimezone(timezone.utc).timestamp()
    return {"mode": "once", "ts_utc": ts_utc, "cron": None, "tz": base_tz}


class NoteIn(BaseModel):
    text: str
    tags: Optional[List[str]] = None


class NoteIdIn(BaseModel):
    id: int


class RemindNLIn(BaseModel):
    utterance: str
    text: Optional[str] = None
    tz: Optional[str] = None


class ReminderIdIn(BaseModel):
    id: int


@app.post("/notes/add")
def notes_add(inp: NoteIn):
    now = time.time()
    payload_tags = json.dumps(inp.tags or [], ensure_ascii=False)
    with _DB_LOCK:
        conn = _db_connect()
        conn.execute(
            "INSERT INTO notes(text, tags, created_at, updated_at) VALUES(?,?,?,?)",
            (inp.text, payload_tags, now, now),
        )
        conn.commit()
        nid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.close()
    return {"ok": True, "id": nid}


@app.get("/notes/list")
def notes_list(q: Optional[str] = None, limit: int = 100):
    with _DB_LOCK:
        conn = _db_connect()
        if q:
            rows = conn.execute(
                "SELECT id, text, tags, created_at, updated_at FROM notes WHERE text LIKE ? ORDER BY id DESC LIMIT ?",
                (f"%{q}%", limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, text, tags, created_at, updated_at FROM notes ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        conn.close()
    items = []
    for row in rows:
        items.append(
            {
                "id": row["id"],
                "text": row["text"],
                "tags": json.loads(row["tags"] or "[]"),
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }
        )
    return {"ok": True, "items": items}


@app.post("/notes/delete")
def notes_delete(req: NoteIdIn):
    with _DB_LOCK:
        conn = _db_connect()
        conn.execute("DELETE FROM notes WHERE id=?", (req.id,))
        conn.commit()
        conn.close()
    return {"ok": True}


@app.post("/reminders/parse_create")
def reminders_parse_create(inp: RemindNLIn):
    ensure_tz = inp.tz or LOCAL_TZ_NAME
    parsed = parse_when(inp.utterance, ensure_tz)
    if not parsed:
        raise HTTPException(
            status_code=400,
            detail="无法解析时间，请尝试诸如‘今天15:00’、‘两小时后’或‘每周一9点’的表达",
        )
    reminder_text = inp.text or inp.utterance
    now = time.time()
    with _DB_LOCK:
        conn = _db_connect()
        conn.execute(
            "INSERT INTO reminders(text, due_ts_utc, cron, tz, status, created_at) VALUES(?,?,?,?,?,?)",
            (
                reminder_text,
                parsed.get("ts_utc"),
                parsed.get("cron"),
                parsed["tz"],
                "scheduled",
                now,
            ),
        )
        conn.commit()
        rid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.close()
    _ensure_scheduler_started()
    _schedule_job(rid, parsed.get("ts_utc"), parsed.get("cron"), parsed["tz"])
    return {"ok": True, "id": rid, "parsed": parsed}


@app.get("/reminders/list")
def reminders_list(status: Optional[str] = None, limit: int = 100):
    with _DB_LOCK:
        conn = _db_connect()
        if status:
            rows = conn.execute(
                "SELECT id, text, due_ts_utc, cron, tz, status, created_at FROM reminders WHERE status=? ORDER BY id DESC LIMIT ?",
                (status, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, text, due_ts_utc, cron, tz, status, created_at FROM reminders ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        conn.close()
    items = []
    for row in rows:
        items.append(
            {
                "id": row["id"],
                "text": row["text"],
                "due_ts_utc": row["due_ts_utc"],
                "cron": row["cron"],
                "tz": row["tz"],
                "status": row["status"],
                "created_at": row["created_at"],
            }
        )
    return {"ok": True, "items": items}


@app.post("/reminders/cancel")
def reminders_cancel(req: ReminderIdIn):
    with _DB_LOCK:
        conn = _db_connect()
        conn.execute("UPDATE reminders SET status='cancelled' WHERE id=?", (req.id,))
        conn.commit()
        conn.close()
    if _SCHED is not None:
        try:
            _SCHED.remove_job(f"rem-{req.id}")
        except Exception:
            pass
    return {"ok": True}


@app.get("/reminders/export/ics")
def reminders_export_ics():
    with _DB_LOCK:
        conn = _db_connect()
        rows = conn.execute(
            "SELECT id, text, due_ts_utc, tz FROM reminders WHERE cron IS NULL AND due_ts_utc IS NOT NULL"
        ).fetchall()
        conn.close()
    from uuid import uuid4

    lines = ["BEGIN:VCALENDAR", "VERSION:2.0", "PRODID:-//xiaozhi//reminders//CN"]
    for row in rows:
        dt = datetime.fromtimestamp(row["due_ts_utc"], tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        lines.extend(
            [
                "BEGIN:VEVENT",
                f"UID:{uuid4()}",
                f"DTSTART:{dt}",
                f"SUMMARY:{row['text']}",
                "END:VEVENT",
            ]
        )
    lines.append("END:VCALENDAR")
    return PlainTextResponse("\r\n".join(lines), media_type="text/calendar; charset=utf-8")

class EdgeEvent(BaseModel):
    type: str
    ts: str
    cam: str
    motion: float | None = None
    cooldown_s: int | None = None
    snapshot_jpeg_b64: str | None = None
    bbox: List[float] | None = None
    scene_tag: str | None = None


class BusEvent(BaseModel):
    topic: str
    data: Dict[str, Any] | None = None


def chat_with_tools(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    headers = {"Authorization": f"Bearer {OPENAI_KEY}"}
    payload = {"model": MODEL_ID, "messages": messages, "tools": TOOLS, "temperature": 0.4}
    with httpx.Client(base_url=OPENAI_BASE, timeout=120) as cli:
        resp = cli.post("/chat/completions", json=payload, headers=headers)
        resp.raise_for_status()
        return resp.json()


def exec_tool_call(call: Dict[str, Any]) -> Dict[str, Any]:
    name = call["function"]["name"]
    args = json.loads(call["function"].get("arguments", "{}"))
    with httpx.Client(timeout=120) as cli:
        if name == "ocr_read":
            res = cli.post(f"{MCP_GATE}/tools/ocr/read", json=args).json()
        elif name == "tts_speak":
            res = cli.post(f"{MCP_GATE}/tts/speak", json=args).json()
        elif name == "mem_write":
            res = cli.post(f"{MCP_GATE}/mem/write", json=args).json()
        elif name == "mem_search":
            res = cli.post(f"{MCP_GATE}/mem/search", json=args).json()
        elif name == "reminder_set":
            res = cli.post(f"{MCP_GATE}/reminder/set", json=args).json()
        elif name == "reminder_list":
            res = cli.get(f"{MCP_GATE}/reminder/list").json()
        elif name == "reminder_cancel":
            res = cli.post(f"{MCP_GATE}/reminder/cancel", json=args).json()
        elif name == "note.create":
            res = notes_add(NoteIn(**args))
        elif name == "note.search":
            limit = args.get("limit", 100)
            try:
                limit = int(limit)
            except (TypeError, ValueError):
                limit = 100
            res = notes_list(q=args.get("query"), limit=limit)
        elif name == "note.delete":
            res = notes_delete(NoteIdIn(**args))
        elif name == "reminder.create":
            res = reminders_parse_create(RemindNLIn(**args))
        elif name == "reminder.list":
            limit = args.get("limit", 100)
            try:
                limit = int(limit)
            except (TypeError, ValueError):
                limit = 100
            res = reminders_list(status=args.get("status"), limit=limit)
        elif name == "reminder.cancel":
            res = reminders_cancel(ReminderIdIn(**args))
        else:
            res = {"ok": False, "error": f"unknown_tool:{name}"}
    return {"tool_name": name, "tool_result": res}


def face_search_top1(image_b64: str) -> dict | None:
    try:
        with httpx.Client(timeout=8) as cli:
            payload = {"image_b64": image_b64, "top_k": 1, "thr": 0.0}
            resp = cli.post(FACE_SEARCH_URL, json=payload)
            resp.raise_for_status()
            return resp.json().get("top1")
    except Exception:
        return None


def _path_length(positions: deque, since_ts: float) -> float:
    points = [(ts, x, y) for ts, x, y in positions if ts >= since_ts]
    if len(points) < 2:
        return 0.0
    dist = 0.0
    for i in range(1, len(points)):
        _, x1, y1 = points[i - 1]
        _, x2, y2 = points[i]
        dx = x2 - x1
        dy = y2 - y1
        dist += (dx * dx + dy * dy) ** 0.5
    return dist


def _dir_flip_count(positions: deque, window_s: int) -> int:
    if len(positions) < 3:
        return 0
    now = time.time()
    pts = [(ts, x, y) for ts, x, y in positions if now - ts <= window_s]
    if len(pts) < 3:
        return 0
    flips = 0
    last_sign = 0
    for i in range(1, len(pts)):
        _, x1, _ = pts[i - 1]
        _, x2, _ = pts[i]
        sign = 0
        if x2 > x1 + 2:
            sign = 1
        elif x2 < x1 - 2:
            sign = -1
        if sign != 0:
            if last_sign != 0 and sign != last_sign:
                flips += 1
            last_sign = sign
    return flips


def _record_position(state: PersonState, bbox: List[float] | None):
    ts = time.time()
    state.last_seen_ts = ts
    if bbox and len(bbox) >= 4:
        x, y, w, h = bbox[:4]
        cx = x + w / 2.0
        cy = y + h / 2.0
        state.positions.append((ts, cx, cy))
    else:
        state.positions.append((ts, 0.0, 0.0))


def detect_anomaly(state: PersonState) -> Dict[str, Any]:
    now = time.time()
    move = _path_length(state.positions, now - LINGER_SECONDS)
    linger = (now - state.last_seen_ts <= 2.0) and (move < MIN_MOVE_PIXELS) and len(state.positions) >= 2
    flips = _dir_flip_count(state.positions, PACER_WINDOW_S)
    pacing = flips >= PACER_FLIPS
    reasons: List[str] = []
    if linger:
        reasons.append(f"linger≥{LINGER_SECONDS}s & move<{MIN_MOVE_PIXELS}px")
    if pacing:
        reasons.append(f"pacing_flips≥{PACER_FLIPS}@{PACER_WINDOW_S}s")
    return {
        "linger": linger,
        "pacing": pacing,
        "reason": "; ".join(reasons) if reasons else "",
        "metrics": {"move_px": move, "flips": flips},
    }


def in_intent_cooldown(state: PersonState, intent: str) -> bool:
    ttl = INTENT_COOLDOWN.get(intent, COOLDOWN_S)
    last = state.last_intent_ts.get(intent, 0.0)
    return time.time() - last < ttl


def mark_intent(state: PersonState, intent: str):
    state.last_intent_ts[intent] = time.time()


def tts_maybe(text: str) -> bool:
    global _LAST_TTS_TS
    now = time.time()
    if now - _LAST_TTS_TS < TTS_MIN_INTERVAL_S:
        return False
    while _TTS_TEXT_CACHE and now - _TTS_TEXT_CACHE[0][0] > TTS_DEDUP_S:
        _TTS_TEXT_CACHE.popleft()
    if any(t == text for _, t in _TTS_TEXT_CACHE):
        return False
    exec_tool_call({"function": {"name": "tts_speak", "arguments": json.dumps({"text": text})}})
    _LAST_TTS_TS = now
    _TTS_TEXT_CACHE.append((now, text))
    return True


def speak(text: str) -> bool:
    return tts_maybe(text)


try:
    _ensure_scheduler_started()
except Exception:
    pass


def _quiet_active() -> bool:
    start = QUIET_HOURS.get("start")
    end = QUIET_HOURS.get("end")
    if not start or not end:
        return False
    try:
        start_dt = datetime.strptime(start, "%H:%M").time()
        end_dt = datetime.strptime(end, "%H:%M").time()
    except ValueError:
        return False
    now = datetime.now().time()
    if start_dt <= end_dt:
        return start_dt <= now < end_dt
    return now >= start_dt or now < end_dt


def _journal(source: str, text: str, meta: Dict[str, Any] | None = None) -> None:
    payload = {"source": source, "text": text, "meta": meta or {}}
    try:
        with httpx.Client(timeout=3) as cli:
            cli.post(JOURNAL_APPEND_URL, json=payload)
    except Exception:
        pass


def _level1_checkin(meta: Dict[str, Any]) -> None:
    if not _quiet_active():
        speak("我在这儿。看起来不太对劲，需要帮忙吗？")
    _journal("guard", "L1 check-in", meta)


def _level2_confirm(meta: Dict[str, Any]) -> None:
    if not _quiet_active():
        speak("我有点担心，您还好吗？如果需要请告诉我。")
    _journal("guard", "L2 confirm", meta)


def _level3_escalate(meta: Dict[str, Any]) -> None:
    if not _quiet_active():
        speak("我已记录异常，会一直在这。")
    _journal("guard", "L3 escalate", meta)


@app.get("/health")
def health():
    return {"ok": True}


def _enter_engaged(reason: str, meta: Dict[str, Any] | None = None) -> Dict[str, Any]:
    now = time.time()
    FUSE["state"] = "ENGAGED"
    FUSE["last_engaged"] = now
    if meta and meta.get("person_id"):
        FUSE["greeted"][meta["person_id"]] = now
    _journal("fuse", f"enter_engaged:{reason}", meta or {})
    return {"ok": True, "state": FUSE["state"], "reason": reason}


@app.post("/events/publish")
def events_publish(evt: BusEvent):
    topic = evt.topic
    data = evt.data or {}
    now = time.time()

    if FUSE["state"] == "IDLE" and now - FUSE.get("last_engaged", 0.0) < FUSE_CFG["backoff_sec"]:
        return {"ok": True, "state": FUSE["state"], "backoff": True}

    if topic == "vision/person":
        FUSE["state"] = "ATTENTIVE"
        FUSE["attentive_until"] = max(FUSE["attentive_until"], now + FUSE_CFG["anyone_attentive_sec"])
        return {"ok": True, "state": FUSE["state"]}

    if topic == "vision/familiar":
        pid = data.get("person_id")
        if not pid:
            return {"ok": True, "state": FUSE["state"]}
        last = FUSE["greeted"].get(pid, 0.0)
        if now - last >= FUSE_CFG["familiar_cooldown"]:
            return _enter_engaged("familiar", {"person_id": pid})
        FUSE["state"] = "ATTENTIVE"
        FUSE["attentive_until"] = max(FUSE["attentive_until"], now + FUSE_CFG["anyone_attentive_sec"])
        return {"ok": True, "state": FUSE["state"], "cooldown": True}

    if topic == "wake/keyword":
        if not _quiet_active():
            tone = CURRENT["cfg"].get("tone", {})
            speak("我在，请说。" if tone.get("greeting_soft", True) else "我在，您请讲。")
        _journal("wake", "keyword", data)
        FUSE["last_engaged"] = now
        return {"ok": True, "state": FUSE["state"]}

    if topic == "vision/fall_suspect":
        if ESCALATE["fall_first"] == 0.0 or now - ESCALATE["fall_first"] > ESCALATE["fall_window"]:
            ESCALATE["fall_first"] = now
            ESCALATE["fall_count"] = 1
        else:
            ESCALATE["fall_count"] += 1
        _level1_checkin({"type": "fall", "data": data})
        if ESCALATE["fall_count"] >= 2:
            _level2_confirm({"type": "fall", "data": data})
        if ESCALATE["fall_count"] >= 3:
            _level3_escalate({"type": "fall", "data": data})
        return {"ok": True, "state": FUSE["state"], "fall_count": ESCALATE["fall_count"]}

    if topic == "audio/loud_bang":
        _level1_checkin({"type": "bang", "data": data})
        return {"ok": True, "state": FUSE["state"]}

    if topic == "audio/distress_keyword":
        _level2_confirm({"type": "distress", "data": data})
        return {"ok": True, "state": FUSE["state"]}

    return {"ok": True, "state": FUSE["state"], "unknown": topic}


@app.post("/edge/event")
def on_edge_event(evt: EdgeEvent):
    if evt.type != "person_entered":
        return {"ok": True, "skipped": True}

    hour = datetime.now().hour
    night = hour >= NIGHT_START or hour < NIGHT_END
    quiet = _quiet_active()

    top1 = None
    if evt.snapshot_jpeg_b64:
        top1 = face_search_top1(evt.snapshot_jpeg_b64)

    if top1 and top1.get("score", 0.0) >= FACE_GREETING_THR:
        who = top1["name"]
    else:
        who = "unknown"

    person_key = f"{who}@{evt.cam}"
    now = time.time()
    last = _COOLDOWN.get(person_key, 0.0)
    in_global_cooldown = (now - last) < COOLDOWN_S

    if who in BLACKLIST:
        return {"ok": True, "blacklisted": who}

    state = _PERSONS.get(person_key)
    if state is None:
        state = PersonState(key=person_key)
        _PERSONS[person_key] = state

    _record_position(state, evt.bbox)
    anomaly = detect_anomaly(state)
    is_anom = anomaly["linger"] or anomaly["pacing"]

    if FUSE["state"] == "IDLE" and now - FUSE.get("last_engaged", 0.0) < FUSE_CFG["backoff_sec"] and not is_anom:
        return {"ok": True, "fuse_backoff": True, "who": who}

    FUSE["state"] = "ATTENTIVE"
    FUSE["attentive_until"] = max(FUSE["attentive_until"], now + FUSE_CFG["anyone_attentive_sec"])

    if who != "unknown" and who not in WHITELIST and not is_anom:
        last_greet = FUSE["greeted"].get(who, 0.0)
        if now - last_greet < FUSE_CFG["familiar_cooldown"]:
            return {"ok": True, "fuse_familiar_cooldown": True, "who": who}

    if night and not is_anom:
        return {"ok": True, "night_quiet": True, "who": who}
    if quiet and not is_anom:
        return {"ok": True, "quiet_hours": True, "who": who}

    intent_order: List[str] = []
    if is_anom:
        intent_order.append("ask_help")
    if who == "unknown":
        intent_order.append("confirm_identity")
    else:
        intent_order.append("greet")
    if evt.scene_tag == "note_board":
        intent_order.append("read_note")

    chosen_intent = None
    for intent in intent_order:
        if not in_intent_cooldown(state, intent):
            chosen_intent = intent
            break

    if chosen_intent is None and not is_anom:
        if who not in WHITELIST and in_global_cooldown:
            return {"ok": True, "cooldown_all": True, "who": who}
        chosen_intent = "greet"

    if chosen_intent is None:
        chosen_intent = "ask_help" if is_anom else "greet"

    if who not in WHITELIST and in_global_cooldown and chosen_intent == "greet" and not is_anom:
        return {"ok": True, "cooldown_silent": True, "who": who}

    scene_capsule = {
        "time_local": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "place": "home_or_workshop",
        "cam": evt.cam,
        "night_mode": night,
        "cooldown_left_s": max(0, COOLDOWN_S - int(now - last)) if last else 0,
        "motion": evt.motion or 0.0,
        "who": who,
        "face_score": top1["score"] if top1 else 0.0,
        "anomaly": anomaly,
        "candidate_intents": intent_order,
        "chosen_intent": chosen_intent,
    }

    user_parts: List[Dict[str, Any]] = [
        {
            "type": "text",
            "text": "事件：摄像头检测到有人进入。请遵循系统原则，自行决定是否发声、是否需要先确认身份、是否需要 OCR 读取胸卡或便签。",
        },
        {"type": "text", "text": f"场景胶囊：{json.dumps(scene_capsule, ensure_ascii=False)}"},
    ]
    if evt.snapshot_jpeg_b64:
        user_parts.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{evt.snapshot_jpeg_b64}"},
            }
        )

    user_parts.append(
        {
            "type": "text",
            "text": f"系统建议意图：{chosen_intent}；异常={is_anom}（{anomaly['reason']}）",
        }
    )

    sys_msg = {"role": "system", "content": SYSTEM_PROMPT}
    user_msg = {"role": "user", "content": user_parts}
    messages = [sys_msg, user_msg]
    tool_messages: List[Dict[str, Any]] = []

    for _ in range(4):
        resp = chat_with_tools(messages + tool_messages)
        choice = resp["choices"][0]
        message = choice["message"]
        tool_calls = message.get("tool_calls") or message.get("function_call")

        if not tool_calls:
            content = message.get("content") or ""
            if isinstance(content, list):
                content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
            if isinstance(content, str):
                text = content.strip()
            else:
                text = ""
            if text:
                if tts_maybe(text):
                    mark_intent(state, chosen_intent)
                    _COOLDOWN[person_key] = time.time()
                    FUSE["state"] = "IDLE"
                    FUSE["last_engaged"] = time.time()
                    if who != "unknown":
                        FUSE["greeted"][who] = FUSE["last_engaged"]
            break

        if isinstance(tool_calls, dict) and "name" in tool_calls:
            tool_calls = [tool_calls]

        for call in tool_calls:
            func = call.get("function", call)
            name = func.get("name")
            arguments_json = func.get("arguments", "{}")
            if name == "tts_speak":
                args = json.loads(arguments_json or "{}")
                spoken = args.get("text", "").strip()
                if spoken:
                    if tts_maybe(spoken):
                        mark_intent(state, chosen_intent)
                        _COOLDOWN[person_key] = time.time()
                        FUSE["state"] = "IDLE"
                        FUSE["last_engaged"] = time.time()
                        if who != "unknown":
                            FUSE["greeted"][who] = FUSE["last_engaged"]
                continue
            if name == "ocr_read":
                args = json.loads(arguments_json or "{}")
                result = exec_tool_call(
                    {"function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}}
                )
                payload = result.get("tool_result", {})
                text_payload = ""
                if isinstance(payload, dict):
                    text_payload = (payload.get("text") or "").strip()
                ingest_info: Dict[str, Any] | None = None
                if text_payload:
                    _journal("ocr", text_payload, {"cam": evt.cam})
                    try:
                        with httpx.Client(timeout=10) as cli:
                            resp = cli.post(
                                INGEST_NOTES_URL, json={"text": text_payload, "source": "ocr"}
                            )
                            resp.raise_for_status()
                            ingest_info = resp.json()
                    except Exception as exc:
                        ingest_info = {"ok": False, "error": str(exc)}
                if isinstance(payload, dict) and ingest_info is not None:
                    payload = {**payload, "ingest": ingest_info}
                    result["tool_result"] = payload
                tool_messages.append(
                    {
                        "role": "tool",
                        "name": name,
                        "content": json.dumps(result["tool_result"], ensure_ascii=False),
                    }
                )
                continue
            if name == "mem_write":
                args = json.loads(arguments_json or "{}")
                note_text = (args.get("text") or "").strip()
                if note_text and not args.get("when"):
                    try:
                        with httpx.Client(timeout=4) as cli:
                            resp = cli.post(PARSE_TIME_URL, json={"text": note_text})
                            resp.raise_for_status()
                            parsed = (resp.json() or {}).get("items") or []
                            if parsed:
                                args["when"] = parsed[0]["iso"]
                                arguments_json = json.dumps(args, ensure_ascii=False)
                    except Exception:
                        pass
                result = exec_tool_call(
                    {"function": {"name": name, "arguments": arguments_json}}
                )
                tool_messages.append(
                    {
                        "role": "tool",
                        "name": name,
                        "content": json.dumps(result["tool_result"], ensure_ascii=False),
                    }
                )
                if note_text:
                    _journal("user", note_text, {"auto_when": args.get("when")})
                mark_intent(state, "read_note")
                continue
            result = exec_tool_call(
                {"function": {"name": name, "arguments": arguments_json}}
            )
            tool_messages.append(
                {
                    "role": "tool",
                    "name": name,
                    "content": json.dumps(result["tool_result"], ensure_ascii=False),
                }
            )
            if name in {"reminder_set"}:
                mark_intent(state, "read_note")

    FUSE["state"] = "IDLE"
    FUSE["last_engaged"] = time.time()
    if who != "unknown":
        FUSE["greeted"][who] = FUSE["last_engaged"]

    return {"ok": True, "decided": True, "who": who, "intent": chosen_intent}


@app.post("/guard/ack_cancel")
def guard_ack_cancel():
    FUSE["state"] = "IDLE"
    FUSE["last_engaged"] = time.time()
    _journal("guard", "user_ack", {})
    return {"ok": True}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=LISTEN_PORT)
