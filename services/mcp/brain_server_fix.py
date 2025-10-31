from fastapi import FastAPI, Query, Response
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta, timezone
import os, sqlite3, json, re

app = FastAPI(title="Boyan Brain (fix)")

DB_PATH = os.getenv("ASSIST_DB", os.path.join(os.path.dirname(__file__), "storage", "assist.db"))
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

def get_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn

def init_db():
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS notes(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        text TEXT NOT NULL,
        tags TEXT,
        created_ts TEXT NOT NULL
    )""")
    cur.execute("""CREATE TABLE IF NOT EXISTS reminders(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        text TEXT NOT NULL,
        due_ts TEXT NOT NULL,
        created_ts TEXT NOT NULL
    )""")
    cur.execute("""CREATE TABLE IF NOT EXISTS ocr_logs(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        device_id TEXT,
        cam TEXT,
        text TEXT,
        lang TEXT,
        ts TEXT NOT NULL
    )""")
    conn.commit(); conn.close()

@app.on_event("startup")
def _startup():
    init_db()

@app.get("/health")
def health():
    try:
        conn = get_db(); conn.execute("SELECT 1"); conn.close()
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "err": str(e)}

class NoteIn(BaseModel):
    text: str
    tags: Optional[List[str]] = None

@app.post("/notes/add")
def notes_add(payload: NoteIn):
    conn = get_db(); cur = conn.cursor()
    tags = json.dumps(payload.tags or [], ensure_ascii=False)
    ts = datetime.now(timezone.utc).isoformat()
    cur.execute("INSERT INTO notes(text,tags,created_ts) VALUES(?,?,?)",
                (payload.text, tags, ts))
    conn.commit()
    nid = cur.lastrowid
    conn.close()
    return {"ok": True, "id": nid}

@app.get("/notes/list")
def notes_list(q: Optional[str] = Query(default=None)):
    conn = get_db(); cur = conn.cursor()
    if q:
        cur.execute("SELECT id,text,tags,created_ts FROM notes WHERE text LIKE ? ORDER BY id DESC",
                    (f"%{q}%",))
    else:
        cur.execute("SELECT id,text,tags,created_ts FROM notes ORDER BY id DESC")
    rows = [dict(r) for r in cur.fetchall()]
    for r in rows:
        try:
            r["tags"] = json.loads(r.get("tags") or "[]")
        except Exception:
            r["tags"] = []
    conn.close()
    return {"ok": True, "items": rows}

class NLIn(BaseModel):
    utterance: str
    text: str

def parse_due_ts(utterance: str):
    now = datetime.now(timezone.utc)
    u = utterance.strip()
    m = re.search(r"(\d+)\s*分钟后", u) or re.search(r"(\d+)\s*分后", u)
    if m: return now + timedelta(minutes=int(m.group(1)))
    m = re.search(r"(\d+)\s*小时后", u)
    if m: return now + timedelta(hours=int(m.group(1)))
    m = re.search(r"明天\s*(\d{1,2}):(\d{2})", u)
    if m:
        h, mm = int(m.group(1)), int(m.group(2))
        return (now + timedelta(days=1)).replace(hour=h, minute=mm, second=0, microsecond=0)
    m = re.search(r"\b(\d{1,2}):(\d{2})\b", u)
    if m:
        h, mm = int(m.group(1)), int(m.group(2))
        tgt = now.replace(hour=h, minute=mm, second=0, microsecond=0)
        return tgt if tgt > now else tgt + timedelta(days=1)
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})\s+(\d{1,2}):(\d{2})", u)
    if m:
        y, mo, d, h, mm = map(int, m.groups())
        return datetime(y, mo, d, h, mm, tzinfo=timezone.utc)
    return now + timedelta(minutes=10)

@app.post("/reminders/parse_create")
def reminders_parse_create(nl: NLIn):
    due = parse_due_ts(nl.utterance)
    conn = get_db(); cur = conn.cursor()
    cur.execute("INSERT INTO reminders(text,due_ts,created_ts) VALUES(?,?,?)",
                (nl.text, due.isoformat(), datetime.now(timezone.utc).isoformat()))
    conn.commit()
    rid = cur.lastrowid
    conn.close()
    return {"ok": True, "id": rid, "due_ts": due.isoformat()}

@app.get("/reminders/list")
def reminders_list():
    conn = get_db(); cur = conn.cursor()
    cur.execute("SELECT id,text,due_ts,created_ts FROM reminders ORDER BY due_ts ASC")
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return {"ok": True, "items": rows}

@app.get("/reminders/export/ics")
def export_ics():
    conn = get_db(); cur = conn.cursor()
    cur.execute("SELECT id,text,due_ts FROM reminders ORDER BY due_ts ASC")
    rows = cur.fetchall(); conn.close()

    def esc(s): return (s or "").replace("\\","\\\\").replace(",","\\,").replace(";","\\;").replace("\n","\\n")
    lines = ["BEGIN:VCALENDAR","VERSION:2.0","PRODID:-//Boyan//MCP Brain//CN","CALSCALE:GREGORIAN","METHOD:PUBLISH"]
    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    for r in rows:
        rid, text, due = r["id"], r["text"], r["due_ts"]
        try:
            dt = datetime.fromisoformat(due.replace("Z","+00:00"))
            dt_utc = dt.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            lines += ["BEGIN:VEVENT", f"UID:rem-{rid}@boyan", f"DTSTAMP:{now}", f"DTSTART:{dt_utc}", f"SUMMARY:{esc(text)}", "END:VEVENT"]
        except Exception:
            continue
    return Response(content="\r\n".join(lines)+ "\r\n", media_type="text/calendar; charset=utf-8")

class EdgeEvent(BaseModel):
    type: str
    device_id: str
    ts: str
    cam: str
    topic: str
    data: Dict[str, Any] = Field(default_factory=dict)

@app.post("/edge/event")
def edge_event(ev: EdgeEvent):
    actions = []
    if ev.topic == "vision/person":
        conf = ev.data.get("confidence")
        actions.append({"action":"notify","title":"检测到人员",
                        "text":f"cam={ev.cam}, confidence={conf}",
                        "device_id":ev.device_id,"ts":ev.ts})
    elif ev.topic == "vision/face/verified":
        name = ev.data.get("name","<unknown>")
        score = ev.data.get("score")
        actions.append({"action":"greet","title":"已识别人脸",
                        "text":f"欢迎 {name} (score={score})",
                        "device_id":ev.device_id,"ts":ev.ts})
    elif ev.topic == "vision/ocr":
        conn = get_db(); cur = conn.cursor()
        cur.execute("INSERT INTO ocr_logs(device_id,cam,text,lang,ts) VALUES(?,?,?,?,?)",
                    (ev.device_id, ev.cam, ev.data.get("text",""), ev.data.get("lang",""), ev.ts))
        conn.commit(); conn.close()
        actions.append({"action":"store_ocr","title":"OCR文本入库",
                        "len":len(ev.data.get("text","")),
                        "preview":ev.data.get("text","")[:50],
                        "device_id":ev.device_id,"ts":ev.ts})
    return {"ok": True, "skipped": len(actions)==0, "actions": actions}
