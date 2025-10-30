import os
import io
import json
import base64
import time
import threading
import subprocess
import tempfile
import re
import collections
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict, Any

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dateutil import parser as dtparser
from PIL import Image
import numpy as np
import cv2
from dateparser.search import search_dates
import pytz

OCR_BASE = os.environ.get("OCR_BASE", "http://127.0.0.1:8000")
GATE_PORT = int(os.environ.get("GATE_PORT", "8787"))
STORE_DIR = os.path.join(os.path.dirname(__file__), "storage")
TMP_DIR = os.path.join(os.path.dirname(__file__), "tmp")
VOICE_DEF = os.environ.get("TTS_VOICE", "")
VOLUME_DEF = int(os.environ.get("TTS_VOLUME", "80"))
YUNET_PATH = os.environ.get(
    "YUNET_MODEL",
    os.path.join(os.path.dirname(__file__), "models", "opencv_zoo", "face_detection_yunet_2023mar.onnx"),
)
SFACE_PATH = os.environ.get(
    "SFACE_MODEL",
    os.path.join(os.path.dirname(__file__), "models", "opencv_zoo", "face_recognition_sface_2021dec.onnx"),
)
COSINE_OK = float(os.environ.get("FACE_COSINE_OK", "0.55"))
TW_TZ = pytz.timezone(os.environ.get("LOCAL_TZ", "Asia/Taipei"))

os.makedirs(STORE_DIR, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)

NOTES_PATH = os.path.join(STORE_DIR, "notes.json")
REMINDERS_PATH = os.path.join(STORE_DIR, "reminders.json")
FACES_PATH = os.path.join(STORE_DIR, "faces.json")
JOURNAL_PATH = os.path.join(STORE_DIR, "journal.jsonl")

_ASR_MODEL = None
_WAKE_MODEL = None
_WAKE_THREAD: threading.Thread | None = None
_AUDIO_THREAD: threading.Thread | None = None
_WAKE_STOP = threading.Event()
_WAKE_CFG = {
    "keywords": [
        s.strip()
        for s in os.environ.get("WAKE_KEYWORDS", "小智,小智同学,hi xiaozhi").split(",")
        if s.strip()
    ],
    "min_interval": float(os.environ.get("WAKE_MIN_INTERVAL_SEC", "20")),
    "win_sec": float(os.environ.get("WAKE_WINDOW_SEC", "1.6")),
    "step_sec": float(os.environ.get("WAKE_STEP_SEC", "0.8")),
    "conf": float(os.environ.get("WAKE_CONFIDENCE", "0.60")),
}
_WAKE_LAST_TS = 0.0
AUDIO_EVT = {"last_bang": 0.0}
DISTRESS_WORDS = ["救命", "好痛", "疼", "help", "help me", "着火", "火灾"]


def _load_json(path: str, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def _save_json(path: str, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


NOTES = _load_json(NOTES_PATH, [])
REMINDERS = _load_json(REMINDERS_PATH, [])
FACES = _load_json(FACES_PATH, {"people": []})
LOCK = threading.Lock()

app = FastAPI(title="MCP Gateway + Face", version="0.2")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class OCRIn(BaseModel):
    image_b64: str
    mime: Optional[str] = "image/png"
    roi: Optional[List[int]] = None


class OCROut(BaseModel):
    text: str
    raw: Dict[str, Any]


class SpeakIn(BaseModel):
    text: str
    voice: Optional[str] = None
    volume: Optional[int] = None


class NoteIn(BaseModel):
    kind: str
    text: str
    when: Optional[str] = None


class SearchIn(BaseModel):
    q: Optional[str] = None


class ReminderIn(BaseModel):
    note_id: Optional[str] = None
    text: Optional[str] = None
    when: str
    repeat_cron: Optional[str] = None


class IDIn(BaseModel):
    id: str


class FaceDetectIn(BaseModel):
    image_b64: str


class FaceEncodeIn(BaseModel):
    image_b64: str
    box: Optional[List[float]] = None


class FaceRegisterIn(BaseModel):
    name: str
    image_b64: str


class FaceSearchIn(BaseModel):
    image_b64: str
    top_k: Optional[int] = 3
    thr: Optional[float] = None


class ImportAllIn(BaseModel):
    notes: list
    reminders: list
    faces: dict


class ParseTimeIn(BaseModel):
    text: str
    ref_utc_iso: Optional[str] = None


class IngestNotesIn(BaseModel):
    text: str
    source: Optional[str] = "ocr"
    default_due_minutes: Optional[int] = None


class JournalAppendIn(BaseModel):
    source: str
    text: str
    meta: Optional[Dict[str, Any]] = None


class JournalSearchIn(BaseModel):
    q: Optional[str] = None
    since_iso: Optional[str] = None
    until_iso: Optional[str] = None
    limit: Optional[int] = 100


class AsrTranscribeIn(BaseModel):
    audio_b64: Optional[str] = None
    audio_url: Optional[str] = None
    lang: Optional[str] = None
    prompt: Optional[str] = None
    use_vad: Optional[bool] = True
    beam_size: Optional[int] = 5
    vad_filter: Optional[bool] = True
    temperature: Optional[float] = 0.0


class AsrSegment(BaseModel):
    start: float
    end: float
    text: str


class AsrTranscribeOut(BaseModel):
    text: str
    language: str
    segments: List[AsrSegment]
    duration_sec: Optional[float] = None


class AsrIngestIn(AsrTranscribeIn):
    default_due_minutes: Optional[int] = None
    source: Optional[str] = "voice"


class AsrIngestOut(BaseModel):
    asr: AsrTranscribeOut
    ingest_result: Dict[str, Any]


class WakeConfigIn(BaseModel):
    keywords: Optional[List[str]] = None
    min_interval: Optional[float] = None
    win_sec: Optional[float] = None
    step_sec: Optional[float] = None


def _to_data_uri(mime: str, blob: bytes) -> str:
    return f"data:{mime};base64," + base64.b64encode(blob).decode("ascii")


def _extract_text_from_ocr_response(obj: Dict[str, Any]) -> str:
    texts: List[str] = []
    if obj.get("output_text"):
        texts.append(str(obj["output_text"]))
    for entry in obj.get("output", []) or []:
        for content in entry.get("content", []) or []:
            if isinstance(content, dict) and content.get("text"):
                texts.append(content["text"])
    for choice in obj.get("choices", []) or []:
        message = choice.get("message", {})
        content = message.get("content")
        if isinstance(content, list):
            for c in content:
                if isinstance(c, dict) and c.get("text"):
                    texts.append(c["text"])
        elif isinstance(content, str):
            texts.append(content)
    text = "\n".join([t for t in texts if t]).strip()
    if re.fullmatch(r"\\\\\[(.|\n)+\\\\\]", text):
        text = re.sub(r"^\\\\\[(.*)\\\\\]$", r"\1", text, flags=re.S)
    if text and all(ch == "?" or ch.isspace() for ch in text):
        return ""
    return text


def _speak_windows_sapi(text: str, voice: Optional[str], volume: Optional[int]) -> bool:
    voice_name = voice or VOICE_DEF
    vol = volume if volume is not None else VOLUME_DEF
    os.makedirs(TMP_DIR, exist_ok=True)
    tf = tempfile.NamedTemporaryFile(delete=False, dir=TMP_DIR, suffix=".txt")
    tf.write(text.encode("utf-8"))
    tf.flush()
    tf.close()
    ps_script = rf'''
Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
try {{ if ("{voice_name}" -ne "") {{ $s.SelectVoice("{voice_name}") }} }} catch {{ }}
$s.Volume = {max(0, min(100, int(vol)))}
$t = Get-Content -Raw -Encoding UTF8 "{tf.name}"
$s.Speak($t)
'''
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps_script],
            capture_output=True,
            text=True,
            timeout=120,
        )
        ok = result.returncode == 0
    except Exception:
        ok = False
    finally:
        try:
            os.remove(tf.name)
        except OSError:
            pass
    return ok


_face_detector = None
_face_recognizer = None


def _ensure_face_models():
    global _face_detector, _face_recognizer
    if _face_detector is None:
        _face_detector = cv2.FaceDetectorYN_create(
            YUNET_PATH,
            "",
            (320, 320),
            0.6,
            0.3,
            5000,
        )
    if _face_recognizer is None:
        _face_recognizer = cv2.FaceRecognizerSF_create(SFACE_PATH, "")


def _b64_to_bgr(image_b64: str) -> np.ndarray:
    data = base64.b64decode(image_b64)
    arr = np.frombuffer(data, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _detect_faces(img: np.ndarray) -> np.ndarray:
    height, width = img.shape[:2]
    _face_detector.setInputSize((width, height))
    _, faces = _face_detector.detect(img)
    if faces is None:
        return np.empty((0, 15), dtype=np.float32)
    return faces


def _encode_face(img: np.ndarray, box: List[float]) -> np.ndarray:
    face = np.array(box, dtype=np.float32)
    aligned = _face_recognizer.alignCrop(img, face)
    feat = _face_recognizer.feature(aligned)
    feat = feat / (np.linalg.norm(feat) + 1e-12)
    return feat.astype(np.float32)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


@app.get("/health")
def health():
    ocr_ok = False
    try:
        with httpx.Client(timeout=2) as cli:
            response = cli.get(f"{OCR_BASE}/v1/health")
            ocr_ok = response.status_code == 200 and "ok" in response.text.lower()
    except Exception:
        ocr_ok = False
    face_ok = os.path.exists(YUNET_PATH) and os.path.exists(SFACE_PATH)
    return {"ok": True, "ocr": ocr_ok, "face": face_ok}


@app.post("/tools/ocr/read", response_model=OCROut)
def ocr_read(inp: OCRIn):
    raw = base64.b64decode(inp.image_b64)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    if inp.roi and len(inp.roi) == 4:
        x, y, w, h = inp.roi
        x = max(0, x)
        y = max(0, y)
        w = max(1, w)
        h = max(1, h)
        img = img.crop((x, y, x + w, y + h))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    data_uri = _to_data_uri("image/png", buf.getvalue())
    payload = {
        "model": "deepseek-ocr",
        "input": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": "只输出图片中的原始文本；不要解释；不要 LaTeX；尽量保持原文原样。",
                    },
                    {"type": "input_image", "image_url": data_uri},
                ],
            }
        ],
        "max_output_tokens": 512,
    }
    with httpx.Client(timeout=120) as cli:
        response = cli.post(f"{OCR_BASE}/v1/responses", json=payload)
        response.raise_for_status()
        obj = response.json()
    text = _extract_text_from_ocr_response(obj)
    return {"text": text, "raw": obj}


@app.post("/tts/speak")
def tts_speak(inp: SpeakIn):
    ok = _speak_windows_sapi(inp.text, inp.voice, inp.volume)
    return {"ok": ok}


@app.post("/mem/write")
def mem_write(inp: NoteIn):
    with LOCK:
        note_id = f"note_{int(time.time() * 1000)}"
        record = {"id": note_id, "kind": inp.kind, "text": inp.text, "when": inp.when}
        NOTES.append(record)
        _save_json(NOTES_PATH, NOTES)
    return {"id": note_id, "ok": True}


@app.post("/mem/search")
def mem_search(inp: SearchIn):
    query = (inp.q or "").strip()
    with LOCK:
        if not query:
            results = NOTES[-50:]
        else:
            results = [n for n in NOTES if query.lower() in n.get("text", "").lower()]
    return {"results": results}


@app.post("/reminder/set")
def reminder_set(inp: ReminderIn):
    with LOCK:
        reminder_id = f"rem_{int(time.time() * 1000)}"
        record = {
            "id": reminder_id,
            "note_id": inp.note_id,
            "text": inp.text,
            "when": dtparser.parse(inp.when).astimezone(timezone.utc).isoformat(),
            "repeat_cron": inp.repeat_cron,
            "fired": False,
            "cancelled": False,
        }
        REMINDERS.append(record)
        _save_json(REMINDERS_PATH, REMINDERS)
    return {"id": reminder_id, "ok": True}


@app.get("/reminder/list")
def reminder_list():
    with LOCK:
        return {"items": REMINDERS}


@app.post("/reminder/cancel")
def reminder_cancel(inp: IDIn):
    with LOCK:
        for item in REMINDERS:
            if item["id"] == inp.id:
                item["cancelled"] = True
                _save_json(REMINDERS_PATH, REMINDERS)
                return {"ok": True}
    return {"ok": False, "error": "not_found"}


def _parse_times(text: str, ref_dt_utc: Optional[datetime] = None) -> List[Dict[str, Any]]:
    if not ref_dt_utc:
        ref_dt_utc = datetime.now(timezone.utc)
    settings = {
        "TIMEZONE": TW_TZ.zone,
        "RETURN_AS_TIMEZONE_AWARE": True,
        "PREFER_DATES_FROM": "future",
        "RELATIVE_BASE": ref_dt_utc.astimezone(TW_TZ).replace(tzinfo=None),
        "PARSERS": [
            "relative-time",
            "custom-formats",
            "absolute-time",
            "no-spaces-time",
        ],
        "SKIP_TOKENS": ["号", "日"],
    }
    results: List[Dict[str, Any]] = []
    try:
        hits = search_dates(text, languages=["zh", "en"], settings=settings) or []
        for phrase, dt_local in hits:
            if dt_local.tzinfo is None:
                dt_local = TW_TZ.localize(dt_local)
            dt_utc = dt_local.astimezone(timezone.utc)
            grain = "second"
            if dt_local.minute == 0 and ("点" in phrase or ":" not in phrase):
                grain = "hour"
            elif any(token in phrase for token in ["上午", "下午", "早上", "晚上"]):
                grain = "hour"
            elif any(token in phrase for token in ["明天", "今天", "后天", "周", "星期", "礼拜"]):
                grain = "day"
            results.append(
                {
                    "phrase": phrase,
                    "local": dt_local.isoformat(),
                    "iso": dt_utc.isoformat(),
                    "grain": grain,
                    "confidence": 0.85,
                }
            )
    except Exception:
        pass
    seen: set[str] = set()
    uniq: List[Dict[str, Any]] = []
    for item in results:
        iso = item["iso"]
        if iso in seen:
            continue
        uniq.append(item)
        seen.add(iso)
    return uniq


@app.post("/nlp/parse_time")
def nlp_parse_time(inp: ParseTimeIn):
    ref = dtparser.parse(inp.ref_utc_iso).astimezone(timezone.utc) if inp.ref_utc_iso else None
    items = _parse_times(inp.text.strip(), ref_dt_utc=ref)
    return {"items": items}


def _normalize_lines(text: str) -> List[str]:
    raw_lines = text.replace("\r", "").split("\n")
    lines: List[str] = []
    for raw in raw_lines:
        raw = raw.strip(" 　\t;；.。•·-—*")
        if not raw:
            continue
        segments = re.split(r"[;；•·]+", raw)
        for seg in segments:
            seg = seg.strip(" 　\t-—*.")
            if seg:
                lines.append(seg)
    return lines


def _dedup_key(text: str) -> str:
    simplified = re.sub(r"\s+", "", text)
    simplified = re.sub(r"[，。；、,.!！?？]+", "", simplified)
    return simplified[:64]


@app.post("/notes/ingest")
def notes_ingest(inp: IngestNotesIn):
    text = (inp.text or "").strip()
    if not text:
        return {"ok": False, "error": "empty_text"}

    lines = _normalize_lines(text)
    created: List[Dict[str, Any]] = []
    now_utc = datetime.now(timezone.utc)

    with LOCK:
        recent = NOTES[-500:]
        recent_keys = {_dedup_key(item.get("text", "")) for item in recent}

        for line in lines:
            key = _dedup_key(line)
            if key in recent_keys:
                created.append({"text": line, "skipped": "duplicate_recent"})
                continue

            times = _parse_times(line, now_utc)
            when_iso = times[0]["iso"] if times else None
            if when_iso is None and inp.default_due_minutes:
                when_iso = (now_utc + timedelta(minutes=int(inp.default_due_minutes))).isoformat()

            note_id = f"note_{int(time.time() * 1000)}"
            record = {
                "id": note_id,
                "kind": "todo",
                "text": line,
                "when": when_iso,
                "source": inp.source,
                "created_at": now_utc.isoformat(),
            }
            NOTES.append(record)
            _save_json(NOTES_PATH, NOTES)

            reminder_id = None
            if when_iso:
                reminder_id = f"rem_{int(time.time() * 1000)}"
                REMINDERS.append(
                    {
                        "id": reminder_id,
                        "note_id": note_id,
                        "text": line,
                        "when": when_iso,
                        "repeat_cron": None,
                        "fired": False,
                        "cancelled": False,
                    }
                )
                _save_json(REMINDERS_PATH, REMINDERS)

            created.append(
                {
                    "id": note_id,
                    "text": line,
                    "when": when_iso,
                    "reminder_id": reminder_id,
                }
            )
            recent_keys.add(key)

    return {"ok": True, "items": created}


@app.post("/journal/append")
def journal_append(inp: JournalAppendIn):
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "source": inp.source,
        "text": inp.text,
        "meta": inp.meta or {},
    }
    os.makedirs(os.path.dirname(JOURNAL_PATH), exist_ok=True)
    with open(JOURNAL_PATH, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {"ok": True}


@app.post("/journal/search")
def journal_search(inp: JournalSearchIn):
    query = (inp.q or "").strip()
    since = dtparser.parse(inp.since_iso).astimezone(timezone.utc) if inp.since_iso else None
    until = dtparser.parse(inp.until_iso).astimezone(timezone.utc) if inp.until_iso else None
    items: List[Dict[str, Any]] = []
    if not os.path.exists(JOURNAL_PATH):
        return {"items": items}
    with open(JOURNAL_PATH, "r", encoding="utf-8") as handle:
        lines = handle.readlines()
    for line in reversed(lines):
        try:
            entry = json.loads(line)
            ts = dtparser.parse(entry.get("ts")).astimezone(timezone.utc)
            if since and ts < since:
                continue
            if until and ts > until:
                continue
            if query and query not in entry.get("text", ""):
                continue
            items.append(entry)
            if len(items) >= int(inp.limit or 100):
                break
        except Exception:
            continue
    return {"items": items}


# ====================== ASR & 语音入库 ======================


def _ensure_asr_model():
    global _ASR_MODEL
    if _ASR_MODEL is None:
        from faster_whisper import WhisperModel

        size = os.environ.get("ASR_MODEL_SIZE", "small")
        compute = os.environ.get("ASR_COMPUTE", "int8")
        device = os.environ.get("ASR_DEVICE", "cpu")
        _ASR_MODEL = WhisperModel(size, device=device, compute_type=compute)
    return _ASR_MODEL


def _ensure_wake_model():
    global _WAKE_MODEL
    if _WAKE_MODEL is None:
        from faster_whisper import WhisperModel

        size = os.environ.get("WAKE_ASR_SIZE", os.environ.get("ASR_MODEL_SIZE", "tiny"))
        compute = os.environ.get("WAKE_ASR_COMPUTE", "int8")
        device = os.environ.get("ASR_DEVICE", "cpu")
        _WAKE_MODEL = WhisperModel(size, device=device, compute_type=compute)
    return _WAKE_MODEL


def _b64_to_temp_wav(b64_data: str) -> str:
    raw = base64.b64decode(b64_data)
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    with open(path, "wb") as handle:
        handle.write(raw)
    return path


def _maybe_vad_trim(in_path: str) -> str:
    if not os.environ.get("ASR_USE_VAD", "1"):
        return in_path
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".wav")
    os.close(tmp_fd)
    try:
        cmd = ["ffmpeg", "-y", "-i", in_path, "-ac", "1", "-ar", "16000", tmp_path]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return tmp_path
    except subprocess.CalledProcessError:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return in_path


@app.post("/asr/transcribe", response_model=AsrTranscribeOut)
def asr_transcribe(inp: AsrTranscribeIn):
    if not inp.audio_b64 and not inp.audio_url:
        raise HTTPException(status_code=400, detail="audio_b64 or audio_url is required")

    wav_path: Optional[str] = None
    temp_paths: List[str] = []
    try:
        if inp.audio_b64:
            wav_path = _b64_to_temp_wav(inp.audio_b64)
        else:
            fd, tmp = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            cmd = ["ffmpeg", "-y", "-i", inp.audio_url, "-ac", "1", "-ar", "16000", tmp]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            wav_path = tmp
        temp_paths.append(wav_path)

        if inp.use_vad and wav_path:
            trimmed = _maybe_vad_trim(wav_path)
            if trimmed != wav_path:
                temp_paths.append(trimmed)
                wav_path = trimmed

        model = _ensure_asr_model()
        segments, info = model.transcribe(
            wav_path,
            language=inp.lang,
            beam_size=inp.beam_size or 5,
            vad_filter=bool(inp.vad_filter),
            temperature=inp.temperature or 0.0,
            initial_prompt=inp.prompt or None,
        )

        items: List[AsrSegment] = []
        full_text: List[str] = []
        duration = 0.0
        for segment in segments:
            items.append(AsrSegment(start=segment.start, end=segment.end, text=segment.text))
            full_text.append(segment.text)
            if segment.end:
                duration = max(duration, float(segment.end))

        return {
            "text": "".join(full_text).strip(),
            "language": info.language or (inp.lang or "unknown"),
            "segments": items,
            "duration_sec": duration or None,
        }
    except subprocess.CalledProcessError as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail="ffmpeg failed") from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        for path in temp_paths:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass


@app.post("/asr/transcribe_and_ingest", response_model=AsrIngestOut)
def asr_transcribe_and_ingest(inp: AsrIngestIn):
    asr_payload = asr_transcribe(inp)
    text = asr_payload["text"] if isinstance(asr_payload, dict) else asr_payload.text
    try:
        with httpx.Client(timeout=3) as cli:
            cli.post(
                JOURNAL_APPEND_URL,
                json={
                    "source": inp.source or "voice",
                    "text": text,
                    "meta": {
                        "language": asr_payload.get("language")
                        if isinstance(asr_payload, dict)
                        else asr_payload.language
                    },
                },
            )
    except Exception:
        pass

    payload = {
        "text": text,
        "source": inp.source or "voice",
        "default_due_minutes": inp.default_due_minutes,
    }
    lowered = text.lower()
    if any(word.lower() in lowered for word in DISTRESS_WORDS):
        _post_audio_event("audio/distress_keyword", {"text": text})
    try:
        with httpx.Client(timeout=15) as cli:
            resp = cli.post(INGEST_NOTES_URL, json=payload)
            resp.raise_for_status()
            ingest_info = resp.json()
    except Exception as exc:  # noqa: BLE001
        ingest_info = {"ok": False, "error": str(exc)}

    return {"asr": asr_payload, "ingest_result": ingest_info}


# ====================== Wake Word & Audio Anomaly ======================


def _normalize_text(value: str) -> str:
    compact = re.sub(r"[\s\u3000]+", "", value.lower())
    return re.sub(r"[、，。；：！!？?\-—_~^··…,.]", "", compact)


def _contains_keyword(transcript: str, keywords: List[str]) -> bool:
    normalized = _normalize_text(transcript)
    return any(_normalize_text(word) in normalized for word in keywords)


def _post_wake_event(transcript: str, score: float) -> None:
    payload = {"topic": "wake/keyword", "data": {"text": transcript, "score": score}}
    try:
        with httpx.Client(timeout=2) as cli:
            cli.post("http://127.0.0.1:8768/events/publish", json=payload)
    except Exception:
        pass
    try:
        with httpx.Client(timeout=2) as cli:
            cli.post(
                JOURNAL_APPEND_URL,
                json={"source": "wake", "text": transcript, "meta": {"score": score}},
            )
    except Exception:
        pass


def _wake_loop() -> None:
    global _WAKE_LAST_TS
    import sounddevice as sd
    import soundfile as sf

    rate = 16000
    chunk = int(rate * max(0.1, min(_WAKE_CFG["step_sec"], 1.0)))
    buffer: collections.deque[np.ndarray] = collections.deque(
        maxlen=int(rate * _WAKE_CFG["win_sec"])
    )
    model = _ensure_wake_model()

    def _callback(indata, frames, time_info, status):  # noqa: ANN001
        if status:
            return
        buffer.extend(indata[:, 0].copy())

    with sd.InputStream(samplerate=rate, channels=1, dtype="int16", callback=_callback, blocksize=chunk):
        while not _WAKE_STOP.is_set():
            sd.sleep(int(_WAKE_CFG["step_sec"] * 1000))
            if len(buffer) < rate * 0.4:
                continue
            arr = np.array(buffer, dtype=np.int16)
            if np.abs(arr).mean() < 90:
                continue
            fd, tmp_path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            try:
                sf.write(tmp_path, arr.astype(np.float32) / 32768.0, rate, subtype="PCM_16")
                segments, _ = model.transcribe(
                    tmp_path,
                    language="zh",
                    beam_size=1,
                    temperature=0.0,
                    vad_filter=True,
                )
                transcript = "".join(segment.text for segment in segments).strip()
            finally:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            if not transcript:
                continue
            if _contains_keyword(transcript, _WAKE_CFG["keywords"]):
                now = time.time()
                if now - _WAKE_LAST_TS < _WAKE_CFG["min_interval"]:
                    continue
                _WAKE_LAST_TS = now
                _post_wake_event(transcript, 1.0)


@app.post("/wake/start")
def wake_start():
    global _WAKE_THREAD
    if _WAKE_THREAD and _WAKE_THREAD.is_alive():
        return {"ok": True, "running": True}
    _WAKE_STOP.clear()
    _WAKE_THREAD = threading.Thread(target=_wake_loop, daemon=True)
    _WAKE_THREAD.start()
    return {"ok": True, "running": True}


@app.post("/wake/stop")
def wake_stop():
    _WAKE_STOP.set()
    return {"ok": True, "running": False}


@app.post("/wake/config")
def wake_config(inp: WakeConfigIn):
    if inp.keywords is not None:
        _WAKE_CFG["keywords"] = inp.keywords
    if inp.min_interval is not None:
        _WAKE_CFG["min_interval"] = float(inp.min_interval)
    if inp.win_sec is not None:
        _WAKE_CFG["win_sec"] = max(0.4, float(inp.win_sec))
    if inp.step_sec is not None:
        _WAKE_CFG["step_sec"] = max(0.2, float(inp.step_sec))
    return {"ok": True, "config": _WAKE_CFG}


def _post_audio_event(topic: str, data: Dict[str, Any]) -> None:
    try:
        with httpx.Client(timeout=2) as cli:
            cli.post("http://127.0.0.1:8768/events/publish", json={"topic": topic, "data": data})
    except Exception:
        pass


def _audio_anomaly_loop() -> None:
    import sounddevice as sd

    rate = 16000
    chunk = int(rate * 0.5)
    cache_until = 0.0
    cached_cfg: Dict[str, Any] | None = None
    with sd.InputStream(samplerate=rate, channels=1, dtype="int16", blocksize=chunk) as stream:
        while not _WAKE_STOP.is_set():
            data, _ = stream.read(chunk)
            buf = data[:, 0].astype(np.float32)
            rms = float(np.sqrt(np.mean(buf * buf)))
            db = 20 * np.log10(max(rms, 1e-6)) + 90
            now = time.time()
            if now >= cache_until:
                try:
                    with httpx.Client(timeout=2) as cli:
                        cached_cfg = cli.get("http://127.0.0.1:8768/policy/profile").json()["cfg"]
                except Exception:
                    cached_cfg = None
                cache_until = now + 5
            threshold = 75.0
            gap = 20.0
            if cached_cfg:
                audio_cfg = cached_cfg.get("audio", {})
                threshold = float(audio_cfg.get("bang_db", threshold))
                gap = float(audio_cfg.get("bang_min_gap", gap))
            if db >= threshold and now - AUDIO_EVT["last_bang"] >= gap:
                AUDIO_EVT["last_bang"] = now
                _post_audio_event("audio/loud_bang", {"db": round(db, 1), "ts": now})


@app.post("/audio_anomaly/start")
def audio_anomaly_start():
    global _AUDIO_THREAD
    if _AUDIO_THREAD and _AUDIO_THREAD.is_alive():
        return {"ok": True}
    _WAKE_STOP.clear()
    _AUDIO_THREAD = threading.Thread(target=_audio_anomaly_loop, daemon=True)
    _AUDIO_THREAD.start()
    return {"ok": True}


@app.get("/face/health")
def face_health():
    return {"ok": os.path.exists(YUNET_PATH) and os.path.exists(SFACE_PATH)}


@app.post("/face/detect")
def face_detect(inp: FaceDetectIn):
    _ensure_face_models()
    img = _b64_to_bgr(inp.image_b64)
    faces = _detect_faces(img)
    boxes = []
    for face in faces:
        x, y, w, h, score = face[:5].tolist()
        boxes.append({"box": [x, y, w, h], "score": score})
    return {"count": len(boxes), "faces": boxes}


@app.post("/face/encode")
def face_encode(inp: FaceEncodeIn):
    _ensure_face_models()
    img = _b64_to_bgr(inp.image_b64)
    faces = _detect_faces(img)
    if inp.box is None:
        if len(faces) == 0:
            return {"ok": False, "error": "no_face"}
        box = faces[0][:4].tolist()
    else:
        box = inp.box
    vector = _encode_face(img, box)
    return {"ok": True, "box": box, "vector": vector.tolist()}


@app.post("/face/register")
def face_register(inp: FaceRegisterIn):
    _ensure_face_models()
    img = _b64_to_bgr(inp.image_b64)
    faces = _detect_faces(img)
    if len(faces) == 0:
        return {"ok": False, "error": "no_face"}
    box = faces[0][:4].tolist()
    vector = _encode_face(img, box).tolist()
    with LOCK:
        for person in FACES["people"]:
            if person["name"] == inp.name:
                person["vectors"].append(vector)
                _save_json(FACES_PATH, FACES)
                return {"ok": True, "id": person["id"], "appended": True}
        person_id = f"p_{int(time.time() * 1000)}"
        FACES["people"].append({"id": person_id, "name": inp.name, "vectors": [vector], "meta": {}})
        _save_json(FACES_PATH, FACES)
        return {"ok": True, "id": person_id, "appended": False}


@app.post("/face/search")
def face_search(inp: FaceSearchIn):
    _ensure_face_models()
    img = _b64_to_bgr(inp.image_b64)
    faces = _detect_faces(img)
    if len(faces) == 0:
        return {"ok": True, "matches": []}
    box = faces[0][:4].tolist()
    query = _encode_face(img, box)
    thr = COSINE_OK if inp.thr is None else float(inp.thr)
    top_k = int(inp.top_k or 3)
    candidates: List[Dict[str, Any]] = []
    with LOCK:
        for person in FACES["people"]:
            best = -1.0
            for vec in person["vectors"]:
                score = _cosine(query, np.asarray(vec, dtype=np.float32))
                if score > best:
                    best = score
            candidates.append({"id": person["id"], "name": person["name"], "score": best})
    candidates.sort(key=lambda item: item["score"], reverse=True)
    matches = [item for item in candidates[:top_k] if item["score"] >= thr]
    return {"ok": True, "box": box, "matches": matches, "top1": matches[0] if matches else None}


@app.get("/face/list")
def face_list():
    with LOCK:
        return {
            "people": [
                {"id": person["id"], "name": person["name"], "shots": len(person["vectors"])}
                for person in FACES["people"]
            ]
        }


@app.post("/face/delete")
def face_delete(inp: IDIn):
    with LOCK:
        before = len(FACES["people"])
        FACES["people"] = [person for person in FACES["people"] if person["id"] != inp.id]
        if len(FACES["people"]) < before:
            _save_json(FACES_PATH, FACES)
            return {"ok": True}
    return {"ok": False, "error": "not_found"}


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _scheduler_loop():
    while True:
        try:
            due: List[Dict[str, Any]] = []
            with LOCK:
                now = _now_utc()
                changed = False
                for item in REMINDERS:
                    if item.get("cancelled"):
                        continue
                    dt = dtparser.parse(item["when"])
                    if dt <= now and not item.get("fired"):
                        due.append(item)
                        item["fired"] = True
                        changed = True
                if changed:
                    _save_json(REMINDERS_PATH, REMINDERS)
            for item in due:
                msg = item.get("text") or f"提醒：{item.get('note_id')}"
                _speak_windows_sapi(msg, None, None)
        except Exception:
            pass
        time.sleep(1)


def _start_scheduler():
    thread = threading.Thread(target=_scheduler_loop, daemon=True)
    thread.start()


_start_scheduler()


@app.get("/export/all")
def export_all():
    try:
        with LOCK:
            notes = _load_json(NOTES_PATH, [])
            reminders = _load_json(REMINDERS_PATH, [])
            faces = _load_json(FACES_PATH, {"people": []})
        return {
            "version": "0.2",
            "exported_at": datetime.now().isoformat(),
            "notes": notes,
            "reminders": reminders,
            "faces": faces,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/import/all")
def import_all(inp: ImportAllIn):
    try:
        with LOCK:
            _save_json(NOTES_PATH, inp.notes or [])
            _save_json(REMINDERS_PATH, inp.reminders or [])
            _save_json(FACES_PATH, inp.faces or {"people": []})
            NOTES[:] = _load_json(NOTES_PATH, [])
            REMINDERS[:] = _load_json(REMINDERS_PATH, [])
            FACES.clear()
            FACES.update(_load_json(FACES_PATH, {"people": []}))
        return {"ok": True}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/export/notes_reminders")
def export_notes_reminders():
    with LOCK:
        return {
            "notes": _load_json(NOTES_PATH, []),
            "reminders": _load_json(REMINDERS_PATH, []),
        }


@app.get("/export/faces")
def export_faces():
    with LOCK:
        return _load_json(FACES_PATH, {"people": []})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=GATE_PORT)
