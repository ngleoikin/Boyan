from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import numpy as np
from fastapi import FastAPI, HTTPException

try:  # Optional dependency required for file uploads.
    import multipart  # type: ignore  # noqa: F401
    from fastapi import File, UploadFile

    _MULTIPART_AVAILABLE = True
except Exception:  # pragma: no cover - dependency optional
    _MULTIPART_AVAILABLE = False

from common import config, db, schemas, utils

LOGGER = logging.getLogger(__name__)

app = FastAPI(title="Boyan OCR & Face Gateway", version="1.0.0")


class FaceEngine:
    """Wrapper around YuNet + SFace with a lightweight fallback."""

    def __init__(self) -> None:
        self.detector = None
        self.recognizer = None
        self.ready = False
        self._load_models()

    def _load_models(self) -> None:
        yunet_path = config.cfg(
            "YUNET_MODEL",
            str(Path(__file__).parent / "models" / "opencv_zoo" / "face_detection_yunet_2023mar.onnx"),
        )
        sface_path = config.cfg(
            "SFACE_MODEL",
            str(Path(__file__).parent / "models" / "opencv_zoo" / "face_recognition_sface_2021dec.onnx"),
        )

        try:
            import cv2  # type: ignore
        except ImportError as exc:
            LOGGER.warning("OpenCV 不可用，启用降级算法: %s", exc)
            self.detector = None
            self.recognizer = None
            self.ready = False
            return

        try:
            if yunet_path and Path(yunet_path).exists():
                self.detector = cv2.FaceDetectorYN.create(yunet_path, "", (320, 320))
            else:
                LOGGER.warning("YuNet 模型未找到，使用降级算法。")
            if sface_path and Path(sface_path).exists():
                self.recognizer = cv2.FaceRecognizerSF.create(sface_path, "")
            else:
                LOGGER.warning("SFace 模型未找到，使用降级算法。")
            if self.detector is not None and self.recognizer is not None:
                self.ready = True
        except Exception as exc:  # pragma: no cover - OpenCV specific
            LOGGER.exception("加载人脸模型失败: %s", exc)
            self.detector = None
            self.recognizer = None
            self.ready = False

    def _detect(self, image: np.ndarray) -> Optional[np.ndarray]:
        if self.detector is not None:
            h, w = image.shape[:2]
            self.detector.setInputSize((w, h))
            _, faces = self.detector.detect(image)
            if faces is None or len(faces) == 0:
                return None
            faces = faces.reshape(-1, faces.shape[-1])
            faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
            return faces[0]

        # Fallback: assume a single face occupying the frame centre.
        h, w = image.shape[:2]
        size = min(h, w)
        x = (w - size) / 2
        y = (h - size) / 2
        return np.array([x, y, size, size], dtype=np.float32)

    def _embed(self, image: np.ndarray, face: np.ndarray) -> Optional[np.ndarray]:
        if face is None:
            return None

        if self.recognizer is not None:
            try:
                import cv2  # type: ignore

                aligned = self.recognizer.alignCrop(image, face)
                feat = self.recognizer.feature(aligned)
            except Exception as exc:  # pragma: no cover - OpenCV specific
                LOGGER.warning("提取 SFace 特征失败: %s", exc)
                return None
        else:
            x1 = max(int(face[0]), 0)
            y1 = max(int(face[1]), 0)
            x2 = max(int(face[0] + face[2]), x1 + 1)
            y2 = max(int(face[1] + face[3]), y1 + 1)
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                crop = image
            gray = crop.mean(axis=2)
            resized = np.resize(gray, (32, 32)).astype(np.float32)
            feat = resized.flatten()

        norm = np.linalg.norm(feat)
        if norm > 0:
            feat = feat / norm
        return feat.astype(np.float32)

    def extract(self, image: np.ndarray) -> Optional[np.ndarray]:
        face = self._detect(image)
        return self._embed(image, face)


FACE_ENGINE = FaceEngine()


async def _call_ocr(image_b64: str, mime: str, prompt: str) -> tuple[str, Dict[str, Any]]:
    base_url = str(config.cfg("OCR_BASE", "http://127.0.0.1:8000")).rstrip("/")
    model_id = str(config.cfg("MODEL_ID", "deepseek-ocr"))
    api_key = config.cfg("OPENAI_API_KEY")
    url = f"{base_url}/v1/responses"

    payload = {
        "model": model_id,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {
                        "type": "input_image",
                        "image_base64": image_b64,
                        "mime_type": mime,
                    },
                ],
            }
        ],
    }

    headers = {"content-type": "application/json"}
    if api_key:
        headers["authorization"] = f"Bearer {api_key}"

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
    data = resp.json()
    text = _extract_ocr_text(data)
    return text, data


def _extract_ocr_text(payload: Dict[str, Any]) -> str:
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
            for key in ("text", "output_text", "content"):
                if key in value:
                    _collect(value[key])

    _collect(payload)
    if candidates:
        return candidates[0]
    raise ValueError("无法从 OCR 响应中解析文本")


async def _probe_ocr() -> bool:
    base_url = str(config.cfg("OCR_BASE", "http://127.0.0.1:8000")).rstrip("/")
    url = f"{base_url}/v1/health"
    try:
        async with httpx.AsyncClient(timeout=3) as client:
            resp = await client.get(url)
            return resp.status_code < 500
    except Exception:
        return False


def _store_face(name: str, embedding: np.ndarray) -> tuple[str, bool]:
    vector_blob = utils.encode_vector(embedding)
    created_ts = utils.isoformat_z(utils.utc_now())

    with db.get_conn() as conn:
        existing = conn.execute("SELECT id FROM faces WHERE name=?", (name,)).fetchone()
        if existing:
            conn.execute(
                "UPDATE faces SET vec=?, created_ts=? WHERE id=?",
                (vector_blob, created_ts, existing["id"]),
            )
            return existing["id"], False
        person_id = utils.generate_uid("p")
        conn.execute(
            "INSERT INTO faces(id, name, vec, created_ts) VALUES(?,?,?,?)",
            (person_id, name, vector_blob, created_ts),
        )
        return person_id, True


def _fetch_faces() -> List[Dict[str, Any]]:
    with db.get_conn() as conn:
        rows = conn.execute("SELECT id, name, vec FROM faces").fetchall()
        return [dict(r) for r in rows]


@app.on_event("startup")
def _startup() -> None:
    db.init_db()


@app.get("/health")
async def health() -> Dict[str, Any]:
    ocr_ok = await _probe_ocr()
    return {"ok": True, "ocr": ocr_ok, "face": FACE_ENGINE.ready}


@app.post("/ocr/from_data_uri", response_model=schemas.OCROut)
async def ocr_from_data_uri(payload: schemas.OCRDataURIRequest) -> schemas.OCROut:
    try:
        mime, data = utils.parse_data_uri(payload.data_uri)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    image_b64 = base64.b64encode(data).decode()
    text, raw = await _call_ocr(image_b64, mime, payload.prompt)
    return schemas.OCROut(text=text, raw=raw)


if _MULTIPART_AVAILABLE:

    @app.post("/ocr/from_upload", response_model=schemas.OCROut)
    async def ocr_from_upload(file: UploadFile = File(...)) -> schemas.OCROut:  # pragma: no cover - optional
        data = await file.read()
        if not data:
            raise HTTPException(status_code=400, detail="未收到文件数据")
        mime = file.content_type or "application/octet-stream"
        image_b64 = base64.b64encode(data).decode()
        text, raw = await _call_ocr(image_b64, mime, "只输出原文")
        return schemas.OCROut(text=text, raw=raw)


@app.post("/face/register", response_model=schemas.FaceRegisterResponse)
async def face_register(req: schemas.FaceRegisterRequest) -> schemas.FaceRegisterResponse:
    try:
        image = utils.decode_base64_image(req.image_b64)
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=400, detail=f"图像解析失败: {exc}") from exc

    embedding = FACE_ENGINE.extract(image)
    if embedding is None:
        raise HTTPException(status_code=422, detail="未检测到人脸")

    person_id, appended = _store_face(req.name, embedding)
    return schemas.FaceRegisterResponse(ok=True, id=person_id, appended=appended)


@app.post("/face/verify", response_model=schemas.FaceVerifyResponse)
async def face_verify(req: schemas.FaceVerifyRequest) -> schemas.FaceVerifyResponse:
    try:
        image = utils.decode_base64_image(req.image_b64)
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=400, detail=f"图像解析失败: {exc}") from exc

    embedding = FACE_ENGINE.extract(image)
    if embedding is None:
        raise HTTPException(status_code=422, detail="未检测到人脸")

    rows = _fetch_faces()
    candidates: List[schemas.FaceCandidate] = []
    for row in rows:
        vec = utils.decode_vector(row["vec"])
        score = utils.cosine_similarity(vec, embedding)
        candidates.append(
            schemas.FaceCandidate(person_id=row["id"], name=row["name"], score=score)
        )

    candidates.sort(key=lambda c: c.score, reverse=True)
    top_k = max(1, min(req.top_k, len(candidates))) if candidates else 0
    candidates = candidates[:top_k]

    threshold = req.threshold if req.threshold is not None else float(config.cfg("FACE_COSINE_OK", 0.6))
    best = candidates[0] if candidates and candidates[0].score >= threshold else None

    return schemas.FaceVerifyResponse(ok=True, candidates=candidates, best=best)
