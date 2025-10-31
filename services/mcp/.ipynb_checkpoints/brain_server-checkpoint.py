from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional

app = FastAPI()

class EdgeEvent(BaseModel):
    type: str = Field(..., description="事件类型字符串，如 detected/alert 等")
    device_id: str
    ts: str             # 要求字符串（ISO8601 或你的自定义字符串时间）
    cam: str
    topic: str          # 如 vision/person、vision/face/verified、vision/ocr
    data: Dict[str, Any] = Field(default_factory=dict)

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/edge/event")
def edge_event(evt: EdgeEvent):
    """
    最小演示策略：
    - vision/person：confidence >= 0.8 触发 notify
    - vision/face/verified：无条件触发 greet（用于你的人脸联动）
    - vision/ocr：如果 text 非空，则存档/通知
    其他一律 skipped=True
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

    return {
        "ok": True,
        "skipped": skipped,
        "actions": actions
    }
