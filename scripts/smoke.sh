#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="$ROOT_DIR/.env"
SERVICES_DIR="$ROOT_DIR/services/mcp"

if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "[smoke] 缺少命令: $1" >&2
    exit 1
  fi
}

require_cmd curl
require_cmd jq
require_cmd python3

OCR_BASE="${OCR_BASE:-http://127.0.0.1:8000}"
GATE_BASE="http://127.0.0.1:${GATE_PORT:-8787}"
BRAIN_BASE="http://127.0.0.1:${BRAIN_PORT:-8768}"

step() { printf '\033[1;34m[smoke]\033[0m %s\n' "$*"; }

assert_contains() {
  local needle="$1" haystack="$2"
  if [[ "$haystack" != *"$needle"* ]]; then
    echo "[smoke] 断言失败，未找到: $needle" >&2
    exit 1
  fi
}

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

step "检查 OCR 服务健康"
curl -fsS "$OCR_BASE/v1/health" | jq '.' >"$TMP_DIR/ocr_health.json"

step "检查网关健康"
curl -fsS "$GATE_BASE/health" | jq '.' >"$TMP_DIR/gate_health.json"

step "检查大脑健康"
curl -fsS "$BRAIN_BASE/health" | jq '.' >"$TMP_DIR/brain_health.json"

step "准备自检图像并执行 Data URI OCR"
python3 - <<'PY' "$TMP_DIR/ocr_input.json"
import base64, io, json, sys
from PIL import Image, ImageDraw, ImageFont
img = Image.new('RGB', (640, 200), color='#f7f7f7')
draw = ImageDraw.Draw(img)
text = '【Boyan自检】小智AI OCR联调成功'
try:
    font = ImageFont.truetype('NotoSansCJK-Regular.ttc', 32)
except Exception:
    font = ImageFont.load_default()
draw.text((40, 80), text, fill='black', font=font)
buf = io.BytesIO()
img.save(buf, format='PNG')
encoded = base64.b64encode(buf.getvalue()).decode()
data_uri = f'data:image/png;base64,{encoded}'
json.dump({'data_uri': data_uri, 'prompt': '只输出原文'}, open(sys.argv[1], 'w'))
PY

curl -fsS "$GATE_BASE/ocr/from_data_uri" \
  -H 'content-type: application/json' \
  -d @"$TMP_DIR/ocr_input.json" \
  | tee "$TMP_DIR/ocr_result.json" \
  | jq '.' >/dev/null
assert_contains "Boyan自检" "$(jq -r '.text' "$TMP_DIR/ocr_result.json")"

step "执行人脸注册"
python3 - <<'PY' "$TMP_DIR/face_register.json" "$ROOT_DIR"
import base64, json, sys
from pathlib import Path
root = Path(sys.argv[2])
img_path = root / 'test2.png'
if not img_path.exists():
    raise SystemExit('缺少 test2.png 用于自检，请放入根目录。')
data = base64.b64encode(img_path.read_bytes()).decode()
payload = {'name': 'smoke_user', 'image_b64': data}
Path(sys.argv[1]).write_text(json.dumps(payload))
PY

curl -fsS "$GATE_BASE/face/register" \
  -H 'content-type: application/json' \
  -d @"$TMP_DIR/face_register.json" \
  | tee "$TMP_DIR/face_register_result.json" \
  | jq '.' >/dev/null
FACE_ID="$(jq -r '.id' "$TMP_DIR/face_register_result.json")"

step "执行人脸核验"
python3 - <<'PY' "$TMP_DIR/face_verify.json" "$ROOT_DIR"
import base64, json, sys
from pathlib import Path
root = Path(sys.argv[2])
img_path = root / 'test2.png'
if not img_path.exists():
    raise SystemExit('缺少 test2.png 用于自检，请放入根目录。')
data = base64.b64encode(img_path.read_bytes()).decode()
Path(sys.argv[1]).write_text(json.dumps({'image_b64': data}))
PY

curl -fsS "$GATE_BASE/face/verify" \
  -H 'content-type: application/json' \
  -d @"$TMP_DIR/face_verify.json" \
  | tee "$TMP_DIR/face_verify_result.json" \
  | jq '.' >/dev/null
BEST_NAME="$(jq -r '.best.name' "$TMP_DIR/face_verify_result.json")"
BEST_SCORE="$(jq -r '.best.score' "$TMP_DIR/face_verify_result.json")"
assert_contains "smoke_user" "$BEST_NAME"
python3 - <<'PY' "$BEST_SCORE"
import sys
score = float(sys.argv[1])
if score < 0.6:
    raise SystemExit('得分低于 0.6')
PY

step "触发事件路由"
cat <<'JSON' >"$TMP_DIR/event_person.json"
{
  "type": "detected",
  "device_id": "smoke-device",
  "ts": "2025-10-31T10:00:00Z",
  "cam": "door-cam-01",
  "topic": "vision/person",
  "data": {"confidence": 0.95}
}
JSON
curl -fsS "$BRAIN_BASE/edge/event" -H 'content-type: application/json' -d @"$TMP_DIR/event_person.json" | jq '.' >"$TMP_DIR/event_person_result.json"
assert_contains "notify" "$(jq -r '.actions[0].action' "$TMP_DIR/event_person_result.json")"

cat <<'JSON' >"$TMP_DIR/event_face.json"
{
  "type": "detected",
  "device_id": "smoke-device",
  "ts": "2025-10-31T10:00:00Z",
  "cam": "door-cam-01",
  "topic": "vision/face/verified",
  "data": {"name": "smoke_user", "score": 0.92}
}
JSON
curl -fsS "$BRAIN_BASE/edge/event" -H 'content-type: application/json' -d @"$TMP_DIR/event_face.json" | jq '.' >"$TMP_DIR/event_face_result.json"
assert_contains "greet" "$(jq -r '.actions[0].action' "$TMP_DIR/event_face_result.json")"

cat <<'JSON' >"$TMP_DIR/event_ocr.json"
{
  "type": "detected",
  "device_id": "smoke-device",
  "ts": "2025-10-31T10:00:00Z",
  "cam": "door-cam-01",
  "topic": "vision/ocr",
  "data": {"text": "测试 OCR"}
}
JSON
curl -fsS "$BRAIN_BASE/edge/event" -H 'content-type: application/json' -d @"$TMP_DIR/event_ocr.json" | jq '.' >"$TMP_DIR/event_ocr_result.json"
assert_contains "store_ocr" "$(jq -r '.actions[0].action' "$TMP_DIR/event_ocr_result.json")"

step "新增笔记并查询"
cat <<'JSON' >"$TMP_DIR/note.json"
{
  "text": "明早联系王工要合同",
  "tags": ["autotest"]
}
JSON
curl -fsS "$BRAIN_BASE/notes/add" -H 'content-type: application/json' -d @"$TMP_DIR/note.json" | jq '.' >"$TMP_DIR/note_add_result.json"
NOTE_ID="$(jq -r '.id' "$TMP_DIR/note_add_result.json")"

curl -fsS "$BRAIN_BASE/notes/list?q=合同" | jq '.' >"$TMP_DIR/note_list.json"
assert_contains "$NOTE_ID" "$(jq -r '.[]? // empty' "$TMP_DIR/note_list.json")"

step "创建提醒并导出 ICS"
cat <<'JSON' >"$TMP_DIR/reminder.json"
{
  "utterance": "10分钟后提醒我喝水",
  "text": "喝水"
}
JSON
curl -fsS "$BRAIN_BASE/reminders/parse_create" -H 'content-type: application/json' -d @"$TMP_DIR/reminder.json" | jq '.' >"$TMP_DIR/reminder_add.json"
REMINDER_ID="$(jq -r '.id' "$TMP_DIR/reminder_add.json")"

curl -fsS "$BRAIN_BASE/reminders/list" | jq '.' >"$TMP_DIR/reminder_list.json"
assert_contains "$REMINDER_ID" "$(cat "$TMP_DIR/reminder_list.json")"

curl -fsS "$BRAIN_BASE/reminders/export/ics" >"$TMP_DIR/reminders.ics"
assert_contains "BEGIN:VCALENDAR" "$(cat "$TMP_DIR/reminders.ics")"

step "冒烟测试完成"
