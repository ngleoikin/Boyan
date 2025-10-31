#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT_DIR/services/mcp/storage"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/deepseek-ocr.log"

PORT="${OCR_PORT:-8000}"
TARGET_BASE="${OCR_BASE:-}"
if [[ -z "$TARGET_BASE" ]]; then
  TARGET_BASE="http://127.0.0.1:${PORT}"
fi
if [[ "$TARGET_BASE" != "http://127.0.0.1:${PORT}" && "$TARGET_BASE" != "http://0.0.0.0:${PORT}" ]]; then
  echo "[run_ocr] 检测到 OCR_BASE=${TARGET_BASE}，假定外部服务已运行，跳过本地启动。"
  exit 0
fi

if ! command -v deepseek-ocr-server >/dev/null 2>&1; then
  cat <<'MSG'
[run_ocr] 未检测到 deepseek-ocr-server 命令。
请参考：https://github.com/deepseek-ai/deepseek-ocr/tree/main/server
安装后将其加入 PATH，或在 .env 中设置 OCR_BASE 指向现有服务。
MSG
  exit 1
fi

MODEL_DIR="${OCR_MODEL_DIR:-$ROOT_DIR/services/mcp/models/deepseek}"

echo "[run_ocr] 在端口 ${PORT} 启动 deepseek-ocr-server (模型目录: ${MODEL_DIR})"
exec deepseek-ocr-server \
  --host 0.0.0.0 \
  --port "$PORT" \
  --model-dir "$MODEL_DIR" \
  >>"$LOG_FILE" 2>&1
