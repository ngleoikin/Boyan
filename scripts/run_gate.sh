#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="$ROOT_DIR/.env"
SERVICES_DIR="$ROOT_DIR/services/mcp"
VENV_DIR="$SERVICES_DIR/.venv"
UVICORN_BIN="${UVICORN_BIN:-$VENV_DIR/bin/uvicorn}"
APP_PATH="mcp_gateway_face:app"
LOG_DIR="$ROOT_DIR/services/mcp/storage"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/gate.log"

if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

PORT="${GATE_PORT:-${PORT:-8787}}"
HOST="${GATE_HOST:-0.0.0.0}"

if [[ ! -x "$UVICORN_BIN" ]]; then
  if command -v uvicorn >/dev/null 2>&1; then
    UVICORN_BIN="$(command -v uvicorn)"
  else
    echo "[run_gate] 未找到 uvicorn。请先执行 scripts/bootstrap_ubuntu.sh" >&2
    exit 1
  fi
fi

echo "[run_gate] 启动 ${APP_PATH} @ ${HOST}:${PORT}"
exec "$UVICORN_BIN" "$APP_PATH" \
  --host "$HOST" \
  --port "$PORT" \
  --app-dir "$SERVICES_DIR" \
  >>"$LOG_FILE" 2>&1
