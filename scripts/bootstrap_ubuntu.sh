#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVICES_DIR="$ROOT_DIR/services/mcp"
VENV_DIR="$SERVICES_DIR/.venv"
REQUIREMENTS_FILE="$SERVICES_DIR/requirements.txt"
DB_PATH="${ASSIST_DB:-$SERVICES_DIR/storage/assist.db}"

info() { printf '\033[1;34m[bootstrap]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[bootstrap]\033[0m %s\n' "$*"; }

SUDO=""
if command -v sudo >/dev/null 2>&1 && [[ ${EUID:-$(id -u)} -ne 0 ]]; then
  SUDO="sudo "
fi

if [[ ${EUID:-$(id -u)} -ne 0 ]]; then
  warn "建议使用 sudo 运行以便自动安装系统依赖（fonts-noto-cjk、jq、curl 等）。"
fi

if command -v apt-get >/dev/null 2>&1; then
  info "安装系统依赖 (python3, python3-venv, fonts-noto-cjk, jq, curl)"
  ${SUDO}apt-get update
  ${SUDO}apt-get install -y python3 python3-venv python3-pip fonts-noto-cjk jq curl
else
  warn "未检测到 apt-get，跳过系统依赖安装。请手动确保 Python3.10+、jq、curl 可用。"
fi

info "准备 Python 虚拟环境：$VENV_DIR"
python3 -m venv "$VENV_DIR"
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip
if [[ -f "$REQUIREMENTS_FILE" ]]; then
  python -m pip install -r "$REQUIREMENTS_FILE"
else
  warn "缺少 $REQUIREMENTS_FILE，请参考 README 填充依赖列表。"
fi

deactivate

mkdir -p "$SERVICES_DIR/models/opencv_zoo"
mkdir -p "$(dirname "$DB_PATH")"

if [[ ! -f "$DB_PATH" ]]; then
  info "初始化 SQLite 数据库：$DB_PATH"
  DB_PATH="$DB_PATH" python - <<'PY'
import os
import sqlite3
path = os.environ["DB_PATH"]
os.makedirs(os.path.dirname(path), exist_ok=True)
conn = sqlite3.connect(path)
conn.executescript("""
CREATE TABLE IF NOT EXISTS notes(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  text TEXT NOT NULL,
  tags TEXT,
  created_ts TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS reminders(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  text TEXT NOT NULL,
  due_ts TEXT NOT NULL,
  created_ts TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS ocr_logs(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  device_id TEXT,
  cam TEXT,
  text TEXT,
  lang TEXT,
  ts TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS faces(
  id TEXT PRIMARY KEY,
  name TEXT,
  vec BLOB,
  created_ts TEXT NOT NULL
);
""")
conn.commit()
conn.close()
PY
else
  info "已检测到现有数据库：$DB_PATH"
fi

cat <<'MSG'
============================================
请将 YuNet / SFace ONNX 模型下载到：
  services/mcp/models/opencv_zoo/
- face_detection_yunet_2023mar.onnx
- face_recognition_sface_2021dec.onnx
============================================
MSG

info "Bootstrap 完成"
