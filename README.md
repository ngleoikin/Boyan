# Boyan v2 — 实施说明（交付 Codex）

> 本文档面向“代码执行型”助手（如 OpenAI Codex / GPT-Engineer / MCP 工程代理）。复制本说明后即可驱动代理完成 Boyan v2 的全部落地工作：包含目录约束、环境变量、接口契约、数据库结构、启动脚本、冒烟测试与验收标准。

## 0. 目标与范围

* **边端事件总线**（Edge Event → Rule → Action）：统一接入边缘事件（人形/人脸/OCR 等），由规则引擎生成动作（通知、问候、入库、提醒等）。
* **OCR 网关**：调用本地 `deepseek-ocr-server`（HTTP）进行图文识别；保证 **中文直出**、**Data URI** 与 **文件上传** 两种路径可用。
* **人脸能力**：OpenCV Zoo 的 **YuNet 检测** + **SFace 识别**（ONNX），提供 `/face/register` 与 `/face/verify`。
* **知识与提醒**：轻量 **SQLite** 的 **笔记** 与 **提醒**（解析“X分钟后 / 明天 HH:MM / 绝对时间”），支持导出 **ICS**。
* **运维体验**：AutoDL 云台友好，一条命令启动、`screen` 托管、健康检查、`smoke` 脚本自检。
* **可维护性**：统一 `.env` 配置、OpenAPI、最小依赖、无外网硬依赖（模型文件可“本地挂载”）。

---


## 1. 仓库结构（建议）

```
Boyan/
├─ README.md
├─ .env.example
├─ .gitignore
├─ Makefile
├─ scripts/
│  ├─ bootstrap_ubuntu.sh        # 首次环境准备（依赖、字体、venv、DB 初始化）
│  ├─ run_ocr.sh                 # 启动 deepseek-ocr-server（若已外部管理可空）
│  ├─ run_gate.sh                # 启动网关（人脸 + OCR）
│  ├─ run_brain.sh               # 启动大脑（规则、笔记、提醒）
│  ├─ smoke.sh                   # 连通性/功能冒烟测试
│  └─ systemd/（可选）           # systemd 单元模板
└─ services/
   └─ mcp/
      ├─ requirements.txt
      ├─ mcp_gateway_face.py     # 端口 :8787, 人脸 + OCR 网关
      ├─ brain_server.py         # 端口 :8768, 规则 / 笔记 / 提醒
      ├─ common/
      │  ├─ config.py
      │  ├─ db.py
      │  ├─ schemas.py
      │  └─ utils.py
      ├─ models/opencv_zoo/      # *.onnx 放这里（gitignore）
      └─ storage/                # SQLite / 缓存（gitignore）
```

> 重要：**不要把 `deepseek-ocr.rs` 仓库加入本仓**。使用 README 指导用户如何独立运行即可（或当作“外部服务”）。

---

## 2. `.env.example`

```dotenv
# ==== OCR 服务 ====
OCR_BASE=http://127.0.0.1:8000

# ==== LLM（可选，解析/补全用；若没有就留空） ====
OPENAI_BASE=https://api.deepseek.com/v1
OPENAI_API_KEY=
MODEL_ID=deepseek-v3.1

# ==== 人脸模型（本机绝对路径） ====
YUNET_MODEL=/root/Boyan/services/mcp/models/opencv_zoo/face_detection_yunet_2023mar.onnx
SFACE_MODEL=/root/Boyan/services/mcp/models/opencv_zoo/face_recognition_sface_2021dec.onnx

# ==== 数据库 ====
ASSIST_DB=/root/Boyan/services/mcp/storage/assist.db

# ==== 端口与时区 ====
GATE_PORT=8787
BRAIN_PORT=8768
LOCAL_TZ=Asia/Taipei
```

> “你的用户名”就是当前 Linux 登录用户。AutoDL 通常是 `root`，所以路径如上。若不同，按实际替换。

---

## 3. 接口契约（OpenAPI 由代码自动暴露）

### 3.1 网关（:8787）`mcp_gateway_face.py`

* `GET /health` → `{"ok": true, "ocr": <bool>, "face": <bool>}`
* `POST /ocr/from_data_uri`
  - 入参：`{ "data_uri": "<data:image/...;base64,xxx>", "prompt": "只输出原文" }`
  - 出参：`{ "text": "xxx", "raw": {...} }`
* `POST /ocr/from_upload`（可选，multipart/file）
* `POST /face/register`
  - 入参：`{ "name": "张三", "image_b64": "<base64>" }`
  - 出参：`{ "ok": true, "id": "p_xxx", "appended": true/false }`
* `POST /face/verify`
  - 入参：`{ "image_b64": "<base64>" }`
  - 出参：`{ "ok": true, "candidates": [ { "name":"张三","person_id":"p_xxx","score":0.91 } ], "best": {...} }`

> 识别阈值、Top-K 在 `.env` 或常量里给默认（如 `0.6`）。

### 3.2 大脑（:8768）`brain_server.py`

* `GET /health` → `{"ok": true}`
* `POST /edge/event`
  - 入参（统一事件模型）：

    ```json
    {
      "type": "detected",
      "device_id": "esp32s3-abc",
      "ts": "2025-10-31T10:00:00Z",
      "cam": "door-cam-01",
      "topic": "vision/person",
      "data": { "confidence": 0.92 }
    }
    ```

  - 返回：`{ "ok": true, "skipped": false, "actions": [ { "action": "notify", "title": "...", "text": "...", "device_id": "...", "ts": "..." } ] }`
  - 规则示例：
    * `topic == "vision/person"` → `notify`
    * `topic == "vision/face/verified"` → `greet`
    * `topic == "vision/ocr"` → `store_ocr`
* `POST /notes/add`
  - 入参：`{ "text": "明早联系王工要合同", "tags": ["autotest"] }`
  - 出参：`{"ok": true, "id": 1}`
* `GET /notes/list?q=关键词` → `[{id,text,tags,created_ts}]`
* `POST /reminders/parse_create`
  - 入参：`{ "utterance": "10分钟后提醒我喝水", "text": "喝水" }`
  - 出参：`{"ok": true, "id": 1, "due_ts": "..."}`
* `GET /reminders/list` → `[{id,text,due_ts,created_ts}]`
* `GET /reminders/export/ics` → `text/calendar`（多 VEVENT）

---

## 4. 数据库（SQLite）

`ASSIST_DB` 初始化于应用启动：

```sql
CREATE TABLE IF NOT EXISTS notes(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  text TEXT NOT NULL,
  tags TEXT,
  created_ts TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS reminders(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  text TEXT NOT NULL,
  due_ts TEXT NOT NULL,   -- ISO8601
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
  id TEXT PRIMARY KEY,         -- p_ 开头
  name TEXT,
  vec BLOB,                    -- 512/128-D 向量（np.float32.tobytes）
  created_ts TEXT NOT NULL
);
```

---

## 5. 关键实现点（Codex 需完成）

* **Config**：`common/config.py` 读取 `.env`，提供 `cfg("OCR_BASE")` 等。
* **DB**：`common/db.py` 暴露 `get_conn()`（WAL 模式）、初始化表；统一 `row_factory=sqlite3.Row`。
* **Schemas**：`common/schemas.py` 定义 Pydantic 模型（EdgeEvent、NoteIn、ReminderIn 等）。
* **Utils**：中文时间解析（优先 `dateparser`，无则纯规则），ICS 生成（拼接 VEVENT），Base64 工具。
* **OCR 调用**：对 `OCR_BASE/v1/responses` 发起 POST（`input_text + input_image`），并兼容提取 `output[0].content[].text / output_text / choices[0].message.content`。
* **Face Pipeline**：
  * YuNet 检测：返回最大人脸 bbox；
  * SFace 提取 embedding，归一化；
  * 与 DB 向量做余弦相似度，返回 Top-K。
* **规则引擎（最简版）**：按 `topic` 路由到动作（如上），返回动作数组。
* **错误处理**：统一 4xx/5xx；`/health` 不依赖外部组件（但可探测 OCR 可用性）。

---

## 6. 启动与冒烟

### 6.1 一键准备

```bash
make bootstrap     # 或 scripts/bootstrap_ubuntu.sh
```

应执行：

1. 安装依赖（Python3.10+、pip、venv、fonts-noto-cjk、jq、curl）。
2. `python3 -m venv services/mcp/.venv && pip install -r services/mcp/requirements.txt`。
3. 创建 `storage/assist.db`。
4. 提示下载 YuNet/SFace 到 `models/opencv_zoo/`。

### 6.2 运行

```bash
# 1) OCR 服务（你已有独立进程则跳过）
./scripts/run_ocr.sh

# 2) 网关
./scripts/run_gate.sh

# 3) 大脑
./scripts/run_brain.sh
```

或用 `screen`：

```bash
screen -S boyan_gate -dm bash -lc "./scripts/run_gate.sh"
screen -S boyan_brain -dm bash -lc "./scripts/run_brain.sh"
```

### 6.3 冒烟

```bash
make smoke         # 或 ./scripts/smoke.sh
```

包括：

* `GET :8000/v1/health`、`GET :8787/health`、`GET :8768/health`。
* 生成“自检中文图”，OCR 返回应包含 `【Boyan自检】小智AI OCR联调成功`。
* 人脸 `register/verify` 流程。
* `/edge/event` 三类主题触发不同动作。
* `notes` 增/查，`reminders` 解析 + ICS 导出。

---

## 7. 依赖与版本

`services/mcp/requirements.txt` 建议内容：

```
fastapi==0.115.0
uvicorn[standard]==0.30.6
pydantic==2.7.4
python-dotenv==1.0.1
requests==2.32.3
python-dateutil==2.9.0
dateparser==1.2.0
numpy==1.26.4
opencv-python-headless==4.10.0.84
onnxruntime==1.18.1
ics==0.7.2    # 可选，或自行拼 VEVENT
```

---

## 8. Makefile

```makefile
.PHONY: bootstrap gate brain ocr smoke stop

bootstrap:
bash scripts/bootstrap_ubuntu.sh

gate:
bash scripts/run_gate.sh

brain:
bash scripts/run_brain.sh

ocr:
bash scripts/run_ocr.sh

smoke:
bash scripts/smoke.sh

stop:
pkill -f "uvicorn .*mcp_gateway_face" || true; pkill -f "uvicorn .*brain_server" || true
```

---

## 9. `.gitignore`

```
# venv / IDE
services/mcp/.venv/
__pycache__/
*.pyc
.vscode/
.idea/

# secrets
.env

# models / storage / logs
services/mcp/models/
services/mcp/storage/
*.onnx
*.log

# OS
.DS_Store
Thumbs.db

# nested repos
deepseek-ocr.rs/
```

---

## 10. 验收标准

* **中文纯文本回显**：调用 OCR，不带图片，输出中文不被 `?` 代替。
* **Data URI OCR**：对自生成中文图，返回字符串包含“**Boyan自检**”。
* **人脸注册/识别**：注册 1 张人脸，`/face/verify` 返回 Top-1 名字与 `score > 0.6`。
* **事件路由**：`vision/person` → `notify`；`vision/face/verified` → `greet`；`vision/ocr` → `store_ocr`（落表 `ocr_logs`）。
* **笔记/提醒/ICS**：成功新增、查询、解析“10分钟后…”，导出 ICS 至 `text/calendar`。
* **健康检查**：三个服务均返回 `{"ok": true}`，无阻塞/崩溃。
* **离线可用**：在完全无外网时，只要 OCR 与模型已本地可用，功能不受影响。

---

## 11. 任务分解（给 Codex）

1. 生成仓库骨架（§1）与 `.gitignore`、`.env.example`、`Makefile`、`scripts/*`（§6、§8、§9）。
2. 实现 `common/*`（配置、DB、Schema、工具）。
3. 实现 `mcp_gateway_face.py`：OCR 两端点 + 人脸两端点 + `/health`。
4. 实现 `brain_server.py`：DB 初始化、规则引擎、notes/reminders/ICS + `/health`。
5. 编写 `smoke.sh` 覆盖 §10 验收用例。
6. 文档化 `README.md`：安装 → 配置 → 运行 → 冒烟 → 常见问题。
7. 在 AutoDL 用 `screen` 方式给出生产运行建议。

---

## 12. 常见问题（FAQ）

* **中文变问号**：确保调用 `deepseek-ocr-server` 且模型有中文 tokenizer；服务端与终端统一 UTF-8；必要时安装 `fonts-noto-cjk`。
* **`/edge/event` 返回 `skipped: true`**：规则未命中，请检查 `topic` 精确匹配和必填字段（`type/ts/cam`）。
* **`/notes` 或 `/reminders` 500**：多为 DB 文件夹不存在或 `ASSIST_DB` 路径不可写；确保启动时初始化 DB（见 §4）。

```
OPENAI_BASE=https://api.deepseek.com
OPENAI_API_KEY=sk-933514f45ed44598a2468168e991dc89
MODEL_ID=deepseek-chat
```

### DeepSeek 测试凭据（临时）

为了方便快速验证，可使用以下示例配置：

```
OPENAI_BASE=https://api.deepseek.com
OPENAI_API_KEY=
MODEL_ID=deepseek-chat
```

> 请勿将真实生产密钥提交至仓库。部署前务必替换为自己的凭据。
