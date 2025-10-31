# 小智 AI 主机端服务使用手册

本手册介绍如何在本地主机上部署并使用“小智 AI”项目的主机端服务，包括网关 (`mcp_gateway_face.py`) 与编排器 (`brain_server.py`)。所有示例均在 Windows/WSL/Linux 的 Python 3.10+ 环境下验证。

## 1. 目录结构

```
Boyan/
├── README.md                # 本手册
├── services/
│   └── mcp/
│       ├── brain_server.py  # 事件编排器（端口 8768）
│       ├── mcp_gateway_face.py # 本地能力网关（端口 8787）
│       ├── requirements.txt # 依赖清单
│       ├── models/opencv_zoo/ # 放置 YuNet/SFace ONNX 模型
│       ├── storage/          # 记事、提醒、人脸向量等持久化文件
│       └── tmp/              # 临时语音缓存等
└── ...
```

## 2. 前置条件

1. **操作系统**：Windows 10/11、WSL2 或主流 Linux 发行版。
2. **Python**：建议 3.10 或 3.11，需具备 `pip`。
3. **依赖服务**：
   - DeepSeek OCR HTTP 服务（默认端口 8000）。
   - DeepSeek/OpenAI 兼容的大模型接口（用于工具调用）。
4. **模型文件**：在 `services/mcp/models/opencv_zoo/` 目录放置：
   - `face_detection_yunet_2023mar.onnx`
   - `face_recognition_sface_2021dec.onnx`
5. **可选工具**：
   - FFmpeg（音频解码/录制）。
   - 麦克风与摄像头（用于唤醒词、视觉异常等功能）。

## 3. 环境准备

```powershell
# 进入项目目录
cd Boyan\services\mcp

# （可选）创建虚拟环境
python -m venv .venv
.\.venv\Scripts\Activate.ps1  # Windows PowerShell
# source .venv/bin/activate     # Linux/WSL

# 安装依赖
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

> 若需语音录制，额外安装：`pip install sounddevice numpy soundfile`。默认已在 `requirements.txt` 中声明 `faster-whisper`、`apscheduler`、`plyer` 等依赖。

## 4. 关键环境变量

在启动前设置以下变量（示例 PowerShell）：

```powershell
$env:OCR_BASE = "http://127.0.0.1:8000"            # DeepSeek OCR 服务
$env:OPENAI_BASE = "https://api.deepseek.com/v1"    # LLM 接口基础地址
$env:OPENAI_API_KEY = "sk-..."                     # LLM API Key
$env:MODEL_ID = "deepseek-v3.1"                    # 默认模型
$env:YUNET_MODEL = "$PWD\models\opencv_zoo\face_detection_yunet_2023mar.onnx"
$env:SFACE_MODEL = "$PWD\models\opencv_zoo\face_recognition_sface_2021dec.onnx"
$env:ASSIST_DB = "$PWD\storage\assist.db"         # 记事/提醒数据库位置
$env:LOCAL_TZ = "Asia/Taipei"                      # 本地时区
```

其他常用可选变量：

| 变量 | 说明 |
| ---- | ---- |
| `GATE_PORT` | MCP 网关端口（默认 8787）。 |
| `BRAIN_PORT` | 编排器端口（默认 8768）。 |
| `ASR_MODEL_SIZE` / `ASR_COMPUTE` | 语音识别模型大小与精度。 |
| `WAKE_KEYWORDS` | 唤醒词列表，例如 `"小智,小智同学"`。 |
| `COOLDOWN_S`、`CD_GREET_S` 等 | 各类冷却时间配置。 |

## 5. 启动服务

### 5.1 深度学习能力

确保 OCR、大模型接口可用。必要时在单独终端启动 DeepSeek OCR 服务。

### 5.2 MCP 网关（端口 8787）

```powershell
cd Boyan\services\mcp
uvicorn mcp_gateway_face:app --host 0.0.0.0 --port 8787
```

常用健康检查：`Invoke-RestMethod http://127.0.0.1:8787/health`。

启动后可通过如下 REST 接口获得能力：

- `/tools/ocr/read`：OCR 识别
- `/tts/speak`：Windows SAPI TTS
- `/face/*`：人脸检测、注册、检索
- `/mem/*` / `/reminder/*`：记事与提醒接口
- `/notes/ingest`、`/nlp/parse_time`：文本批量入库与自然语言时间解析
- `/asr/transcribe`、`/asr/transcribe_and_ingest`：离线语音转写及自动入库
- `/wake/start`、`/audio_anomaly/start`：唤醒词与音频异常监听
- `/export/*`、`/import/all`：数据导出与恢复

### 5.3 编排器（端口 8768）

```powershell
uvicorn brain_server:app --host 0.0.0.0 --port 8768
```

编排器职责：

- 接收 ESP32 设备 `/edge/event` 上报的人形事件
- 调用网关进行人脸、OCR、ASR、记事、提醒等工具
- 管理注意/唤醒状态机、异常检测、分级关怀
- 提供笔记/提醒 REST 接口、ICS 导出、会话回溯

健康检查：`Invoke-RestMethod http://127.0.0.1:8768/health`。

## 6. 可选守护进程

### 6.1 唤醒词与音频异常

在网关启动后调用：

```powershell
Invoke-RestMethod http://127.0.0.1:8787/wake/start -Method POST
Invoke-RestMethod http://127.0.0.1:8787/audio_anomaly/start -Method POST
```

### 6.2 视觉异常检测

若需本地摄像头监控，在新终端运行：

```powershell
python vision_anomaly_worker.py
```

事件将通过 `/events/publish` 推送给编排器。

## 7. 常用操作示例

### 7.1 OCR 与语音播报

```powershell
# OCR
$b64 = [Convert]::ToBase64String([IO.File]::ReadAllBytes("C:\\tmp\\note.png"))
Invoke-RestMethod "http://127.0.0.1:8787/tools/ocr/read" -Method POST -Body (@{image_b64=$b64} | ConvertTo-Json) -ContentType "application/json"

# TTS
Invoke-RestMethod "http://127.0.0.1:8787/tts/speak" -Method POST -Body (@{text="小智已就绪"} | ConvertTo-Json) -ContentType "application/json"
```

### 7.2 人脸注册与识别

```powershell
$b64 = [Convert]::ToBase64String([IO.File]::ReadAllBytes("C:\\tmp\\me.jpg"))
Invoke-RestMethod "http://127.0.0.1:8787/face/register" -Method POST -Body (@{name="甲"; image_b64=$b64} | ConvertTo-Json) -ContentType "application/json"
Invoke-RestMethod "http://127.0.0.1:8787/face/search" -Method POST -Body (@{image_b64=$b64} | ConvertTo-Json) -ContentType "application/json"
```

### 7.3 记事与提醒

```powershell
# 记一条笔记
Invoke-RestMethod http://127.0.0.1:8768/notes/add -Method POST -Body (@{text="联系王工要合同"} | ConvertTo-Json) -ContentType "application/json"

# 自然语言创建提醒
Invoke-RestMethod http://127.0.0.1:8768/reminders/parse_create -Method POST -Body (@{utterance="今天下午3点提醒我打电话"} | ConvertTo-Json) -ContentType "application/json"

# 查看未触发提醒
Invoke-RestMethod "http://127.0.0.1:8768/reminders/list?status=scheduled"

# 取消提醒
Invoke-RestMethod http://127.0.0.1:8768/reminders/cancel -Method POST -Body (@{id=1} | ConvertTo-Json) -ContentType "application/json"
```

### 7.4 语音一键记事

```powershell
# 录音 6 秒（FFmpeg）
$mic = '麦克风 (Realtek(R) Audio)'
ffmpeg -y -f dshow -i audio="$mic" -t 6 -ac 1 -ar 16000 C:\tmp\say6.wav

# 上传转写并自动入库
$b64 = [Convert]::ToBase64String([IO.File]::ReadAllBytes("C:\\tmp\\say6.wav"))
Invoke-RestMethod http://127.0.0.1:8787/asr/transcribe_and_ingest -Method POST -Body (@{audio_b64=$b64; default_due_minutes=120} | ConvertTo-Json) -ContentType "application/json"
```

### 7.5 事件融合

手动触发事件以调试：

```powershell
Invoke-RestMethod http://127.0.0.1:8768/events/publish -Method POST -Body (@{topic="vision/person"; data=@{}} | ConvertTo-Json) -ContentType "application/json"
Invoke-RestMethod http://127.0.0.1:8768/events/publish -Method POST -Body (@{topic="wake/keyword"; data=@{text="小智"}} | ConvertTo-Json) -ContentType "application/json"
```

## 8. 数据与备份

- `storage/notes.json`、`storage/reminders.json`、`storage/faces.json`（由网关维护）与 `assist.db`（由编排器维护）存储核心数据。
- 使用 `/export/all` 导出全部数据或 `/reminders/export/ics` 导出日历文件。
- 建议定期备份 `storage/` 与 `assist.db`。

## 9. 常见问题

| 问题 | 处理建议 |
| ---- | ---- |
| 中文识别出现问号 | 刷新 DeepSeek OCR tokenizer 或在 WSL 下运行 OCR。 |
| 人脸识别误报或漏报 | 调整 `FACE_COSINE_OK` 阈值，或为同一人登记多张照片。 |
| ASR 速度慢 | 将 `ASR_MODEL_SIZE` 调为 `small`，`ASR_COMPUTE` 设置 `int8`。 |
| 唤醒词误触发 | 增大 `WAKE_MIN_INTERVAL_SEC`，缩小关键词列表。 |
| 提醒未触发 | 检查 `assist.db` 是否创建成功，并确认 APScheduler 正常运行。 |

## 10. 下一步

- 若需与 LLM 对接，将上述 REST 封装为 MCP 工具 (`note.create`, `reminder.create` 等)，并在提示词中声明使用策略。
- 如需远程报警，可订阅 `/reminder/fired`、`/guard` 相关事件并转发至企业 IM/短信。

---

如在部署或使用过程中遇到问题，可通过调试日志与 REST 接口排查；建议在生产环境中启用守护进程/服务管理（如 PowerShell 脚本或 systemd）确保 OCR、网关与编排器自动拉起。
