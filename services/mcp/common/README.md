# common 模块约定

此目录存放共享组件，供 `mcp_gateway_face.py` 与 `brain_server.py` 引用。

* `config.py`：包装 `.env` 配置访问。
* `db.py`：SQLite 连接池与初始化逻辑。
* `schemas.py`：Pydantic 数据模型定义。
* `utils.py`：Base64、时间解析、ICS 生成等工具函数。

> 实现时请确保导出的接口同时满足 FastAPI 与冒烟脚本在 README 中列举的契约。
