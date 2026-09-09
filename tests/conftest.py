"""全仓测试 conftest：测试配置与开发机 .env 隔离。

单 Agent 全量优化计划·测试验收基线：测试配置不得读取工作目录 `.env`。
本文件必须在任何 `app.config.settings` 导入之前执行（pytest 根 conftest
先于测试模块加载），通过 ECOM_ENV_FILE=none 让 Settings 单例跳过 .env——
CI 无 .env 时行为不变；本地开发机 .env（RAG_BACKEND=es、自定义模型名等）
不再泄漏进测试配置。需要测 .env 相关行为的用例自行显式覆盖 settings。
"""

from __future__ import annotations

import os

os.environ.setdefault("ECOM_ENV_FILE", "none")
