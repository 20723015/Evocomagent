"""fake 服务包：确定性 OpenAI 兼容 / Commerce 契约服务（kind/本地无密钥测试）。

- fake_openai：/v1/chat/completions（按输入返回预置脚本回复）+
  /v1/embeddings（确定性向量：按 token 哈希生成 1024 维）；
- fake_commerce：HTTPCommerceGateway 契约（X-Actor-Id + Bearer；
  GET /orders/{id}、POST /orders/{id}/refunds），固有 mock 数据 + 归属校验。
"""

from __future__ import annotations