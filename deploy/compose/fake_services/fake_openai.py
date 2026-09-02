"""fake_openai.py：确定性 OpenAI 兼容服务（chat + embeddings）。

用途：kind/本地集成环境无需真实密钥即可跑通 Agent 全链路与评测脚本；
回复由「脚本表 + 关键词规则」生成，temperature 无关（确定性）。

- POST /v1/chat/completions：按最后一条 user 消息命中规则表返回固定回复；
  含 tools 时返回一次工具调用（query_order 等），供 ReAct 循环冒烟；
- POST /v1/embeddings：按文本 token 的 sha256 取模生成 1024 维确定性向量
  （同文本同向量，不同文本大概率不同——混合检索冒烟够用）。

运行：python -m fake_openai（默认 0.0.0.0:8000）
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()

EMBED_DIM = 1024
MODEL = "bge-m3"
CHAT_MODEL = "deepseek-v4-flash-fake"

# 关键词 → 固定回复（冒烟脚本；真实评测走外部模型，不经过本服务）
_RULES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"七天无理由|退货|退换"), "支持七天无理由退货，运费规则见退换货政策。"),
    (re.compile(r"物流|快递|到哪"), "您的订单正在配送中，物流信息请以查询结果为准。"),
    (re.compile(r"退款"), "退款申请已提交，预计 1-3 个工作日审核。"),
    (re.compile(r"你好|您好|在吗"), "您好，我是小夕，很高兴为您服务。"),
]

# 无关键词命中时的 ReAct 工具调用（让冒烟走真实工具链路）
_DEFAULT_TOOL_CALL = {
    "id": "call_fake_1",
    "type": "function",
    "function": {"name": "search_knowledge",
                 "arguments": json.dumps({"query": "帮助", "top_k": 3},
                                         ensure_ascii=False)},
}


def _embedding(text: str) -> list[float]:
    h = hashlib.sha256(text.encode("utf-8")).digest()
    return [((h[i % 32] + i) % 255) / 255.0 for i in range(EMBED_DIM)]


def _chat_reply(messages: list[dict]) -> dict:
    last_user = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            last_user = str(m.get("content", ""))
            break
    system = "".join(str(m.get("content", "")) for m in messages
                     if m.get("role") == "system")
    # 结构化提取调用：返回 CustomerServiceResponse 形状的 JSON
    if "提取结构化信息" in system or "extract" in system.lower():
        reply_text = last_user.strip('"') if last_user.strip('"') else "好的"
        payload = {
            "intent": "greeting",
            "confidence": 0.99,
            "reply": reply_text,
            "requires_human": False,
            "follow_up_question": None,
        }
        return {"content": json.dumps(payload, ensure_ascii=False),
                "tool_calls": None}
    for pattern, reply in _RULES:
        if pattern.search(last_user):
            return {"content": reply, "tool_calls": None}
    # 无关键词 → 走 ReAct 工具调用（search_knowledge）
    return {"content": "我先查一下知识库。", "tool_calls": [_DEFAULT_TOOL_CALL]}


@app.get("/health")
def health():
    return {"status": "ok"}


import asyncio

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    # 固定小延迟：制造锁竞争窗口（同 session 并发 409 验收用）
    await asyncio.sleep(0.8)
    messages = body.get("messages", [])
    reply = _chat_reply(messages)
    content = reply["content"]
    tool_calls = reply["tool_calls"]
    message = {"role": "assistant"}
    if content:
        message["content"] = content
    if tool_calls:
        message["tool_calls"] = tool_calls
    return JSONResponse({
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "model": body.get("model", CHAT_MODEL),
        "choices": [{"index": 0, "message": message,
                     "finish_reason": "tool_calls" if tool_calls else "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5,
                  "total_tokens": 15},
    })


@app.post("/v1/embeddings")
async def embeddings(request: Request):
    body = await request.json()
    inputs = body.get("input", [])
    if isinstance(inputs, str):
        inputs = [inputs]
    return JSONResponse({
        "object": "list",
        "model": body.get("model", MODEL),
        "data": [
            {"object": "embedding", "index": i,
             "embedding": _embedding(str(text))}
            for i, text in enumerate(inputs)
        ],
        "usage": {"prompt_tokens": 0, "total_tokens": 0},
    })


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)