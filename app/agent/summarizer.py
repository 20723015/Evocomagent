import json
from typing import Optional

from openai import OpenAI

from app.agent.tools.digest import (
    render_tool_result_line,
    tool_call_name_map,
    visible_assistant_text,
)
from app.config.settings import settings
from app.prompts.summarizer import SUMMARY_PROMPT

# tool_calls.arguments 转录截断：结果侧有 digest 预算（tool_digest_budget_chars），
# 参数侧此前无上限——大段自由文本参数（如退款理由）会撑大摘要 LLM 输入而不增信息量
_ARGS_CUT = 120


def summarize(
    client: OpenAI,
    model: str,
    old_messages: list[dict],
    prev_summary: Optional[str],
) -> str:
    """把老对话（可选地带上上一次 summary）压缩成新的 summary 文本。

    支持 user / assistant / tool 以及含 tool_calls 的 assistant 消息；
    assistant 的块列表 content（推理画像回传形态）只取 text 块（见
    digest.visible_assistant_text）。
    """
    parts: list[str] = []
    if prev_summary:
        parts.append(f"【此前摘要】\n{prev_summary}")

    transcript_lines = []
    call_map = tool_call_name_map(old_messages)
    for msg in old_messages:
        role = msg.get("role")
        content = msg.get("content") or ""

        if role == "user":
            transcript_lines.append(f"用户：{content}")
        elif role == "assistant":
            tool_calls = msg.get("tool_calls")
            if tool_calls:
                for tc in tool_calls:
                    func = tc.get("function", {})
                    name = func.get("name", "?")
                    args = _format_tool_args(func.get("arguments", "{}"))
                    transcript_lines.append(f"客服：[调用工具 {name}({args})]")
            text = visible_assistant_text(msg.get("content"))
            if text:
                transcript_lines.append(f"客服：{text}")
        elif role == "tool":
            name = call_map.get(msg.get("tool_call_id"), "?")
            transcript_lines.append(render_tool_result_line(name, content))

    parts.append("【待压缩对话】\n" + "\n".join(transcript_lines))

    user_content = "\n\n".join(parts)

    summary_max_chars = max(int(settings.summary_max_chars), 100)
    # 输出上限按字数预算推导（CJK≈1 token/字，×2 留英文与句边界余量），不再
    # 沿用 llm_max_tokens——500 字摘要配 8k 输出上限纯属浪费；推理画像的
    # min_max_tokens 会在参数改写层抬回下限，思考预算语义不受影响。
    summary_tokens = min(int(settings.llm_max_tokens), summary_max_chars * 2 + 256)
    response = client.chat.completions.create(
        model=model,
        temperature=0.3,
        messages=[
            {"role": "system", "content": SUMMARY_PROMPT.format(
                summary_max_chars=summary_max_chars,
            )},
            {"role": "user", "content": user_content},
        ],
        max_tokens=summary_tokens,
    )
    summary = response.choices[0].message.content.strip()
    return _truncate_summary(summary, summary_max_chars)


def _format_tool_args(args) -> str:
    """tool_calls.arguments 转录：str 原样、其他类型序列化；超限截断带省略号。"""
    text = args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)
    if len(text) <= _ARGS_CUT:
        return text
    return text[: _ARGS_CUT - 1] + "…"


_TRUNCATION_MARKER = "（摘要已截断）"
_SENTENCE_BOUNDARIES = "。！？；\n"


def _truncate_summary(summary: str, max_chars: int) -> str:
    """批次5（Review #5）：硬截断先回退到最后一个句边界。

    摘要预算必须代码保证（prompt 纪律不可靠），但直接 ``summary[:max]``
    会把订单号/承诺句切成半截再注入 prompt。策略：
    - 超限时回退到预算内最后一个句边界（。！？；换行）；
    - 仅当首句即超限（预算内无任何句边界）才硬切兜底；
    - 截断后追加「（摘要已截断）」标记，且最终长度仍 ≤ max_chars。
    """
    if len(summary) <= max_chars:
        return summary
    budget = max_chars - len(_TRUNCATION_MARKER)
    if budget <= 0:
        # 极小 max_chars：标记放不下，退化为纯硬切
        return summary[:max_chars]
    window = summary[:budget]
    cut = max(window.rfind(ch) for ch in _SENTENCE_BOUNDARIES)
    if cut >= 0:
        return window[: cut + 1].rstrip() + _TRUNCATION_MARKER
    return window.rstrip() + _TRUNCATION_MARKER
