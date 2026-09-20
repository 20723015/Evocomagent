"""目标模型能力探针（推理模型全量适配计划·T0，门禁任务）。

**为什么先探针**：DeepSeek-reasoner / OpenAI o 系列 / Claude extended thinking 在
temperature 接受度、max_tokens 参数名、reasoning 字段形状与「能否/必须回传」上
两两矛盾（厂商差异速查表见 `docs/推理模型适配-实施记录.md`）。那张表是文档知识，
实现常有出入；
探针成本几毛钱，换回来的画像直接驱动 T1 的参数改写与 T3 的双通道策略。

用法：
  # 1) 探针（需要真实凭证；输出画像 JSON）
  python -m app.scripts.probe_model_capability --model deepseek-reasoner \
      --out app/sessions/model_profile.json

  # 2) 把画像灌进部署配置（免发版）
  python -m app.scripts.probe_model_capability --model deepseek-reasoner \
      --emit-overrides

  # 3) 只验证脚本接线（无网络、无凭证；用脚本化假客户端）
  python -m app.scripts.probe_model_capability --dry-run

**门禁语义**：第 1 项（挂 tools 能否正常返回 `tool_calls`）不通过 → 退出码 2，
整个适配计划终止（文本协议回退不在计划范围内，是另一个量级的工程）。
其余各项只决定画像取值，不阻塞。

**产物带用途标注**：`latency`（P50/P95 → T8 的 timeout/轮次预算）与
`reasoning_tokens`（典型消耗 → T7 的预算校准）都会被后续任务消费，
因此脚本同时打印这两项的结论。
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from app.config.settings import settings  # noqa: E402
from app.observability.logging import get_logger  # noqa: E402

log = get_logger("app.scripts.probe_model_capability")

PROBE_VERSION = "1"
GATE_FAILED_EXIT = 2

BUSINESS_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "get_order_status",
        "description": "查询订单状态（只读）",
        "parameters": {
            "type": "object",
            "properties": {"order_id": {"type": "string", "description": "订单号"}},
            "required": ["order_id"],
        },
    },
}
FINAL_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "final_response",
        "description": "提交最终答复并结束本轮（终止调用）",
        "parameters": {
            "type": "object",
            "properties": {
                "intent": {"type": "string"},
                "reply": {"type": "string"},
                "requires_human": {"type": "boolean"},
                "follow_up_question": {"type": ["string", "null"]},
            },
            "required": ["intent", "reply", "requires_human"],
        },
    },
}
_KNOWN_REASONING_ATTRS = (
    "reasoning_content", "reasoning", "thinking_blocks", "thinking",
)
_THINKING_BLOCK_TYPES = ("thinking", "redacted_thinking", "reasoning")


# ============================================================
# 结果容器
# ============================================================
class ProbeLog:
    """探针过程记录：每次调用一行（延迟、是否 400、usage、reasoning 形状）。"""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def add(self, name: str, *, ok: bool, latency_ms: float = 0.0,
            error: str = "", **extra: Any) -> None:
        entry = {
            "probe": name, "ok": ok, "latency_ms": round(latency_ms, 1),
        }
        if error:
            entry["error"] = error
        entry.update({k: v for k, v in extra.items() if v is not None})
        self.calls.append(entry)
        log.info("probe=%s ok=%s latency=%.0fms %s", name, ok, latency_ms,
                 entry.get("error", "") or extra.get("note", ""))

    def latencies(self) -> list[float]:
        # 注意用键存在判定而非真值：dry-run 的假客户端延迟为 0.0ms（0 是假值）
        return [c["latency_ms"] for c in self.calls if "latency_ms" in c]

    def reasoning_tokens(self) -> list[int]:
        return [c["reasoning_tokens"] for c in self.calls
                if c.get("reasoning_tokens")]


class _BadRequest(Exception):
    """SDK BadRequestError 的替身：探针只关心「是不是参数不合法」。"""


def _is_bad_request(exc: Exception) -> bool:
    name = type(exc).__name__
    return name in ("BadRequestError", "UnprocessableEntityError") or isinstance(
        exc, _BadRequest,
    )


def _call(client, log_: ProbeLog, name: str, **kwargs):
    """执行一次探针调用：统一计时/异常归类；返回 (response|None, error|None)。"""
    start = time.time()
    try:
        response = client.chat.completions.create(**kwargs)
    except Exception as e:  # noqa: BLE001 —— 探针要把任何异常都变成证据
        latency = (time.time() - start) * 1000
        kind = "BadRequest" if _is_bad_request(e) else type(e).__name__
        log_.add(name, ok=False, latency_ms=latency, error=f"{kind}: {e}")
        return None, e
    latency = (time.time() - start) * 1000
    usage = _usage_dict(response)
    log_.add(name, ok=True, latency_ms=latency, **usage)
    return response, None


def _message(response) -> Any:
    try:
        return response.choices[0].message
    except (AttributeError, IndexError, TypeError):
        return None


def _tool_calls(response) -> list:
    message = _message(response)
    return list(getattr(message, "tool_calls", None) or [])


def _usage_dict(response) -> dict:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    details = getattr(usage, "completion_tokens_details", None)
    reasoning = getattr(details, "reasoning_tokens", 0) if details is not None else 0
    if not reasoning:
        reasoning = getattr(usage, "reasoning_tokens", 0)
    finish = ""
    try:
        finish = str(response.choices[0].finish_reason or "")
    except (AttributeError, IndexError, TypeError):
        finish = ""
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
        "reasoning_tokens": int(reasoning or 0),
        "finish_reason": finish or None,
    }


def _thinking_blocks(message) -> list[dict]:
    content = getattr(message, "content", None)
    if not isinstance(content, list):
        return []
    return [
        dict(item) for item in content
        if isinstance(item, dict) and str(item.get("type") or "") in _THINKING_BLOCK_TYPES
    ]


def _reasoning_shape(message) -> tuple[str, Any, bool]:
    """返回 (field, payload, signed)。

    field: none | reasoning_content | thinking_blocks
    """
    blocks = _thinking_blocks(message)
    if blocks:
        signed = any(b.get("signature") for b in blocks)
        return "thinking_blocks", blocks, signed
    for attr in _KNOWN_REASONING_ATTRS:
        value = getattr(message, attr, None)
        if isinstance(value, str) and value.strip():
            return "reasoning_content", value, False
        if isinstance(value, list) and value:
            return "thinking_blocks", value, any(
                isinstance(b, dict) and b.get("signature") for b in value
            )
    return "none", None, False


# ============================================================
# 六项探针
# ============================================================
def probe_tools(client, model: str, log_: ProbeLog) -> dict:
    """1) 挂 tools 是否正常返回 tool_calls（业务工具 + 终止工具各一次）。"""
    business, err = _call(
        client, log_, "tools_business",
        model=model,
        messages=[{"role": "user", "content": "帮我查一下订单 A12345 的状态"}],
        tools=[BUSINESS_TOOL],
    )
    terminal, err2 = _call(
        client, log_, "tools_terminal",
        model=model,
        messages=[{"role": "user", "content": "订单 A12345 已发货，请给出最终答复"}],
        tools=[BUSINESS_TOOL, FINAL_TOOL],
    )
    business_calls = _tool_calls(business) if business is not None else []
    terminal_calls = _tool_calls(terminal) if terminal is not None else []
    return {
        "supports_tools": bool(business_calls),
        "business_tool_calls": len(business_calls),
        "terminal_tool_calls": len(terminal_calls),
        "business_error": str(err) if err else "",
        "terminal_error": str(err2) if err2 else "",
    }


def probe_forced_tool_choice(client, model: str, log_: ProbeLog) -> dict:
    """2) forced tool_choice 是否接受（不接受 → 400）。"""
    response, err = _call(
        client, log_, "forced_tool_choice",
        model=model,
        messages=[{"role": "user", "content": "请提交最终答复：订单已发货。"}],
        tools=[FINAL_TOOL],
        tool_choice={"type": "function", "function": {"name": "final_response"}},
    )
    if response is None:
        return {
            "accepted": False,
            "honored": False,
            "error": str(err) if err else "",
        }
    calls = _tool_calls(response)
    honored = bool(calls) and getattr(calls[0].function, "name", "") == "final_response"
    return {"accepted": True, "honored": honored, "error": ""}


def probe_temperature(client, model: str, log_: ProbeLog) -> dict:
    """3) temperature=0.7 / 0.0 / 不传：400 / 忽略 / 生效。

    「生效」只能启发式判定：同一 prompt 在 0.0 与 0.7 下各跑两次，四次输出完全
    相同 → 大概率被忽略（确定性）；出现分歧 → free（尊重该值）。400 是确定证据。
    """
    prompt = "用一句话说明客服处理退款申请时最需要注意什么。"
    results: dict[str, Any] = {"accepted": {}, "samples": {}}
    for label, value in (("0.7", 0.7), ("0.0", 0.0), ("absent", None)):
        texts: list[str] = []
        for _ in range(2):
            kwargs: dict[str, Any] = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
            }
            if value is not None:
                kwargs["temperature"] = value
            response, err = _call(client, log_, f"temperature_{label}", **kwargs)
            if response is None:
                results["accepted"][label] = (
                    "rejected" if err is None or _is_bad_request(err)
                    else f"error:{type(err).__name__}"
                )
                break
            results["accepted"][label] = "accepted"
            texts.append(_text_of(response))
        results["samples"][label] = texts

    free = results["accepted"].get("0.7") == "accepted"
    zero = results["accepted"].get("0.0") == "accepted"
    hot = results["samples"].get("0.7") or []
    cold = results["samples"].get("0.0") or []
    if not free and zero:
        mode = "fixed_1"   # 只接受某个固定值（Claude thinking 语义）
    elif not free and not zero:
        mode = "forbidden"  # 传值即 400
    elif hot and cold and len(set(hot + cold)) == 1:
        mode = "ignored"    # 接受但 0.0/0.7 输出一致 → 大概率忽略
    else:
        mode = "free"
    results["temperature_mode"] = mode
    return results


def probe_max_tokens(client, model: str, log_: ProbeLog, param: str) -> dict:
    """4) max_tokens=2048 出多步推理题：截断行为 + usage 构成。"""
    question = (
        "一家电商的退款规则：商品 7 天内可无理由退，15 天内质量问题可退，"
        "超过 15 天但未拆封可申请特批。用户买了 20 天的商品且未拆封，"
        "请分步推理：他是否符合退款条件？需要走哪条流程？请把推理过程完整写出。"
    )
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": question}],
        param: 2048,
    }
    response, err = _call(client, log_, "max_tokens_2048", **kwargs)
    if response is None:
        return {"accepted": False, "error": str(err) if err else ""}
    usage = _usage_dict(response)
    truncated = usage.get("finish_reason") == "length"
    return {
        "accepted": True,
        "error": "",
        "truncated": truncated,
        "finish_reason": usage.get("finish_reason"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "reasoning_tokens": usage.get("reasoning_tokens"),
        "text_chars": len(_text_of(response)),
    }


def probe_reasoning_shape(client, model: str, log_: ProbeLog, param: str) -> dict:
    """5) reasoning 字段的确切形状。"""
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "user", "content": "一个订单 20 天未拆封，能退款吗？请先推理再回答。"},
        ],
        param: 2048,
    }
    response, err = _call(client, log_, "reasoning_shape", **kwargs)
    if response is None:
        return {"reasoning_field": "none", "error": str(err) if err else ""}
    message = _message(response)
    field, payload, signed = _reasoning_shape(message)
    attrs = sorted(
        a for a in dir(message)
        if not a.startswith("_") and a not in ("content", "tool_calls", "role")
    )
    return {
        "reasoning_field": field,
        "signed": signed,
        "message_attrs": attrs,
        "raw": payload if isinstance(payload, (list, dict)) else _preview(payload),
        "payload_preview": _preview(payload),
        "error": "",
    }


def probe_reasoning_replay(client, model: str, log_: ProbeLog, param: str,
                           shape: dict) -> dict:
    """6) 把 reasoning 回传进下一轮：400 还是接受。"""
    field = shape.get("reasoning_field", "none")
    if field == "none":
        return {"replay": "nothing_to_replay", "accepted": None}
    raw = shape.get("raw")
    assistant_msg: dict[str, Any] = {"role": "assistant", "content": "已根据规则给出答复。"}
    if field == "reasoning_content":
        assistant_msg["reasoning_content"] = raw if isinstance(raw, str) else ""
    else:
        # thinking block 形态：原样回传 content 块列表（含 signature）
        assistant_msg["content"] = raw if isinstance(raw, list) else assistant_msg["content"]
    response, err = _call(
        client, log_, "reasoning_replay",
        model=model,
        messages=[
            {"role": "user", "content": "一个订单 20 天未拆封，能退款吗？"},
            assistant_msg,
            {"role": "user", "content": "请补充一句对时效的说明。"},
        ],
        **{param: 512},
    )
    if response is not None:
        return {"replay": "accepted", "accepted": True}
    if err is not None and _is_bad_request(err):
        return {"replay": "rejected", "accepted": False, "error": str(err)}
    return {
        "replay": f"error:{type(err).__name__ if err else 'unknown'}",
        "accepted": None,
    }


def _text_of(response) -> str:
    message = _message(response)
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(item.get("text") or item.get("thinking") or "")
            for item in content if isinstance(item, dict)
        )
    return ""


def _preview(value: Any, limit: int = 400) -> str:
    if value is None:
        return ""
    text = value if isinstance(value, str) else json.dumps(
        value, ensure_ascii=False, default=str,
    )
    return text[:limit] + ("…" if len(text) > limit else "")


# ============================================================
# 画像合成
# ============================================================
def _token_param(tools: dict, forced: dict, max_tokens: dict) -> str:
    """max_tokens 被 400 拒绝时改用 max_completion_tokens（o 系列语义）。"""
    if max_tokens.get("accepted"):
        return "max_tokens"
    if max_tokens.get("error") and "max_completion_tokens" in max_tokens["error"]:
        return "max_completion_tokens"
    # 400 但错误信息没提参数名：用工具调用探针的错误信息再判一次
    for source in (forced, tools):
        error = str(source.get("error") or "")
        if "max_completion_tokens" in error:
            return "max_completion_tokens"
    return "max_tokens"


def _return_policy(shape: dict, replay: dict) -> str:
    field = shape.get("reasoning_field", "none")
    if field == "none":
        return "internal"          # 不返回原文（厂商内部管理，无物可回传）
    if replay.get("replay") == "rejected":
        return "forbidden"         # 明确禁止回传（DeepSeek 语义）
    if replay.get("accepted") and shape.get("signed"):
        return "required_signed"   # 必须带 signature 原样回传（Claude thinking）
    return "internal"


def build_profile(probes: dict) -> dict:
    shape = probes["reasoning_shape"]
    replay = probes["reasoning_replay"]
    return {
        "supports_tools": bool(probes["tools"]["supports_tools"]),
        "supports_forced_tool_choice": bool(
            probes["forced_tool_choice"].get("honored")
        ),
        "temperature_mode": probes["temperature"]["temperature_mode"],
        "reasoning_field": shape.get("reasoning_field", "none"),
        "reasoning_return_policy": _return_policy(shape, replay),
        "max_tokens_param": probes["max_tokens_param"],
        "min_max_tokens": 8192,
        "reasoning_budget_param": (
            "thinking" if shape.get("signed") else ""
        ),
        "reasoning_budget_tokens": 8192 if shape.get("signed") else 0,
    }


def summarize_latency(log_: ProbeLog) -> dict:
    samples = log_.latencies()
    if not samples:
        return {"p50_ms": None, "p95_ms": None, "samples": 0}
    ordered = sorted(samples)
    p50 = statistics.median(ordered)
    # 样本量小（十余次）时用最近秩近似，只作为 T8 定参的起点
    index = min(len(ordered) - 1, max(0, int(round(0.95 * (len(ordered) - 1)))))
    return {
        "p50_ms": round(p50, 1),
        "p95_ms": round(ordered[index], 1),
        "min_ms": round(ordered[0], 1),
        "max_ms": round(ordered[-1], 1),
        "samples": len(ordered),
    }


def summarize_reasoning(log_: ProbeLog) -> dict:
    tokens = log_.reasoning_tokens()
    if not tokens:
        return {"typical": 0, "max": 0, "samples": 0}
    return {
        "typical": int(statistics.median(tokens)),
        "max": int(max(tokens)),
        "samples": len(tokens),
    }


# ============================================================
# dry-run（无网络）
# ============================================================
def _scripted_client(scripted: str = "deepseek-reasoner"):
    """脚本化假客户端：只验证脚本自身的接线与异常路径（无网络、无凭证）。"""
    def _reasoning_message(content, reasoning):
        return SimpleNamespace(
            content=content, tool_calls=None, reasoning_content=reasoning,
            role="assistant",
        )

    def _tool_message():
        tc = SimpleNamespace(
            id="call_1", type="function",
            function=SimpleNamespace(
                name="get_order_status", arguments='{"order_id":"A12345"}',
            ),
        )
        return SimpleNamespace(content=None, tool_calls=[tc], role="assistant")

    def create(**kwargs):
        usage = SimpleNamespace(
            prompt_tokens=120, completion_tokens=900, total_tokens=1020,
            completion_tokens_details=SimpleNamespace(reasoning_tokens=700),
        )
        message = _tool_message() if kwargs.get("tools") else _reasoning_message(
            "订单已发货。", "先确认时效，再看是否拆封……",
        )
        if "temperature" in kwargs and kwargs["temperature"] == 0.7:
            message = _reasoning_message("随机一点的答案。", "思考……")
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason="stop")],
            usage=usage, model=kwargs.get("model", scripted),
        )

    completions = SimpleNamespace(create=create)
    return SimpleNamespace(chat=SimpleNamespace(completions=completions))


# ============================================================
# 主流程
# ============================================================
def run_probes(client, model: str) -> tuple[dict, ProbeLog]:
    log_ = ProbeLog()
    probes: dict[str, Any] = {}
    probes["tools"] = probe_tools(client, model, log_)
    probes["forced_tool_choice"] = probe_forced_tool_choice(client, model, log_)
    probes["temperature"] = probe_temperature(client, model, log_)
    # max_tokens 参数名：先按 max_tokens 试；被拒时换 max_completion_tokens 重试
    max_tokens = probe_max_tokens(client, model, log_, "max_tokens")
    param = _token_param(probes["tools"], probes["forced_tool_choice"], max_tokens)
    if param != "max_tokens" and not max_tokens.get("accepted"):
        max_tokens = probe_max_tokens(client, model, log_, "max_completion_tokens")
    probes["max_tokens_param"] = param
    probes["max_tokens"] = max_tokens
    probes["reasoning_shape"] = probe_reasoning_shape(client, model, log_, param)
    probes["reasoning_replay"] = probe_reasoning_replay(
        client, model, log_, param, probes["reasoning_shape"],
    )
    return probes, log_


def build_report(model: str, probes: dict, log_: ProbeLog) -> dict:
    profile = build_profile(probes)
    return {
        "probe_version": PROBE_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": model,
        "endpoint": settings.openai_base_url,
        "gate": {
            "rule": "挂 tools 调用业务工具必须返回 tool_calls（不支持则计划终止）",
            "passed": bool(probes["tools"]["supports_tools"]),
        },
        "profile": profile,
        "latency": summarize_latency(log_),
        "reasoning_tokens": summarize_reasoning(log_),
        "probes": probes,
        "calls": log_.calls,
    }


def _print_conclusions(report: dict) -> None:
    profile = report["profile"]
    latency = report["latency"]
    reasoning = report["reasoning_tokens"]
    gate = report["gate"]
    log.info("=" * 66)
    log.info("T0 探针结论 model=%s", report["model"])
    log.info("=" * 66)
    log.info("门禁（tools→tool_calls）: %s", "PASS" if gate["passed"] else "FAIL")
    log.info("画像: temperature_mode=%s reasoning_field=%s return_policy=%s",
             profile["temperature_mode"], profile["reasoning_field"],
             profile["reasoning_return_policy"])
    log.info("      forced_tool_choice=%s max_tokens_param=%s（T1 画像取值）",
             profile["supports_forced_tool_choice"], profile["max_tokens_param"])
    log.info("延迟: P50=%sms P95=%sms（n=%s）→ T8 定 timeout 与轮次预算",
             latency["p50_ms"], latency["p95_ms"], latency["samples"])
    log.info("推理 token: 典型=%s 峰值=%s → T7 日预算校准",
             reasoning["typical"], reasoning["max"])
    if not gate["passed"]:
        log.error(
            "门禁未通过：目标模型未能返回 tool_calls。按计划**终止**本次适配"
            "（文本协议回退不在计划范围内，是另一个量级的工程）。",
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="目标模型能力探针（T0 门禁）")
    parser.add_argument("--model", default=settings.model_name)
    parser.add_argument("--base-url", default=settings.openai_base_url)
    parser.add_argument("--api-key", default=settings.openai_api_key)
    parser.add_argument(
        "--out", default="app/sessions/model_profile.json",
        help="画像 JSON 输出路径（建议 gitignore；用 --emit-overrides 灌配置）",
    )
    parser.add_argument(
        "--emit-overrides", action="store_true",
        help="打印可直接粘贴到 MODEL_PROFILE_OVERRIDES 的 JSON",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="脚本化假客户端：只验证脚本接线，不发网络请求（无需凭证）",
    )
    args = parser.parse_args(argv)

    if args.dry_run:
        client = _scripted_client(args.model)
        log.info("dry-run：使用脚本化假客户端，不发起真实请求")
    else:
        if not args.api_key:
            log.error("缺少 API key（--api-key 或 OPENAI_API_KEY），无法探针")
            return 3
        from openai import OpenAI

        client = OpenAI(api_key=args.api_key, base_url=args.base_url)

    probes, log_ = run_probes(client, args.model)
    report = build_report(args.model, probes, log_)
    _print_conclusions(report)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    log.info("画像已写入 %s", out)

    if args.emit_overrides:
        overrides = {args.model: report["profile"]}
        log.info("MODEL_PROFILE_OVERRIDES=%s",
                 json.dumps(overrides, ensure_ascii=False, separators=(",", ":")))

    return 0 if report["gate"]["passed"] else GATE_FAILED_EXIT


if __name__ == "__main__":
    sys.exit(main())
