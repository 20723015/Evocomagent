"""模型画像（推理模型全量适配计划·T1）：按模型名描述厂商能力差异。

为什么需要：DeepSeek-reasoner / OpenAI o 系列 / Claude extended thinking 在
temperature 接受度、max_tokens 参数名、reasoning 字段形状、以及「能否/必须回传
reasoning」上两两矛盾（厂商差异速查表见 `docs/推理模型适配-实施记录.md`）。
按通用假设写代码会
400/被忽略，按模型画像分支才是可维护做法。

两条不变式（本模块的存在理由）：

1. **未命中画像 = 现状行为逐字节一致**：`resolve_profile()` 返回 None 时
   `apply_profile()` 是空操作——旧模型（`gpt-4o-mini`、`DeepSeek-V4-Flash` 等）
   的部署零行为变化，这是回归保护；
2. **参数改写收敛在唯一出口**：`app/llm/client.py` 的 `_wrap` 调用
   `apply_profile()`，17+ 个 `chat.completions.create` 调用点零改动。

画像来源优先级：`settings.model_profile_overrides`（探针输出直接灌入，免发版）
> 内置 registry（文档基线，**须经 probe_model_capability.py 实证**，见 T0）。

配置容错：覆写值非法（拼错的 mode/字段名）不抛异常，落回默认值并告警——
画像错误不该让每一轮请求都 500（N3 的「响失败」留给真正的参数不兼容）。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from typing import Any

from app.config.settings import settings
from app.observability.logging import get_logger

log = get_logger("app.llm.model_profile")

TEMPERATURE_MODES = ("free", "fixed_1", "ignored", "forbidden")
REASONING_FIELDS = ("none", "reasoning_content", "thinking_blocks")
RETURN_POLICIES = ("forbidden", "internal", "required_signed")
TOKEN_PARAMS = ("max_tokens", "max_completion_tokens")

# 思考预算参数名（reasoning_budget_param）中唯一需要嵌套构造的特例：
# Claude 的 thinking={"type":"enabled","budget_tokens":N}。其余按扁平参数注入。
THINKING_PARAM = "thinking"


@dataclass(frozen=True)
class ModelProfile:
    """单个模型（族）的能力画像。字段名与计划 T1 一一对应。"""

    name: str = "default"
    supports_tools: bool = True
    supports_forced_tool_choice: bool = True
    temperature_mode: str = "free"          # free | fixed_1 | ignored | forbidden
    reasoning_field: str = "none"           # none | reasoning_content | thinking_blocks
    reasoning_return_policy: str = "forbidden"  # forbidden | internal | required_signed
    max_tokens_param: str = "max_tokens"    # max_tokens | max_completion_tokens
    min_max_tokens: int = 0                 # 推理模型输出下限（思考 token 计入其中）
    reasoning_budget_param: str = ""        # 思考预算参数名（无则 ""）
    reasoning_budget_tokens: int = 0        # 思考预算取值（0 = 不注入）

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# ------------------------------------------------------------
# 内置 registry（文档基线；部署前必须由 T0 探针覆写为实测值）
# ------------------------------------------------------------
_DEEPSEEK_REASONING = dict(
    # reasoning_content 明确禁止回传；temperature 接受但被忽略
    temperature_mode="ignored",
    reasoning_field="reasoning_content",
    reasoning_return_policy="forbidden",
    max_tokens_param="max_tokens",
    min_max_tokens=8192,
)
_OPENAI_REASONING = dict(
    # 仅支持默认 temperature=1（传值 400）；输出上限参数改名 max_completion_tokens；
    # 不返回 reasoning 原文（厂商内部管理）
    temperature_mode="forbidden",
    reasoning_field="none",
    reasoning_return_policy="internal",
    max_tokens_param="max_completion_tokens",
    min_max_tokens=8192,
)
_CLAUDE_THINKING = dict(
    # thinking 开启时必须 temperature=1；thinking block 必须带 signature 原样回传
    temperature_mode="fixed_1",
    reasoning_field="thinking_blocks",
    reasoning_return_policy="required_signed",
    max_tokens_param="max_tokens",
    min_max_tokens=8192,
    reasoning_budget_param=THINKING_PARAM,
    reasoning_budget_tokens=8192,
)

REGISTRY: tuple[tuple[str, ModelProfile], ...] = (
    ("deepseek-reasoner", ModelProfile(name="deepseek-reasoner", **_DEEPSEEK_REASONING)),
    ("deepseek-r1", ModelProfile(name="deepseek-reasoner", **_DEEPSEEK_REASONING)),
    ("o1", ModelProfile(name="openai-reasoning", **_OPENAI_REASONING)),
    ("o3", ModelProfile(name="openai-reasoning", **_OPENAI_REASONING)),
    ("o4", ModelProfile(name="openai-reasoning", **_OPENAI_REASONING)),
    # 只匹配显式开启 thinking 的模型 ID：普通 Claude ID 的 temperature 语义与
    # 非推理模型一致，不应被强制成 1（需要时用 model_profile_overrides 显式指定）
    ("claude-3-7-sonnet-thinking", ModelProfile(name="claude-thinking", **_CLAUDE_THINKING)),
    ("claude-sonnet-4-thinking", ModelProfile(name="claude-thinking", **_CLAUDE_THINKING)),
    ("claude-opus-4-thinking", ModelProfile(name="claude-thinking", **_CLAUDE_THINKING)),
)

_REGISTRY_MAP: dict[str, ModelProfile] = {prefix: p for prefix, p in REGISTRY}

# 覆写解析缓存：(原始 JSON 字符串, 解析结果)。键为原始值——测试 monkeypatch
# settings 后自动失效，不需要清缓存钩子。
_OVERRIDE_CACHE: tuple[str, dict[str, ModelProfile]] | None = None

# 不支持 tools 的画像只告警一次/模型（避免每轮每次调用刷屏）
_WARNED_NO_TOOLS: set[str] = set()


# ------------------------------------------------------------
# 解析
# ------------------------------------------------------------
def _pick(value: Any, allowed: tuple[str, ...], default: str, field: str) -> str:
    text = str(value if value is not None else "").strip()
    if text in allowed:
        return text
    log.warning(
        "model_profile.invalid_field field=%s value=%r fallback=%s",
        field, value, default,
    )
    return default


def _pick_int(value: Any, default: int = 0) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return max(number, 0)


def _pick_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("true", "1", "yes"):
            return True
        if lowered in ("false", "0", "no"):
            return False
    if isinstance(value, int):
        return bool(value)
    return default


def profile_from_dict(data: Any, *, name: str = "override") -> ModelProfile | None:
    """从（探针/覆写）字典构造画像；非字典返回 None。"""
    if not isinstance(data, dict):
        return None
    known = {f.name for f in fields(ModelProfile)}
    unknown = sorted(set(data) - known)
    if unknown:
        log.warning("model_profile.unknown_fields name=%s fields=%s", name, unknown)
    return ModelProfile(
        name=str(data.get("name") or name),
        supports_tools=_pick_bool(data.get("supports_tools"), True),
        supports_forced_tool_choice=_pick_bool(
            data.get("supports_forced_tool_choice"), True,
        ),
        temperature_mode=_pick(
            data.get("temperature_mode"), TEMPERATURE_MODES, "free", "temperature_mode",
        ),
        reasoning_field=_pick(
            data.get("reasoning_field"), REASONING_FIELDS, "none", "reasoning_field",
        ),
        reasoning_return_policy=_pick(
            data.get("reasoning_return_policy"), RETURN_POLICIES,
            "forbidden", "reasoning_return_policy",
        ),
        max_tokens_param=_pick(
            data.get("max_tokens_param"), TOKEN_PARAMS, "max_tokens", "max_tokens_param",
        ),
        min_max_tokens=_pick_int(data.get("min_max_tokens"), 0),
        reasoning_budget_param=str(data.get("reasoning_budget_param") or ""),
        reasoning_budget_tokens=_pick_int(data.get("reasoning_budget_tokens"), 0),
    )


def _parse_overrides(raw: str) -> dict[str, ModelProfile]:
    text = str(raw or "").strip()
    if not text:
        return {}
    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, TypeError, ValueError):
        log.warning("model_profile.overrides_unparsable（已忽略）")
        return {}
    if not isinstance(payload, dict):
        log.warning("model_profile.overrides_not_object（已忽略）")
        return {}
    table: dict[str, ModelProfile] = {}
    for prefix, value in payload.items():
        profile = profile_from_dict(value, name=str(prefix))
        if profile is not None:
            table[str(prefix).strip().lower()] = profile
    return table


def overrides() -> dict[str, ModelProfile]:
    """当前 settings 的覆写表（按原始 JSON 字符串缓存）。"""
    global _OVERRIDE_CACHE
    raw = str(getattr(settings, "model_profile_overrides", "") or "")
    if _OVERRIDE_CACHE is not None and _OVERRIDE_CACHE[0] == raw:
        return _OVERRIDE_CACHE[1]
    table = _parse_overrides(raw)
    _OVERRIDE_CACHE = (raw, table)
    return table


def _match_prefix(table: dict[str, ModelProfile], name: str) -> ModelProfile | None:
    """最长前缀优先；键 `*` 为兜底通配（长度最短，因此最后才命中）。"""
    for prefix in sorted(table, key=len, reverse=True):
        if prefix == "*" or name.startswith(prefix):
            return table[prefix]
    return None


def resolve_profile(model: str) -> ModelProfile | None:
    """按模型名解析画像；未命中返回 None（= 调用方按现状行为处理）。"""
    name = str(model or "").strip().lower()
    if not name:
        return None
    return _match_prefix(overrides(), name) or _match_prefix(_REGISTRY_MAP, name)


def effective_profile(model: str) -> ModelProfile:
    """消费方（T3/T4/T9）用的画像：未命中返回全默认画像（= 现状语义）。"""
    return resolve_profile(model) or ModelProfile()


def temperature_honored(model: str) -> bool:
    """该画像下 temperature 是否被厂商尊重（free）。judge 守卫用（T9）。"""
    profile = resolve_profile(model)
    return profile is None or profile.temperature_mode == "free"


def profile_fingerprint(model: str) -> dict[str, Any]:
    """进评测 manifest 的画像指纹（T9）：含是否命中，保证新旧报告不可混淆。"""
    profile = resolve_profile(model)
    return {
        "matched": profile is not None,
        "profile": (profile or ModelProfile()).as_dict(),
    }


# ------------------------------------------------------------
# 统一参数改写（唯一出口）
# ------------------------------------------------------------
def apply_profile(kwargs: dict, model: str) -> dict:
    """按画像原地改写请求参数；无画像时逐字节不动。

    改写内容：temperature（删除/固定 1）、max_tokens 参数名与下限、
    思考预算参数注入。**不做** tools 剥离（不支持 tools 的模型由 T0 门禁拦在
    上线前，运行期剥离 tools 只会把 ReAct 变成静默的纯文本循环）。
    """
    profile = resolve_profile(model)
    if profile is None:
        return kwargs

    if profile.temperature_mode == "fixed_1":
        kwargs["temperature"] = 1.0
    elif profile.temperature_mode in ("ignored", "forbidden"):
        # ignored：厂商忽略该值，不再发送（避免厂商收紧后由忽略变 400）；
        # forbidden：发送即 400，必须删除
        kwargs.pop("temperature", None)

    param = profile.max_tokens_param or "max_tokens"
    if param != "max_tokens" and "max_tokens" in kwargs:
        kwargs[param] = kwargs.pop("max_tokens")
    if profile.min_max_tokens > 0:
        current = _pick_int(kwargs.get(param), 0)
        if current < profile.min_max_tokens:
            kwargs[param] = profile.min_max_tokens

    if profile.reasoning_budget_param and profile.reasoning_budget_tokens > 0:
        if profile.reasoning_budget_param == THINKING_PARAM:
            kwargs[THINKING_PARAM] = {
                "type": "enabled",
                "budget_tokens": profile.reasoning_budget_tokens,
            }
        else:
            kwargs[profile.reasoning_budget_param] = profile.reasoning_budget_tokens

    if not profile.supports_tools and kwargs.get("tools"):
        if profile.name not in _WARNED_NO_TOOLS:
            _WARNED_NO_TOOLS.add(profile.name)
            log.warning(
                "model_profile.tools_unsupported model=%s profile=%s（T0 门禁应拦在上线前）",
                model, profile.name,
            )
    return kwargs
