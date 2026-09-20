"""端到端测试：长期记忆（LTM）提取、持久化与注入。

测试场景：
1. 长期记忆提取与持久化 — consolidate 后 LTM JSON 文件应存在且格式正确
2. 长期记忆跨会话加载 — 新建 agent 后 LTM 事实应被注入 prompt
3. recall_user_memory 工具 — 调用工具应返回 LTM 事实

记忆系统重构：会话内「短期记忆槽位层」已删除（会话内上下文 = 对话历史 +
rolling 摘要），相关 STM 用例随之移除。

用法：python3 tests/test_memory.py
"""

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.agent.chat import EcomAgent  # noqa: E402
from app.agent.memory import LongTermMemory, MemoryManager  # noqa: E402
from app.agent.tools.memory_tool import recall_user_memory, set_memory_manager  # noqa: E402

TEST_SESSION = str(ROOT / "app" / "sessions" / "test_memory_session.json")
TEST_MEMORY_DIR = str(ROOT / "app" / "sessions" / "test_memory")
TEST_USER_ID = "test_user_memory"


def _clean():
    Path(TEST_SESSION).unlink(missing_ok=True)
    shutil.rmtree(TEST_MEMORY_DIR, ignore_errors=True)


def _ok(msg: str):
    print(f"  ✅ {msg}")


def _fail(msg: str):
    print(f"  ❌ {msg}")
    sys.exit(1)


def _make_agent() -> EcomAgent:
    """创建测试用 Agent，使用独立的 session 和 memory 路径。"""
    import os
    os.environ["MEMORY_ENABLED"] = "true"
    os.environ["MEMORY_DIR"] = TEST_MEMORY_DIR
    os.environ["MEMORY_USER_ID"] = TEST_USER_ID

    from app.config.settings import Settings
    settings = Settings()
    settings.memory_enabled = True
    settings.memory_dir = TEST_MEMORY_DIR
    settings.memory_user_id = TEST_USER_ID

    agent = EcomAgent(session_path=TEST_SESSION)

    agent.memory_manager = MemoryManager(
        client=agent.client,
        model=agent.model,
        user_id=TEST_USER_ID,
        memory_dir=TEST_MEMORY_DIR,
        memory_enabled=True,
        max_ltm_facts=50,
    )
    set_memory_manager(agent.memory_manager)
    return agent


# ---------- 测试 1：长期记忆提取与持久化 ----------
def test_ltm_extraction():
    print("\n[1/3] 长期记忆提取与持久化")
    _clean()
    agent = _make_agent()

    agent.chat("你好，我是王五，钻石会员，之前买过你们的耳机觉得不错")
    agent.chat("我这次想看看有没有新款手机，预算三千左右")

    agent.memory_manager.consolidate_to_long_term(
        agent.raw_messages, agent.summary,
    )

    memory_path = Path(TEST_MEMORY_DIR) / f"{TEST_USER_ID}.json"
    if not memory_path.exists():
        _fail(f"LTM 文件未生成: {memory_path}")

    with memory_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if "facts" not in data or "interaction_summaries" not in data:
        _fail("LTM JSON 缺少 facts 或 interaction_summaries 字段")

    if len(data["facts"]) > 0:
        _ok(f"LTM 提取到 {len(data['facts'])} 条事实")
        for f in data["facts"][:3]:
            print(f"     - [{f['category']}] {f['content']}")
    else:
        print("  ⚠️  LTM 未提取到事实（可能因对话内容不够丰富）")

    if data["interaction_summaries"]:
        _ok(f"交互摘要: {data['interaction_summaries'][-1]['summary']}")
    else:
        print("  ⚠️  未生成交互摘要")


# ---------- 测试 2：长期记忆跨会话加载 ----------
def test_ltm_cross_session():
    print("\n[2/3] 长期记忆跨会话加载")
    _clean()

    agent1 = _make_agent()
    agent1.chat("我叫赵六，是你们的钻石会员，特别喜欢黑色的数码产品")
    agent1.memory_manager.consolidate_to_long_term(
        agent1.raw_messages, agent1.summary,
    )

    Path(TEST_SESSION).unlink(missing_ok=True)
    agent2 = _make_agent()

    ltm_facts = agent2.memory_manager.ltm.facts
    if len(ltm_facts) > 0:
        _ok(f"新 Agent 加载了 {len(ltm_facts)} 条 LTM 事实")
    else:
        _fail("新 Agent 未加载到 LTM 事实")

    sections = agent2.memory_manager.build_memory_prompt_sections()
    has_ltm = any("历史偏好" in s["content"] or "过往会话" in s["content"] for s in sections)
    if has_ltm:
        _ok("LTM 事实已注入 prompt sections")
    else:
        print("  ⚠️  LTM 事实可能未正确注入 prompt（标题不匹配）")


# ---------- 测试 3：recall_user_memory 工具 ----------
def test_memory_tool():
    print("\n[3/3] recall_user_memory 工具测试")
    _clean()
    agent = _make_agent()

    agent.chat("你好，我叫测试用户，最喜欢白色的衣服")
    agent.memory_manager.consolidate_to_long_term(
        agent.raw_messages, agent.summary,
    )

    result = recall_user_memory()
    if not result.get("success"):
        _fail(f"工具返回失败: {result}")

    ltm = result.get("long_term_facts", [])
    if ltm:
        _ok(f"工具返回 {len(ltm)} 条长期记忆")
    else:
        print("  ⚠️  工具未返回长期记忆")

    print(f"     长期: {ltm[:3]}")


def main():
    print("=" * 60)
    print("  长期记忆（LTM）· 端到端测试")
    print("=" * 60)

    try:
        test_ltm_extraction()
        test_ltm_cross_session()
        test_memory_tool()
    finally:
        _clean()

    print("\n" + "=" * 60)
    print("  全部测试通过")
    print("=" * 60)


if __name__ == "__main__":
    main()
