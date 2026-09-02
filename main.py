"""CLI 入口（阶段一 1.6）：与 API 共用 build_agent 工厂，仅多一个 --user 参数。

用法：python main.py --user u1
"""

import argparse

from app.config.settings import settings
from app.observability.logging import configure_logging, get_logger
from app.schemas.response import IntentType

log = get_logger("cli")

# 意图类型的中文映射
INTENT_LABELS = {
    IntentType.ORDER_QUERY: "订单查询",
    IntentType.RETURN_REQUEST: "退换货",
    IntentType.PRODUCT_CONSULT: "商品咨询",
    IntentType.COMPLAINT: "投诉",
    IntentType.AFTER_SALE: "售后服务",
    IntentType.PROMOTION: "优惠活动",
    IntentType.ACCOUNT: "账户问题",
    IntentType.GREETING: "打招呼",
    IntentType.OTHER: "其他",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="并夕夕智能客服 CLI")
    parser.add_argument(
        "--user", default="u1",
        help="用户标识（默认 u1=小明；会话/记忆按用户隔离）",
    )
    parser.add_argument("--session-id", default="", help="会话标识；默认使用该用户的默认会话")
    return parser.parse_args()


def main():
    # CLI：控制台彩色可读（结构化数据仍然一致，只是呈现不同）
    configure_logging(json_output=False)
    args = parse_args()

    from app.server.deps import build_agent, build_pod_components

    components = build_pod_components()
    agent = build_agent(args.user, args.session_id, components)
    mode = "Multi-Agent 协作模式" if settings.multi_agent_enabled else "ReAct + MCP + RAG"

    log.info("=" * 50)
    log.info(f"  并夕夕 · 智能客服「小夕」({mode})")
    log.info(f"  用户: {args.user} | 支持工具调用 + 政策检索 + 用户记忆 + 技能编排")
    log.info("  输入 quit/exit 退出, reset 重置, memory 查看记忆, skills 查看技能")
    log.info("=" * 50)
    log.info("")

    if agent.history_size > 0:
        log.info(f"💬 已恢复上次对话（{agent.history_size} 条历史）\n")

    while True:
        try:
            user_input = input("👤 你: ").strip()
        except (EOFError, KeyboardInterrupt):
            agent.save()
            agent.close()
            log.info("\n再见，欢迎下次光临！")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit"):
            agent.save()
            agent.close()
            log.info("再见，欢迎下次光临！")
            break

        if user_input.lower() == "reset":
            agent.reset()
            log.info("对话已重置。\n")
            continue

        if user_input.lower() == "skills":
            if hasattr(agent, "skill_manager") and agent.skill_manager.enabled:
                catalog = agent.skill_manager.get_catalog()
                log.info(f"\n--- 已加载 {len(catalog)} 个技能 ---")
                for s in catalog:
                    log.info(f"  - {s['name']}：{s['description']}")
                log.info("")
            else:
                log.info("技能系统未启用\n")
            continue

        if user_input.lower() == "memory":
            if hasattr(agent, "memory_manager") and agent.memory_manager.memory_enabled:
                stm = agent.memory_manager.stm
                ltm = agent.memory_manager.ltm
                log.info("\n--- 短期记忆（本次对话）---")
                if stm.facts:
                    for f in stm.facts:
                        log.info(f"  - {f}")
                else:
                    log.info("  （暂无）")
                log.info(f"\n--- 长期记忆（跨会话，用户: {ltm.user_id}）---")
                if ltm.facts:
                    for f in ltm.facts:
                        log.info(f"  - [{f.category}] {f.content}")
                else:
                    log.info("  （暂无）")
                if ltm.interaction_summaries:
                    log.info("\n--- 最近交互 ---")
                    for s in ltm.interaction_summaries[-3:]:
                        log.info(f"  - {s['summary']}")
                log.info("")
            else:
                log.info("记忆功能未启用\n")
            continue

        try:
            response = agent.chat(user_input)

            # 打印客服回复
            log.info(f"\n🤖 小夕: {response.reply}")

            # 打印结构化元信息
            intent_label = INTENT_LABELS.get(response.intent, response.intent)
            log.info(
                f"   [意图: {intent_label} | "
                f"置信度: {response.confidence:.0%} | "
                f"转人工: {'是' if response.requires_human else '否'}]"
            )

            if response.follow_up_question:
                log.info(f"   [追问: {response.follow_up_question}]")

            log.info("")

        except Exception as e:
            log.info(f"\n⚠️  出错了: {e}\n")


if __name__ == "__main__":
    main()
