"""离线构建知识库向量索引（第10期：版本化 generation + 单写者锁）。

用法：
  # 默认走 settings.rag_backend
  python app/scripts/build_kb_index.py

  # 显式指定后端，便于一份知识库同时构建两种后端的索引做对比
  python app/scripts/build_kb_index.py --backend numpy
  python app/scripts/build_kb_index.py --backend chroma

  # 只构建候选 generation，供发布评测；不切换线上 alias / 指针
  python app/scripts/build_kb_index.py --backend es --no-activate \
    --json-out /tmp/rag_candidate.json

  # 门禁通过后激活同一个候选 generation
  python app/scripts/build_kb_index.py --backend es --activate-candidate \
    /tmp/rag_candidate.json

流程：
  1. 获取 evolution 单写者锁（与 pipeline 互斥）。
  2. IndexBuildService.build：扫描（含 evolved/ 子目录）→ 向量化 → 写版本化目标 → 验证。
  3. 默认由 IndexBuildService.activate 切换 generation 指针 + 清理旧代；
     --no-activate 只写候选描述，--activate-candidate 在门禁通过后提交该候选。
不触碰 ledger（ledger 只归 pipeline 管）。
"""

import argparse
import json
import sys
from pathlib import Path
from app.observability.logging import get_logger
log = get_logger("app.scripts.build_kb_index")


ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from app.config.settings import settings  # noqa: E402
from app.agent.rag.embedder import create_embedder  # noqa: E402
from app.agent.rag.fingerprint import config_fingerprint  # noqa: E402
from app.agent.rag.parsers import chunk_kb_dir  # noqa: E402
from app.evolution.generation import GenerationInfo, GenerationStore  # noqa: E402
from app.evolution.index_service import IndexBuildService  # noqa: E402
from app.stores.kb_write_lock import (  # noqa: E402
    KbWriteLockBackendError,
    KbWriteLockError,
    get_kb_write_lock,
)
from app.stores.redis_client import get_redis  # noqa: E402
from app.stores.sql.engine import get_engine  # noqa: E402
from app.evolution.lock import CrossHostLockError, LockHeldError  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="构建知识库向量索引（版本化 generation）")
    parser.add_argument(
        "--backend",
        choices=["numpy", "chroma", "es"],
        default=settings.rag_backend,
        help=f"向量后端（默认: {settings.rag_backend}；es 需 ES_URL 可达）",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--no-activate", action="store_true",
        help="只构建并验证候选 generation，不切 alias/指针",
    )
    mode.add_argument(
        "--activate-candidate", metavar="PATH",
        help="激活由 --no-activate --json-out 生成的候选描述，不重新构建",
    )
    parser.add_argument(
        "--json-out", default="",
        help="将构建出的候选 generation 描述写入 JSON（--no-activate 时必填）",
    )
    args = parser.parse_args()
    if args.no_activate and not args.json_out:
        parser.error("--no-activate 要求同时提供 --json-out")
    if args.activate_candidate and args.json_out:
        parser.error("--activate-candidate 不可与 --json-out 同时使用")

    kb_dir = ROOT / settings.kb_dir
    if not kb_dir.exists():
        log.info(f"❌ 知识库目录不存在: {kb_dir}")
        sys.exit(1)

    # v7 统一写锁（部署期选择：mysql/redis/file；auto 双不可用 → 显式报错）
    try:
        lock = get_kb_write_lock(engine=get_engine(), redis=get_redis())
    except KbWriteLockBackendError as e:
        log.info(f"❌ {e}")
        sys.exit(1)
    try:
        lock.acquire(phase="build")
    except (KbWriteLockError, CrossHostLockError, LockHeldError) as e:
        log.info(f"❌ 构建被其他进程占用：{e}")
        sys.exit(1)

    log.info("=" * 60)
    log.info("  并夕夕 · 知识库索引构建（版本化）")
    log.info(f"  后端      : {args.backend}")
    log.info(f"  源目录    : {kb_dir}")
    log.info(f"  Embedding : {settings.embedding_model}")
    log.info("=" * 60)

    try:
        embedder = create_embedder()
        # 指针必须经 Redis 共享（strict_shared）：否则多 Pod 下本工具只写
        # 本地文件、线上检索器读 Redis 旧代——检索器将解析到已删除的索引
        # （2026-09 实测：手工构建后 535 题评测报 index_not_found）。
        _redis = get_redis()
        generation_store = GenerationStore(
            ROOT / settings.kb_generation_path,
            redis_client=_redis,
            strict_shared=_redis is not None,
        )
        service = IndexBuildService(
            embedder=embedder,
            kb_dir=kb_dir,
            generation_store=generation_store,
            # 7.1：多格式接入（md/txt/pdf/docx/html，含 parent-child 装配）
            chunker=chunk_kb_dir,
            # v7：手工全量构建会切 alias → strict（坏文件中止，不静默丢知识）
            strict_build=True,
        )

        if args.activate_candidate:
            candidate_path = Path(args.activate_candidate)
            payload = json.loads(candidate_path.read_text(encoding="utf-8"))
            if payload.get("protocol") != "kb-generation-candidate-v1":
                raise ValueError("候选描述 protocol 非法或缺失")
            candidate_backend = str(payload.get("backend", "")).lower()
            if candidate_backend != args.backend:
                raise ValueError(
                    f"候选后端 {candidate_backend or '缺失'} 与 --backend {args.backend} 不一致"
                )
            info = GenerationInfo.from_dict(payload["generation"])
            expected_target = service.target_for(args.backend, info.generation_id)
            if info.target != expected_target:
                raise ValueError("候选 target 与 generation_id/当前配置不匹配")
            if payload.get("config_fingerprint") != config_fingerprint():
                raise ValueError("候选索引配置指纹与当前配置不一致")
            if "expected_active_generation_id" not in payload:
                raise ValueError("候选描述缺少 expected_active_generation_id")
            expected_active = str(payload["expected_active_generation_id"] or "")
            current = generation_store.active(args.backend)
            current_id = current.generation_id if current is not None else ""
            if current_id != expected_active:
                raise RuntimeError(
                    "候选构建后活动 generation 已变化，拒绝覆盖并发发布："
                    f" expected={expected_active or '<none>'},"
                    f" current={current_id or '<none>'}"
                )
            log.info("\n[1/1] 激活已通过门禁的候选 generation...")
            service.activate(args.backend, info)
            log.info(f"   已激活 {args.backend} generation={info.generation_id}")
        else:
            active_before = generation_store.active(args.backend)
            log.info("\n[1/3] 扫描并切分知识文档（7.1 多格式 + parent-child）...")
            info = service.build(args.backend)
            log.info(f"   已构建 {service.last_built_size} 个 chunk")

            log.info("\n[2/3] 验证版本化索引...")
            log.info(f"   generation : {info.generation_id}")
            log.info(f"   目标       : {info.target}")
            if args.json_out:
                output = Path(args.json_out)
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(
                    json.dumps(
                        {
                            "protocol": "kb-generation-candidate-v1",
                            "backend": args.backend,
                            "generation": info.to_dict(),
                            "config_fingerprint": config_fingerprint(),
                            "expected_active_generation_id": (
                                active_before.generation_id if active_before else ""
                            ),
                        },
                        ensure_ascii=False, indent=2,
                    ),
                    encoding="utf-8",
                )
                log.info(f"   候选描述   : {output}")

            if args.no_activate:
                log.info("\n[3/3] 保持未激活，等待发布门禁完成。")
            else:
                log.info("\n[3/3] 切换 generation 并清理旧代...")
                service.activate(args.backend, info)
                log.info(f"   已激活 {args.backend} generation={info.generation_id}")

        log.info("\n🎉 索引构建完成。")
    except Exception as e:  # noqa: BLE001 —— 构建失败不切换 generation，旧索引可用
        log.info(f"❌ 构建失败（generation 未切换，旧索引仍可用）: {e}")
        sys.exit(1)
    finally:
        lock.release()


if __name__ == "__main__":
    main()
