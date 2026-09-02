"""离线构建知识库向量索引（第10期：版本化 generation + 单写者锁）。

用法：
  # 默认走 settings.rag_backend
  python app/scripts/build_kb_index.py

  # 显式指定后端，便于一份知识库同时构建两种后端的索引做对比
  python app/scripts/build_kb_index.py --backend numpy
  python app/scripts/build_kb_index.py --backend chroma

流程：
  1. 获取 evolution 单写者锁（与 pipeline 互斥）。
  2. IndexBuildService.build：扫描（含 evolved/ 子目录）→ 向量化 → 写版本化目标 → 验证。
  3. IndexBuildService.activate：切换 generation 指针 + 清理旧代。
不触碰 ledger（ledger 只归 pipeline 管）。
"""

import argparse
import sys
from pathlib import Path
from app.observability.logging import get_logger
log = get_logger("app.scripts.build_kb_index")


ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from app.config.settings import settings  # noqa: E402
from app.agent.rag.embedder import create_embedder  # noqa: E402
from app.agent.rag.parsers import chunk_kb_dir  # noqa: E402
from app.evolution.generation import GenerationStore  # noqa: E402
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
    args = parser.parse_args()

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
        service = IndexBuildService(
            embedder=embedder,
            kb_dir=kb_dir,
            generation_store=GenerationStore(ROOT / settings.kb_generation_path),
            # 7.1：多格式接入（md/txt/pdf/docx/html，含 parent-child 装配）
            chunker=chunk_kb_dir,
            # v7：手工全量构建会切 alias → strict（坏文件中止，不静默丢知识）
            strict_build=True,
        )

        log.info("\n[1/3] 扫描并切分知识文档（7.1 多格式 + parent-child）...")
        info = service.build(args.backend)
        log.info(f"   已构建 {service.last_built_size} 个 chunk")

        log.info("\n[2/3] 验证版本化索引...")
        log.info(f"   generation : {info.generation_id}")
        log.info(f"   目标       : {info.target}")

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