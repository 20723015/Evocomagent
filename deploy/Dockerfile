# 阶段二 2.5：multi-stage 容器化（生产镜像不含 .env、不含源码缓存）
# 构建（在有 Docker 的机器上）：docker build -f deploy/Dockerfile -t ecom-agent:latest .
# 注意：镜像内不携带 .env；配置全部走环境变量（pydantic-settings 原生支持）。

# ---------- builder ----------
FROM python:3.12-slim AS builder

WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ---------- runtime ----------
FROM python:3.12-slim AS runtime

# 非 root 运行 + 时区
RUN useradd -m -u 10001 appuser \
    && apt-get update && apt-get install -y --no-install-recommends tzdata curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /install /usr/local
COPY app ./app
COPY main.py ./
COPY requirements.txt ./
# 2.9：迁移 SQL 随镜像分发（migrate_db.py 默认读 deploy/sql）
COPY deploy/sql ./deploy/sql

USER appuser

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

EXPOSE 8000

# 优雅退出：SIGTERM 交给 uvicorn（停接新请求 → drain 在途 → 写回），
# 由 Helm chart 的 terminationGracePeriodSeconds 给出足够等待时间。
HEALTHCHECK --interval=10s --timeout=3s --retries=5 \
    CMD curl -fsS http://127.0.0.1:8000/healthz || exit 1

CMD ["python", "-m", "uvicorn", "app.server.main:app", "--host", "0.0.0.0", "--port", "8000"]
