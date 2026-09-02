"""对象存储抽象（阶段二 2.4 / 2.8 完成化）：知识文档 / evolution turns /
上传分片 / 上传原件落盘。

- LocalDirObjectStore：开发/测试（目录即桶）。
- S3ObjectStore：S3/OSS 兼容（boto3，endpoint_url 支持 OSS/MinIO）。

2.8 协议补全：delete / delete_prefix / healthcheck 双实现同步；
S3ChunkStorage.finalize_object 在正式对象写成功后删除临时对象；
delete_temp / delete_chunks 不再空操作。
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Optional

from app.stores.base import ObjectStore


class ObjectStoreUnavailable(Exception):
    """对象存储不可用（模块缺失/连接失败），调用方应降级本地。"""


class LocalDirObjectStore:
    """目录即对象存储：key = 相对路径（2.8 补全 delete/delete_prefix/healthcheck）。"""

    def __init__(self, root: str | Path):
        self._root = Path(root)

    def _path(self, key: str) -> Path:
        return self._root / key

    def put(self, key: str, data: bytes) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def get(self, key: str) -> Optional[bytes]:
        path = self._path(key)
        return path.read_bytes() if path.exists() else None

    def list(self, prefix: str = "") -> list[str]:
        root = self._root / prefix
        if not root.exists():
            return []
        return [
            str(p.relative_to(self._root)).replace(os.sep, "/")
            for p in root.rglob("*") if p.is_file()
        ]

    def delete(self, key: str) -> None:
        self._path(key).unlink(missing_ok=True)

    def delete_prefix(self, prefix: str) -> None:
        root = self._root / prefix
        if root.exists() and root.is_dir():
            shutil.rmtree(root, ignore_errors=True)

    def healthcheck(self) -> bool:
        probe = ".healthcheck"
        try:
            self.put(probe, b"ok")
            return self.get(probe) == b"ok"
        except Exception:  # noqa: BLE001 —— 探活失败即不可用
            return False


class S3ObjectStore:
    """S3/OSS 兼容实现（boto3 可选依赖，配置 endpoint_url 支持 OSS/MinIO）。

    2.8：delete（单对象，幂等）、delete_prefix（列举后批量删除）、
    healthcheck（put+get 探针）。所有操作显式错误包装为 ObjectStoreUnavailable。
    """

    def __init__(self, bucket: str, endpoint_url: str = "", region: str = "cn-north-1",
                 client=None, access_key: str = "", secret_key: str = ""):
        if client is not None:
            self._client = client  # 测试注入（botocore Stubber / 本地假客户端）
        else:
            try:
                import boto3  # noqa: F401
            except ImportError as e:
                raise ObjectStoreUnavailable("boto3 未安装，无法使用对象存储") from e
            try:
                from botocore.config import Config

                cfg = Config(connect_timeout=3, retries={"max_attempts": 2})
                kwargs: dict = {"region_name": region, "config": cfg}
                if endpoint_url:
                    kwargs["endpoint_url"] = endpoint_url
                if access_key:
                    kwargs["aws_access_key_id"] = access_key
                    kwargs["aws_secret_access_key"] = secret_key
                self._client = boto3.client("s3", **kwargs)
            except Exception as e:  # noqa: BLE001
                raise ObjectStoreUnavailable(f"S3 客户端初始化失败: {e}") from e
        self._bucket = bucket

    def put(self, key: str, data: bytes) -> None:
        try:
            self._client.put_object(Bucket=self._bucket, Key=key, Body=data)
        except Exception as e:  # noqa: BLE001
            from app.observability.metrics import record_object_store_failure
            record_object_store_failure("put")
            raise ObjectStoreUnavailable(f"S3 put 失败: {e}") from e

    def get(self, key: str) -> Optional[bytes]:
        try:
            resp = self._client.get_object(Bucket=self._bucket, Key=key)
            return resp["Body"].read()
        except Exception as e:  # noqa: BLE001 —— boto3 为可选依赖，按响应字段判型
            response = getattr(e, "response", {}) or {}
            if not isinstance(response, dict):
                response = {}
            error = response.get("Error", {}) or {}
            metadata = response.get("ResponseMetadata", {}) or {}
            code = str(error.get("Code", ""))
            status = metadata.get("HTTPStatusCode")
            if code in {"NoSuchKey", "NotFound", "404"} or status == 404:
                return None
            from app.observability.metrics import record_object_store_failure
            record_object_store_failure("get")
            raise ObjectStoreUnavailable(f"S3 get 失败（{key}）: {e}") from e

    def list(self, prefix: str = "") -> list[str]:
        try:
            keys = []
            paginator = self._client.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=self._bucket, Prefix=prefix):
                keys.extend(obj["Key"] for obj in page.get("Contents", []))
            return keys
        except Exception as e:  # noqa: BLE001
            from app.observability.metrics import record_object_store_failure
            record_object_store_failure("list")
            raise ObjectStoreUnavailable(f"S3 list 失败: {e}") from e

    def delete(self, key: str) -> None:
        """删除单对象（幂等：NoSuchKey 视为成功）。"""
        try:
            self._client.delete_object(Bucket=self._bucket, Key=key)
        except Exception as e:  # noqa: BLE001
            from app.observability.metrics import record_object_store_failure
            record_object_store_failure("delete")
            raise ObjectStoreUnavailable(f"S3 delete 失败（{key}）: {e}") from e

    def delete_prefix(self, prefix: str) -> None:
        """删除 prefix 下全部对象（列举→批量删除，幂等）。"""
        try:
            keys = self.list(prefix)
            if not keys:
                return
            self._client.delete_objects(
                Bucket=self._bucket,
                Delete={"Objects": [{"Key": k} for k in keys]},
            )
        except Exception as e:  # noqa: BLE001
            from app.observability.metrics import record_object_store_failure
            record_object_store_failure("delete_prefix")
            raise ObjectStoreUnavailable(f"S3 delete_prefix 失败（{prefix}）: {e}") from e

    def healthcheck(self) -> bool:
        probe = ".healthcheck"
        try:
            self.put(probe, b"ok")
            return self.get(probe) == b"ok"
        except Exception:  # noqa: BLE001
            return False