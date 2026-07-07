#!/usr/bin/env python3
"""数据迁移:嵌入式 Qdrant → server 模式(阶段 E,docs/SCALE_OUT.md §7)。

compose 的 qdrant 是 server 模式、持久卷从零起。本脚本把裸机嵌入式库(~/rag_real/qdrant)的
全部 point 搬进 server。sidecar(~/rag_real/sidecar/*.json)是另一条数据,宿主机 cp -a 单独搬(见 §7)。

四个必踩的坑(缺一层即"假迁完"):
  1. 建 collection 复用 Store.ensure_collection —— 一次建 dense+sparse + 全 7 个 payload index。
     手抄漏一个 → server 端 ACL fail-closed 失效(acl_tenant/acl_allow 等无索引 → 过滤静默失准)。
  2. scroll 必带 with_vectors=True —— 默认不返向量,漏了迁进"无向量 point",检索全崩却不报错。
  3. named vectors(dense+sparse)原样透传;image-only chunk 无 sparse,保持缺省别补空。
  4. dst.count == src.count 才算迁完;point id 走 uuid5 确定性 → 幂等可重跑。

本脚本只做第 1 层(count)校验;向量真在/sidecar 覆盖/ACL fail-closed 三层在 SCALE_OUT §7,单独跑。

跑(slim 环境即可,纯 qdrant-client 不吃 GPU):
  # 先起 server:docker compose --env-file .env.compose up -d qdrant
  python scripts/migrate_to_server.py \
      --src ~/rag_real/qdrant --dst http://localhost:6333 --collection real --dense-dim 1024
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from qdrant_client import QdrantClient, models   # noqa: E402

from embedder.config import EmbedConfig           # noqa: E402
from embedder.store import Store                  # noqa: E402

BATCH = 256


def migrate(src_path: str, dst_url: str, collection: str, dense_dim: int) -> None:
    src_path = os.path.expanduser(src_path)
    if not os.path.isdir(src_path):
        raise SystemExit(f"源嵌入式库目录不存在:{src_path}(应指向 ~/rag_real/qdrant)")

    # 1) 先拿源库独占锁 —— 在对 dst 产生任何副作用【前】失败(评审 migrate-F2:否则先建好 dst 空 collection 再抛错,
    #    留下"dst 已建库但没迁"的中间态 + 裸 RuntimeError 不含前置指引)。
    print(f"[1/4] 打开源嵌入式库 {src_path}(独占锁;迁移期须停所有 pharos serve/index)…", flush=True)
    try:
        src = QdrantClient(path=src_path)
    except RuntimeError as e:
        raise SystemExit(f"源库被占用(通常是 pharos serve/index 仍在跑);先停所有访问 {src_path} 的进程再迁移。原始错误:{e}")

    # 1b) 校验源 collection 存在且**非空**——在对 dst 产生任何副作用【前】(阶段F 审查·migrate):否则源库没有
    #     该 collection / 点数为 0 时,下一步仍会在 dst 建一个空 collection,恰好击穿 pharos /readyz 的
    #     collection_missing 守卫(集合存在但 0 点 → readyz 报 ready → nginx 导流量到空库 → 全绿查空)。
    if not src.collection_exists(collection):
        raise SystemExit(f"源库无 collection {collection!r}(检查 --src {src_path} / --collection);未对 dst 做任何改动。")
    src_count = src.count(collection).count
    if src_count == 0:
        raise SystemExit(f"源 collection {collection!r} 为空(0 点);拒绝迁移空库(会在 dst 建空集合、误导 /readyz 判就绪)。")

    # 2) 在 dst 建 collection —— 复用 Store.ensure_collection(dense/sparse + 7 payload index,绝不手抄)
    dst_cfg = EmbedConfig(qdrant_url=dst_url, collection=collection, dense_dim=dense_dim)
    print(f"[2/4] 在 server 建 collection {collection!r}(复用 ensure_collection:dense/sparse + 7 payload index)…", flush=True)
    Store(dst_cfg).ensure_collection()   # 幂等:已存在且维度一致则跳过;维度不一致会 loud 报错
    dst = QdrantClient(url=dst_url)

    # 3) scroll 全量 point(with_vectors=True 必带),原样 upsert
    print("[3/4] scroll + upsert(with_vectors=True;named dense+sparse 原样透传)…", flush=True)
    offset = None
    migrated = 0
    while True:
        points, offset = src.scroll(
            collection, limit=BATCH, with_payload=True, with_vectors=True, offset=offset)
        if not points:
            break
        dst.upsert(collection, points=[
            models.PointStruct(id=p.id, vector=p.vector, payload=p.payload) for p in points])
        migrated += len(points)
        print(f"      migrated {migrated} …", flush=True)
        if offset is None:
            break

    # 4) count 校验(第一层;向量/sidecar/ACL 三层见 SCALE_OUT §7)
    sc = src.count(collection).count
    dc = dst.count(collection).count
    print(f"[4/4] count 校验:src={sc}  dst={dc}", flush=True)
    if sc != dc or dc == 0:      # dc==0 也算失败(空迁移无意义,且会误导 /readyz 判就绪)
        raise SystemExit(f"count 校验失败(src={sc}, dst={dc});迁移未完成或迁了空库,可重跑(幂等)。")
    # 评审 migrate-F1:退出语不得暗示"向量已验"——只验了点数,向量降级/payload 截断 count 仍相等(假迁完)。
    print(f"OK 点数迁移完成({dc} points);⚠ 仅验点数,向量/ACL【未校验】—— 必须继续跑 SCALE_OUT §7 三层校验"
          f"(向量真在 dense==dim/sparse 非空、sidecar 覆盖、ACL fail-closed)+ cp -a sidecar/。", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="嵌入式 Qdrant → server 迁移(阶段 E)")
    ap.add_argument("--src", default="~/rag_real/qdrant", help="源嵌入式库目录(默认 ~/rag_real/qdrant)")
    ap.add_argument("--dst", default="http://localhost:6333", help="目标 server URL")
    ap.add_argument("--collection", default="real")
    ap.add_argument("--dense-dim", type=int, default=1024)
    args = ap.parse_args()
    migrate(args.src, args.dst, args.collection, args.dense_dim)


if __name__ == "__main__":
    main()
