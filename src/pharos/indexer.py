"""`pharos index`:把 MinerU 解析产物目录批量建成 Pharos 索引(qdrant + sidecar)。

产品化自引擎仓 mcp_server/index_real.py(那份保留作历史脚本):corpus/目标/collection/ACL
全部参数化,不再写死 parsed/ 与 ~/rag_real。语料要求:corpus 目录下每篇一个 MinerU 输出目录
(content_list.json + layout.json…),目录名即 doc_id,`<doc_type>__<name>` 前缀约定可选。

需 GPU(Qwen3-VL-Embedding-8B)。⚠ 嵌入式 Qdrant 单客户端:目标索引正被守护进程打开时
无法写入 —— 先停 `pharos serve` 再 index,或 index 到新目录后切换 PHAROS_INDEX_DIR。
"""
from __future__ import annotations

import os

from chunker import Chunker
from chunker.adapters.mineru import from_mineru_dir
from embedder import EmbedConfig, Embedder

from . import config


def detect_lang(elements) -> str:
    """按内容 CJK 比例判语言(比按 doc_type 猜可靠;引自引擎 index_real.py)。"""
    sample = "".join((e.text or "") for e in elements[:40])[:2000]
    if not sample:
        return "en"
    cjk = sum(1 for c in sample if "一" <= c <= "鿿")
    return "ch" if cjk > len(sample) * 0.12 else "en"


def run_index(cfg: config.PharosConfig, corpus: str | None = None, dest: str | None = None,
              collection: str | None = None, tenant: str | None = None, visibility: str = "public",
              allow: str = "", only: str | None = None, limit: int | None = None) -> int:
    allow_list = [a.strip() for a in allow.split(",") if a.strip()]
    # 评审修(C1):restricted + 空 allow = 任何身份都检索不到(fail-closed),但建库全程零告警,
    # 只在检索期表现为"空结果",极难排障 —— 在建库入口显式拒绝,要求给出 principals。
    if visibility == "restricted" and not allow_list:
        raise SystemExit("--visibility restricted 需要 --allow 至少一个 principal:"
                         "restricted+空 allow 的文档对任何身份都不可见(fail-closed),"
                         "建库会\"成功\"但全部静默检索不到。")
    corpus = corpus or cfg.corpus_dir
    if not corpus:
        raise SystemExit("未指定语料目录:请用 --corpus 或设 PHAROS_CORPUS_DIR"
                         "(MinerU 解析产物目录;数据留仓外,不随迁移入库)。")
    # 建库写路径 vs serve 读路径必须同源(阶段F 审查·S3 命门):serve 用 cfg.qdrant_path/cfg.sidecar_dir
    # (from_env 已按 PHAROS_QDRANT_PATH/PHAROS_SIDECAR_DIR 或 index_dir 派生)。indexer 若自己从 dest 派生
    # sidecar,遇到显式设了 PHAROS_SIDECAR_DIR 的部署就**写读分家** —— 建库写 dest/sidecar、serve 读
    # cfg.sidecar_dir,命中却取空 → 静默降级(single_chunk_degraded)。故:未显式 --dest 时直接用 cfg 同源路径;
    # 显式 --dest 才走 dest 派生(用户明确要建到别处,自负一致性)。
    explicit_dest = dest is not None
    dest = os.path.expanduser(dest or cfg.index_dir)
    qdrant_path = os.path.join(dest, "qdrant") if explicit_dest else cfg.qdrant_path
    sidecar_dir = os.path.join(dest, "sidecar") if explicit_dest else cfg.sidecar_dir
    collection = collection or cfg.collection
    acl = {"tenant": (tenant or cfg.tenant or "demo"), "allow": allow_list,
           "visibility": visibility, "unset": False}
    if not os.path.isdir(corpus):
        raise SystemExit(f"语料目录不存在:{corpus}")

    # 模型路径/gpu_name 也从 cfg 透传(阶段F 审查):否则定制 PHAROS_DENSE_MODEL_PATH 时建库用默认路径,
    # 与 serve 加载的模型不一致 —— 向量空间错位而无告警。
    ecfg = EmbedConfig(qdrant_path=qdrant_path, qdrant_url=cfg.qdrant_url,
                       sidecar_dir=sidecar_dir,
                       dense_dim=cfg.dense_dim, collection=collection,
                       dense_model_path=cfg.dense_model_path, rerank_model_path=cfg.rerank_model_path,
                       gpu_name_must_contain=cfg.gpu_name)   # server 模式(qdrant_url非空)写 server;qdrant_path 被 store 三分支忽略
    try:
        emb = Embedder(ecfg)
    except Exception as e:
        if "already accessed" in str(e):
            raise SystemExit(f"索引目录被占用(嵌入式 Qdrant 单客户端):{dest}\n"
                             f"先停掉 pharos serve 再 index,或用 --dest 指到新目录。") from e
        raise

    dirs = sorted(d for d in os.listdir(corpus) if os.path.isdir(os.path.join(corpus, d)))
    if only:
        dirs = [d for d in dirs if d.startswith(only)]
    if limit:
        dirs = dirs[:limit]
    print(f"{corpus} 下 {len(dirs)} 篇,建库 -> {dest} (collection={collection}, "
          f"acl tenant={acl['tenant']}/{acl['visibility']})", flush=True)
    ok = total = 0
    for d in dirs:
        ddir = os.path.join(corpus, d)
        doc_type = d.split("__")[0] if "__" in d else "unknown"
        try:
            els = from_mineru_dir(ddir)
            if not els:
                print(f"  跳过 {d}: 空", flush=True)
                continue
            lang = detect_lang(els)
            res = Chunker().chunk(els, doc_id=d, doc_type=doc_type, lang=lang,
                                  doc_meta={"title": d}, acl=acl)
            emb.index_document(d, els, res, image_root=ddir)
            ok += 1
            total += len(res.chunks)
            print(f"  [{ok:2d}] {doc_type:20s} {lang} {len(res.chunks):4d} chunk  {d[:44]}", flush=True)
        except Exception as e:
            print(f"  跳过 {d}: {type(e).__name__}: {e}", flush=True)
    print(f"\nDONE -> {dest}  {ok} 篇 / {total} chunk;collection={collection} dense_dim={cfg.dense_dim}",
          flush=True)
    return ok
