"""`pharos index`:把 MinerU 解析产物目录批量建成 Pharos 索引(qdrant + sidecar)。

产品化自引擎仓 mcp_server/index_real.py(那份保留作历史脚本):corpus/目标/collection/ACL
全部参数化,不再写死 parsed/ 与 ~/rag_real。语料要求:corpus 目录下每篇一个 MinerU 输出目录
(content_list.json + layout.json…),目录名即 doc_id,`<doc_type>__<name>` 前缀约定可选。

需 GPU(Qwen3-VL-Embedding-8B)。⚠ 嵌入式 Qdrant 单客户端:目标索引正被守护进程打开时
无法写入 —— 先停 `pharos serve` 再 index,或 index 到新目录后切换 PHAROS_INDEX_DIR。
"""
from __future__ import annotations

import os

from . import config
from .engine import bootstrap


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
    bootstrap(cfg.engine)
    from chunker import Chunker
    from chunker.adapters.mineru import from_mineru_dir
    from embedder import EmbedConfig, Embedder

    corpus = corpus or os.path.join(cfg.engine, "parsed")
    dest = os.path.expanduser(dest or cfg.index_dir)
    collection = collection or cfg.collection
    acl = {"tenant": (tenant or cfg.tenant or "demo"),
           "allow": [a.strip() for a in allow.split(",") if a.strip()],
           "visibility": visibility, "unset": False}
    if not os.path.isdir(corpus):
        raise SystemExit(f"语料目录不存在:{corpus}")

    ecfg = EmbedConfig(qdrant_path=os.path.join(dest, "qdrant"),
                       sidecar_dir=os.path.join(dest, "sidecar"),
                       dense_dim=cfg.dense_dim, collection=collection)
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
