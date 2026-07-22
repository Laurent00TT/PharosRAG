"""扩语料:从 parsed/ 选一组表格密集 + 可成主题簇的文档,灌进更大的评估库 ~/rag_eval_big。
全部 tenant=demo/public(ACL 隔离另由 acl_regression.py 测);doc_id=目录名,doc_type/lang 按类型定。

选型:academic_paper(en,NLP 簇,结果表)/ financial_research_zh(ch,研报,表格密集)/ financial_report_en(en,财报)。
双语言 + 两类强表格内容 → 同时压"表格数值 grounding"瓶颈和"多跳/跨文档"评估。

跑:conda activate pharos && python eval/index_eval_corpus.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)   # 引擎包已 pip install -e . 安装,无需 sys.path 注入

from chunker import Chunker
from chunker.adapters.mineru import from_mineru_dir
from embedder import EmbedConfig, Embedder

PARSED = os.path.join(REPO, "parsed")
BIG = os.path.expanduser("~/rag_eval_big")
ACL = {"tenant": "demo", "allow": [], "visibility": "public", "unset": False}

# 类型 -> (lang, 取几篇)
PLAN = {
    "academic_paper": ("en", 5),
    "financial_research_zh": ("ch", 6),
    "financial_report_en": ("en", 4),
}


def select() -> list[tuple]:
    """返回 [(parsed_dir, doc_id, doc_type, lang, title)]。每类按名排序取前 N。"""
    out = []
    for dt, (lang, n) in PLAN.items():
        dirs = sorted(d for d in os.listdir(PARSED)
                      if d.startswith(dt + "__") and os.path.isdir(os.path.join(PARSED, d)))
        for d in dirs[:n]:
            out.append((d, d, dt, lang, d))   # doc_id=title=目录名(稳定、唯一)
    return out


def main():
    cfg = EmbedConfig(qdrant_path=os.path.join(BIG, "qdrant"),
                      sidecar_dir=os.path.join(BIG, "sidecar"), dense_dim=1024, collection="evalbig")
    emb = Embedder(cfg)
    docs = select()
    print(f"选中 {len(docs)} 篇:", flush=True)
    total = 0
    for d, doc_id, dt, lang, title in docs:
        ddir = os.path.join(PARSED, d)
        try:
            els = from_mineru_dir(ddir)
            res = Chunker().chunk(els, doc_id=doc_id, doc_type=dt, lang=lang,
                                  doc_meta={"title": title}, acl=ACL)
            stat = emb.index_document(doc_id, els, res, image_root=ddir)
            total += len(res.chunks)
            print(f"  {dt:22s} {len(res.chunks):4d} chunk  {doc_id[:46]}", flush=True)
        except Exception as e:
            print(f"  跳过 {doc_id}: {type(e).__name__}: {e}", flush=True)
    print(f"\nDONE -> {BIG}  共 {total} chunk / {len(docs)} 篇", flush=True)


if __name__ == "__main__":
    main()
