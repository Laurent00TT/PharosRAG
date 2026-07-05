"""7.C ACL 隔离回归(端到端、走真检索路径)。demo 库全 public 测不出隔离,故现建一个 2 租户合成库,
用真 Chunker+Embedder 灌入(真向量),再以不同身份跑 Retriever,断言【跨租户/无权/unset 身份对受限内容 0 召回】。

这是 fail-closed 的回归底线:把每篇的【独有 sentinel 原文】当 query,以无权身份去搜 —— 哪怕精确匹配也必须召回 0
(证明 ACL 硬过滤先于相关性,而非靠"搜不到"侥幸)。同时覆盖 Batch2/3 的 doc 级直读面(get_document/expand fail-closed)。

策略(对齐 acl_admits:跨租户永远拒;public 仅同租户内;allow∩principals 或 public 才可见):
  docA  t1 / restricted / allow=[g_research]   -> 仅 g_research(同租户)可见
  docB  t1 / public                            -> t1 任何身份可见
  docC  t2 / public                            -> 仅 t2 可见(对 t1 跨租户不可见)
  docD  t1 / restricted / allow=[g_finance]    -> 仅 g_finance 可见
用法:conda activate navikb && python eval/acl_regression.py   (退出码 0=全过,1=有泄漏)
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)   # 引擎包已 pip install -e . 安装,无需 sys.path 注入

from chunker import Chunker
from chunker.types import Element
from embedder import EmbedConfig, Embedder, Retriever, User

# 每篇:标题 + 独有 sentinel 正文(查询时拿 sentinel 精确命中,验证无权身份连精确匹配都召回不到)
DOCS = {
    "docA": ("t1", "restricted", ["g_research"],
             "Project Alpha Confidential",
             "Project Alpha reported secret quarterly revenue of forty two million dollars in the restricted research ledger."),
    "docB": ("t1", "public", [],
             "Bravo Public Roadmap",
             "Bravo public roadmap announces the next-year product launch open to everyone in tenant one."),
    "docC": ("t2", "public", [],
             "Charlie Tenant-Two Memo",
             "Charlie memo belongs to tenant two and discusses an unrelated logistics reorganization plan."),
    "docD": ("t1", "restricted", ["g_finance"],
             "Delta Finance Only",
             "Delta finance-only confidential gross margin data shows a sensitive eighty seven percent figure."),
}

# 身份 -> 应可见的 doc 集合(其余必须 0 召回)
USERS = {
    "u_research(t1,g_research)": (User(tenant="t1", principals=["g_research"]), {"docA", "docB"}),
    "u_finance(t1,g_finance)":   (User(tenant="t1", principals=["g_finance"]), {"docB", "docD"}),
    "u_t2(t2,-)":                (User(tenant="t2", principals=[]), {"docC"}),
    "u_noperm(t1,g_none)":       (User(tenant="t1", principals=["g_none"]), {"docB"}),
    "u_unset('',-)":             (User(tenant="", principals=[]), set()),
}


def build_doc(title: str, body: str) -> list[Element]:
    return [Element(idx=0, kind="text", text=title, text_level=1, page=1),
            Element(idx=1, kind="text", text=body, page=1)]


def main():
    work = os.path.join(tempfile.gettempdir(), "rag_acl_regression")
    if os.path.exists(work):
        shutil.rmtree(work, ignore_errors=True)
    cfg = EmbedConfig(qdrant_path=os.path.join(work, "qdrant"),
                      sidecar_dir=os.path.join(work, "sidecar"), dense_dim=1024, collection="aclreg")
    emb = Embedder(cfg)
    for doc_id, (tenant, vis, allow, title, body) in DOCS.items():
        acl = {"tenant": tenant, "allow": allow, "visibility": vis, "unset": False}
        els = build_doc(title, body)
        res = Chunker().chunk(els, doc_id=doc_id, doc_type="memo", lang="en",
                              doc_meta={"title": title}, acl=acl)
        emb.index_document(doc_id, els, res, image_root=None)
    retr = Retriever(cfg, store=emb.store, dense=emb.dense)

    fails = []

    def check(desc, ok):
        print(f"  {'PASS' if ok else 'FAIL'}  {desc}", flush=True)
        if not ok:
            fails.append(desc)

    # 1) 召回隔离矩阵:每个身份 × 每篇 sentinel 查询 -> 命中 doc 集合必须 ⊆ 授权集
    print("=== 1) search 召回隔离(sentinel 精确查询)===", flush=True)
    for uname, (user, allowed) in USERS.items():
        for doc_id, (_t, _v, _a, _title, body) in DOCS.items():
            hits = retr.search(body, user, top_k=10)
            got = {h.doc_id for h in hits}
            leaked = got - allowed
            check(f"{uname} 搜《{doc_id}》原文 -> 命中 {sorted(got) or '∅'};授权 {sorted(allowed) or '∅'}"
                  + (f"  ⚠泄漏 {sorted(leaked)}" if leaked else ""),
                  not leaked and (doc_id in got if doc_id in allowed else True))

    # 2) 授权正向:有权身份必须能搜到自己该看的(防"全拒"假过)
    print("\n=== 2) 授权正向(防全拒假过)===", flush=True)
    for uname, (user, allowed) in USERS.items():
        for doc_id in allowed:
            body = DOCS[doc_id][4]
            got = {h.doc_id for h in retr.search(body, user, top_k=10)}
            check(f"{uname} 应能召回授权的《{doc_id}》", doc_id in got)

    # 3) doc 级直读 fail-closed:无权身份 get_document 受限篇 -> PermissionError
    print("\n=== 3) get_document 直读 fail-closed ===", flush=True)
    for uname, (user, allowed) in USERS.items():
        for doc_id in DOCS:
            if doc_id in allowed:
                continue
            try:
                retr.get_document(doc_id, user)
                check(f"{uname} get_document《{doc_id}》(无权)应拒", False)
            except PermissionError:
                check(f"{uname} get_document《{doc_id}》(无权)-> PermissionError", True)

    # 4) expand 跨 ACL fail-closed:用 docA 的真实 chunk_id,以无权身份 expand -> None
    print("\n=== 4) expand 跨 ACL fail-closed ===", flush=True)
    owner = USERS["u_research(t1,g_research)"][0]
    a_hits = retr.search(DOCS["docA"][4], owner, top_k=5)
    a_cid = next((h.chunk_id for h in a_hits if h.doc_id == "docA"), None)
    check("能取到 docA 的 chunk_id(前置)", a_cid is not None)
    if a_cid:
        for uname in ("u_finance(t1,g_finance)", "u_t2(t2,-)", "u_noperm(t1,g_none)", "u_unset('',-)"):
            user = USERS[uname][0]
            check(f"{uname} expand(docA.chunk)(无权)-> None", retr.expand(a_cid, user) is None)

    # 5) RRF fusion prefetch 下推隔离(R5.M2):废掉出口 acl_admits 复核(打成恒 True),只留 acl_filter 的 prefetch-level
    #    下推。若跨租户/无权仍 0 召回,证明 fusion 下推**本身**就挡住(而非靠出口闸掩盖),锁死"嵌入式 QdrantLocal
    #    fusion 丢顶层 should" 那个坑的修复。英文 sentinel 有 sparse token -> 走 dense+sparse 两路 prefetch 的 RRF 融合。
    print("\n=== 5) RRF fusion prefetch 下推隔离(禁出口闸后仍须 0 泄漏)===", flush=True)
    import embedder.store as _store
    _orig_admits = _store.acl_admits
    _store.acl_admits = lambda acl, user: True          # 废出口复核
    try:
        for uname, (user, allowed) in USERS.items():
            for doc_id, (_t, _v, _a, _title, body) in DOCS.items():
                got = {h.doc_id for h in retr.search(body, user, top_k=10)}   # 默认 hybrid=RRF fusion
                leaked = got - allowed
                check(f"[禁出口闸] {uname} 搜《{doc_id}》-> 命中 {sorted(got) or '∅'}"
                      + (f"  ⚠fusion下推未挡,泄漏 {sorted(leaked)}" if leaked else ""), not leaked)
    finally:
        _store.acl_admits = _orig_admits                # 恢复,不影响后续

    print(f"\n{'='*48}\n{'全部通过 ✓' if not fails else f'有 {len(fails)} 项泄漏/失败 ✗'}", flush=True)
    shutil.rmtree(work, ignore_errors=True)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
