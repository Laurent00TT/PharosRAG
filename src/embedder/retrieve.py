"""检索期:query -> hybrid 召回(dense+BM25,RRF,ACL 硬过滤)-> dedup_by_section
-> chunker.assemble_big(small-to-big,ACL 感知)。chunker 走 lazy import(import 卫生)。"""
from __future__ import annotations

import json
import os

from .acl import acl_admits
from .config import SIDECAR_VERSION, EmbedConfig
from .dense import Dense
from .sparse import query_sparse
from .store import Store
from .types import User


class _ChunkShim:
    """从 Qdrant payload 重建 assemble_big 需要读的那几个 hit_chunk 属性(不必造完整 Chunk)。"""
    __slots__ = ("acl", "lang", "doc_id", "section_id", "section_anchor",
                 "breadcrumb", "page_start", "page_end", "source_indices", "doc_type")

    def __init__(self, p: dict):
        self.acl = p.get("acl") or {}
        self.lang = p.get("lang", "en")
        self.doc_id = p.get("doc_id", "")
        self.section_id = p.get("section_id")
        self.section_anchor = p.get("section_anchor")
        self.breadcrumb = p.get("breadcrumb") or []
        self.page_start = p.get("page_start", 0)
        self.page_end = p.get("page_end", 0)
        self.source_indices = p.get("source_indices") or []
        self.doc_type = p.get("doc_type")              # per-doc_type 预算用(seal#8)


class Retriever:
    def __init__(self, cfg: EmbedConfig | None = None, store: Store | None = None,
                 dense: Dense | None = None, reranker=None):
        # 同进程内与 Embedder 共享 store(嵌入式 Qdrant 单 client 约束)+ dense(避免重复 load 8B)。
        self.cfg = cfg or EmbedConfig()
        self.dense = dense or Dense(self.cfg)
        self.store = store or Store(self.cfg)
        self._reranker = reranker          # 可选 cross-encoder 精排(Reranker);首次开 rerank 时 lazy 建

    def _get_reranker(self):
        if self._reranker is None:
            from .rerank import Reranker   # lazy:只有真开 rerank 才 load 第二个 8B 模型(+16G 显存)
            self._reranker = Reranker(self.cfg)
        return self._reranker

    def _load_sidecar(self, doc_id: str, cache: dict | None = None, user: User | None = None):
        if cache is not None and doc_id in cache:           # 查询作用域缓存:同 doc 多命中只读盘+反序列化一次(review 性能)
            return cache[doc_id]
        path = os.path.join(self.cfg.sidecar_dir, f"{doc_id}.json")
        if not os.path.exists(path):                        # 明确错误而非裸 open(seal#10);调用方降级
            raise FileNotFoundError(f"sidecar 缺失:{path}(doc {doc_id} 入库但 sidecar 丢,需重新 index_document)")
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        # 版本校验:不符(含旧 sidecar 无 version 字段)即抛 ValueError —— 故意不在 search_with_context 的
        # (FileNotFoundError, JSONDecodeError) catch 范围内,向上传播响亮失败。缺文件/坏 JSON 是单 doc 瞬态
        # (降级),版本不符是系统性(schema 漂移,所有 sidecar 皆旧),应提示整体重建而非逐条静默降级。
        if d.get("version") != SIDECAR_VERSION:
            raise ValueError(
                f"sidecar 版本不符:{path} 是 v{d.get('version')},当前需 v{SIDECAR_VERSION}"
                f"(schema 变更);请对 doc {doc_id} 重新 index_document 以重建 sidecar。")
        acl_index = {int(k): v for k, v in d.get("acl_index", {}).items()}
        # Batch2.B doc 级 fail-closed 预检 —— 给 **doc_id 直读工具**(Batch3 get_document/expand)用:user 给定时,
        # 该 doc 至少要有一个元素对 user 可见,否则拒载(空 acl_index -> any([])=False -> 也拒,fail-closed)。
        # **user=None = hit 驱动调用方自证 doc_id 已过 ACL**(如 _assemble:hit 已过硬过滤+2.A 复核,下游还有 assemble_big
        # 逐元素门控 + 出口 acl_admits(ctx.acl)两道闸),此时**不预检** —— 因为 chunker 的 _stricter 会把命中块 idx 在
        # acl_index 改写成更严兄弟块 acl,doc 级预检会误伤本该交付的 public 命中(B2 对抗 review 实证 HIGH)。
        # 直读工具必须传 user;细粒度 per-element 门控仍由 get_document 的 admit= 负责。
        if user is not None and not any(acl_admits(a or {}, user) for a in acl_index.values()):
            raise PermissionError(f"doc {doc_id} 对当前身份无任何可见元素(fail-closed,Batch2.B)")
        from chunker.types import Element, Section          # lazy:只有真做 small-to-big 才需要 chunker
        elements = [Element(**e) for e in d["elements"]]
        # B3 review:doc 级直读(get_document/get_outline 的 _gather)与 assemble_big 都按**位置**索引 elements、但 acl_index
        # 键是 el.idx —— 稀疏/乱序 sidecar 会按位置错配 ACL、把无权元素当可见交付。assemble_big 内有此断言,但直读绕过它,
        # 故在加载源头统一断言(elements[i].idx==i),所有消费方一并 fail-closed(标准 adapter 密集有序,不触发)。
        if any(el.idx != i for i, el in enumerate(elements)):
            raise ValueError(f"sidecar elements 非密集有序(elements[i].idx==i),doc {doc_id} 取材会按位置错配 ACL;需重建 index_document")
        secs = {s["sec_id"]: Section(**s) for s in d["sections"]}
        result = (elements, secs, frozenset(d.get("banners", [])), acl_index)
        if cache is not None:
            cache[doc_id] = result
        return result

    def search(self, query: str, user: User, top_k: int | None = None, rerank: bool = False,
               doc_ids: list[str] | None = None, doc_type: str | None = None, kind: str | None = None,
               strategy: str = "hybrid", rerank_top_n: int | None = None):
        """召回(已过 ACL 硬过滤)。rerank=True:多召回 rerank_top_n -> cross-encoder 精排 -> 取 top_k。
        可选过滤(3.A/3.F):doc_ids/doc_type/kind 与 ACL AND。strategy(4.A):hybrid/dense/sparse 选路。
        rerank_top_n(4.B):精排候选池深度(默认 cfg.rerank_top_n,带护栏);rerank 失败 -> 降级返回 hybrid 召回(不崩)。"""
        qd = self.dense.encode_query(query)
        qs = query_sparse(query, self.cfg.stopwords)         # query 端 values=1.0;无有效 token -> None
        # Batch1.C:`top_k or cfg.top_k` 是 0-falsy bug(top_k=0 静默变默认、负数切错子集)。显式 is None + clamp。
        k = self.cfg.top_k if top_k is None else int(top_k)
        k = max(1, min(k, self.cfg.prefetch_limit))          # clamp 到 [1, 召回上限],超召回数无意义
        flt = dict(doc_ids=doc_ids, doc_type=doc_type, kind=kind, strategy=strategy)
        if rerank:
            rtn = self.cfg.rerank_top_n if rerank_top_n is None else max(1, min(int(rerank_top_n), self.cfg.prefetch_limit))
            n = max(rtn, k)                     # 召回至少 k 个供精排,否则 top_k>候选池时反而少给(review#4)
            hits = self.store.hybrid_search(qd.tolist(), qs, user, top_k=n, **flt)
            try:
                return self._get_reranker().rerank(query, hits, top_k=k)
            except Exception:                   # 4.B:rerank 失败(OOM/模型缺失)-> 降级返回 hybrid 召回,不让精排拖垮基础检索
                return hits[:k]                 # 返回的 hit score_kind 仍为 hybrid 量纲 -> _build 据此标 rerank_degraded
        return self.store.hybrid_search(qd.tolist(), qs, user, top_k=k, **flt)

    def search_with_context(self, query: str, user: User, top_k: int | None = None,
                            rerank: bool = False, doc_ids: list[str] | None = None,
                            doc_type: str | None = None, kind: str | None = None,
                            assemble: bool = True, strategy: str = "hybrid",
                            rerank_top_n: int | None = None) -> list[dict]:
        """召回(可选 rerank 精排)+ 按 section 去重 + small-to-big 组装(ACL 感知:big.text 只含与命中块同 ACL 的原文)。
        每条返回 {hit, context, context_status}。context=None 表示降级返回裸 hit;context_status 说明原因
        (full_section/climbed_N=完整小节(可直接用);**section_window=token 受限的窗口/截断片(非完整小节,要更全可对 chunk_id 调 expand)**;
        asset_no_prose=资产页无散文 big-block(数据靠命中块 content_raw,非完整段);single_chunk_degraded=sidecar 丢损;
        single_chunk_acl=big-block 出口 ACL 未过;deduped=与前命中同段/同窗的重复 big-block 被折叠,只回本命中块;concise=未做 small-to-big)。
        assemble=False(3.E concise):跳过 small-to-big 只回裸命中块(省 token/不读盘),仍做 section 去重。
        doc_ids/doc_type/kind(3.A/3.F):过滤透传。"""
        out, seen, cache, seen_anchor = [], set(), {}, set()   # cache:查询作用域 sidecar 缓存;seen_anchor:big-block 去重
        for h in self.search(query, user, top_k, rerank=rerank, doc_ids=doc_ids, doc_type=doc_type,
                             kind=kind, strategy=strategy, rerank_top_n=rerank_top_n):
            sid = h.payload.get("section_id")
            # section_id=None 是"无小节"而非"同一小节",不能折叠(否则同 doc 无小节命中只剩首个,seal#6)。
            # 资产块(图表/表格)的数据在自身(chunk.content_raw),不在散文 big-block 里 —— assemble_big 的 _gather
            # 只取 el.text/caption、不含 asset_content,故图表/表的数被排除。若资产与散文兄弟同 section 被折叠,就彻底
            # 丢失("召回到却答不出表/图里的数",eval 实证)。资产块按 chunk_id 独立成键,不参与 section 去重。
            kind = h.payload.get("kind")
            key = (h.doc_id, sid) if (sid is not None and kind not in ("chart", "table")) else ("__chunk__", h.chunk_id)
            if key in seen:                                  # dedup_by_section:同小节多 chunk 命中只留首个
                continue
            seen.add(key)
            if not assemble:                                 # 3.E concise:跳过 small-to-big,只回裸命中块(省 token/不读盘)
                out.append({"hit": h, "context": None, "context_status": "concise"})
                continue
            status = "full_section"
            try:
                ctx = self._assemble(h, user, cache)
            except (FileNotFoundError, json.JSONDecodeError, ValueError):  # sidecar 丢/损/版本漂移/非密集 -> 降级裸 hit
                # B3 review:ValueError(版本漂移/idx 非密集)此前未捕获 -> 冒泡到 retrieve 工具,把含 sidecar 绝对路径的
                # 异常裸抛给 agent(info_leak)。改为与缺文件同样降级(裸 hit 已过 store 2.A 复核,安全),不泄路径不崩整查询。
                ctx, status = None, "single_chunk_degraded"
            # 出口二次校验(铁律5):big.acl 必须对 user 可见,否则不交付。acl=None 视为"未经访问校验"-> 拒绝
            # (fail-closed,review#4:assemble_big legacy 分支会返回 acl=None 且同时跨 ACL 取材,不能放行)
            if ctx is not None and (ctx.acl is None or not acl_admits(ctx.acl, user)):
                ctx, status = None, "single_chunk_acl"
            # 多命中去重:同 doc 同 anchor 的 big-block 是 climb 到同父产出的重复,只留首个,其余降级裸 hit
            # —— 省 LLM context、不重复喂同段(lazy-tree review:多命中重叠)。
            # R2#5:窗口块(windowed)不同种子产 anchor 不等、按 anchor 精确判重会漏折叠 -> 同一 bound 内的窗口
            # resolved_section 相同(secid#window),改按 resolved_section 折叠,合并高度重叠的近重复窗口。
            if ctx is not None and ctx.anchor is not None:
                akey = ((h.doc_id, ctx.resolved_section) if getattr(ctx, "windowed", False)
                        else (h.doc_id, tuple(ctx.anchor)))
                if akey in seen_anchor:
                    ctx, status = None, "deduped"
                else:
                    seen_anchor.add(akey)
            # R2#1/#4:状态须区分"完整小节"与"窗口/截断"。windowed=True 是 _window_within 的 token 受限片(非完整段);
            # text 全空(资产页无散文,数据在 content_raw)单标 asset_no_prose。否则 climbed_N / full_section 才表完整小节。
            if ctx is not None:
                if not (ctx.text or "").strip():
                    status = "asset_no_prose"
                elif getattr(ctx, "windowed", False):
                    status = "section_window"
                elif getattr(ctx, "climbed", 0):
                    status = f"climbed_{ctx.climbed}"
                else:
                    status = "full_section"
            out.append({"hit": h, "context": ctx, "context_status": status})
        return out

    def _assemble(self, hit, user: User, cache: dict | None = None):
        from chunker import assemble_big                     # lazy
        from chunker.core import BUDGETS, DEFAULT_BUDGET
        # hit 驱动:hit 已过 store 硬过滤+2.A 复核,下游有 assemble_big 逐元素门控 + 出口 acl_admits(ctx.acl)两道闸,
        # 故**不传 user**触发 doc 级预检(那是给 Batch3 doc_id 直读工具的;在 hit 路径会被 _stricter 误伤,B2 review HIGH)。
        elements, secs, banners, acl_index = self._load_sidecar(hit.doc_id, cache)
        shim = _ChunkShim(hit.payload)
        # 只取与命中块**同 ACL** 的元素(INTEGRATION 默认 acl_index 路径):big.acl=hit_acl 自然准确、出口可校验,
        # 比 admit(取一切可见)更 fail-closed,且规避 chunker admit 分支的 big.acl 低报(seal#2)。
        mn, tg, mx = BUDGETS.get(shim.doc_type, DEFAULT_BUDGET)   # per-doc_type 预算:slides/policy 整节保留(seal#8)
        return assemble_big(shim, secs, elements, target=tg, min_tokens=mn, max_tokens=mx,
                            banners=banners, acl_index=acl_index)

    # ---- Batch 3 控制面:doc_id 直读 / 深挖 / 大纲 / 跨文档 ----
    def _visible_own_sections(self, secs: dict, acl_index: dict, user: User) -> list:
        """own-body(本节范围**减去子节范围**)有 ≥1 可见元素的 section 列表。供 get_outline 列举 / get_document 标题门控。
        B3 review:按整段 range 判'任一可见'会因父节吞入可见子孙而泄漏受限父标题,故只看本节直属 body。"""
        sec_list = list(secs.values())
        vis = []
        for sec in sec_list:
            child = [(c.start_idx, c.end_idx) for c in sec_list if c.parent_sec_id == sec.sec_id]
            if any((not any(cs <= i < ce for cs, ce in child)) and acl_admits(acl_index.get(i) or {}, user)
                   for i in range(sec.start_idx, sec.end_idx)):
                vis.append(sec)
        return vis

    def get_document(self, doc_id: str, user: User, max_tokens: int = 6000) -> dict:
        """通读一篇文档(3.B):doc 级 fail-closed 预检 + 逐 element acl_admits 门控(归一谓词,非 ==hit_acl)。
        标题元素不进 acl_index(被吸入 section 树),故对 **own-section 可见** 的小节额外纳入其标题(B3 review:否则通读丢全部标题)。
        lang 取 store 权威值而非粗猜(B3 review:粗猜误判 _cap 量纲会过度截断)。超 max_tokens 截断并标 truncated。"""
        from chunker.core import est_tokens
        from chunker.retrieve import _cap, _gather
        elements, secs, banners, acl_index = self._load_sidecar(doc_id, user=user)   # doc_id 驱动:预检 + idx==i 断言
        heading_idxs = {sec.start_idx for sec in self._visible_own_sections(secs, acl_index, user)}  # 可见小节的标题元素位

        def admits(i):
            a = acl_index.get(i)
            if a is not None:
                return acl_admits(a, user)
            return i in heading_idxs                          # 标题:own-section 可见即放行;其余未知 idx fail-closed 拒

        n_vis = sum(1 for i in range(len(elements)) if admits(i))
        text = _gather(elements, 0, len(elements), banners, admits)
        lang = self.store.get_doc_lang(doc_id)
        capped = _cap(text, lang, max_tokens)
        return {"doc_id": doc_id, "text": capped, "n_tokens": round(est_tokens(capped, lang)),
                "n_elements_visible": n_vis, "truncated": len(capped) < len(text)}   # 不回含无权计数的 total(B3 review info_leak)

    def get_outline(self, doc_id: str, user: User) -> list[dict]:
        """文档小节大纲(3.D),ACL 作用域:仅含 **own-body 有可见元素** 的小节(B3 review:防受限父标题经可见子节泄漏)。
        不回 breadcrumb(避免把受限祖先标题带出;层级用 level 表达)。"""
        elements, secs, banners, acl_index = self._load_sidecar(doc_id, user=user)   # doc_id 驱动:预检
        out = [{"sec_id": s.sec_id, "title": s.title, "level": s.level,
                "start_idx": s.start_idx, "end_idx": s.end_idx}
               for s in self._visible_own_sections(secs, acl_index, user)]
        out.sort(key=lambda s: s["start_idx"])
        return out

    def expand(self, chunk_id: str, user: User, target_tokens: int = 1500):
        """围绕某命中块取更大上下文(3.C):chunk_id -> point payload(ACL 复核)-> 更大预算重跑 assemble_big。
        **hit 驱动**(chunk_id=具体命中):同 _assemble,_load_sidecar **不传 user**(避免 _stricter false-deny,B2 教训);
        安全靠 get_by_chunk_id 的 acl_admits + assemble_big 的 ==hit_acl 逐元素门控 + 出口 acl_admits 三道闸。
        返回 BigBlock 或 None(找不到/无权/出口未过)。"""
        from chunker import assemble_big
        payload = self.store.get_by_chunk_id(chunk_id, user)   # ACL-gated;None=找不到/无权(fail-closed,不区分)
        if payload is None:
            return None
        shim = _ChunkShim(payload)
        elements, secs, banners, acl_index = self._load_sidecar(payload.get("doc_id", ""))   # hit 驱动:不传 user
        big = assemble_big(shim, secs, elements, target=target_tokens, min_tokens=max(target_tokens // 4, 1),
                           max_tokens=max(target_tokens * 2, 1500), banners=banners, acl_index=acl_index)
        if big.acl is None or not acl_admits(big.acl, user):   # 出口闸(铁律5)
            return None
        return big

    def search_grouped(self, query: str, user: User, doc_ids: list[str], top_k: int = 3,
                       rerank: bool = False) -> dict:
        """跨文档分组检索(3.G):对每个 doc_id 各取 top_k,返回 {doc_id: [Hit]}。对比/汇总类任务用。
        B3 review:query 只编码一次复用(避免 len(doc_ids) 次 8B 前向放大成本)。"""
        qd = self.dense.encode_query(query).tolist()
        qs = query_sparse(query, self.cfg.stopwords)
        k = max(1, min(int(top_k), self.cfg.prefetch_limit))
        # R2#7:rerank 候选池深度须与 search() 同口径 clamp 到 [1,prefetch_limit](否则 cfg.rerank_top_n 过大时超候选并集)
        rtn = max(1, min(int(self.cfg.rerank_top_n), self.cfg.prefetch_limit))
        out = {}
        for d in doc_ids:
            if rerank:
                hits = self.store.hybrid_search(qd, qs, user, top_k=max(rtn, k), doc_ids=[d])
                try:
                    out[d] = self._get_reranker().rerank(query, hits, top_k=k)
                except Exception:                        # R4.E:与 search() 同款优雅降级 —— reranker OOM/缺失退回 hybrid 召回,
                    out[d] = hits[:k]                     # 不让一个 doc 的精排失败崩掉整组分组检索(降级 hit 仍是 hybrid 量纲)
            else:
                out[d] = self.store.hybrid_search(qd, qs, user, top_k=k, doc_ids=[d])
        return out
