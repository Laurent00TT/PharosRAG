"""Generator:RAG 的 "G" —— 检索 → prompt 组装 → LLM → 解析引用。LLM 与 retriever 都依赖注入(解耦)。

ACL 出口:Generator **只消费 retriever.search_with_context 的返回**(hits 已过 Qdrant ACL 硬过滤,context 已过
出口二次校验)——不引入任何新内容,所以喂进 LLM 的每段都是 user 有权看到的。context=None(sidecar 丢/损)时降级
用命中块自身 text(单 chunk 同样已授权),不丢答案。
"""
from __future__ import annotations

import re

from chunker.core import est_tokens
from .llm import LLMClient
from .prompt import CITE_RE, PromptBuilder
from .types import Answer, Citation


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", s or "")


class Generator:
    def __init__(self, retriever, llm: LLMClient, prompt_builder: PromptBuilder | None = None, acl_check=None,
                 max_context_tokens: int | None = None):
        self.retriever = retriever
        self.llm = llm
        self.pb = prompt_builder or PromptBuilder()
        # 可选 (acl:dict, user)->bool 防御纵深(review#3):不传则信任 retriever 已 ACL 硬过滤;
        # 复用到未做硬过滤/直返裸 hit 的 retriever 时,注入它对每个命中块 acl 二次 fail-closed 校验。
        self.acl_check = acl_check
        # 闭管道 context 总量软预算(est-token,超预算整条截尾);None=不限(默认,行为不变)。
        # 与工具面 PHAROS_MAX_CONTEXT_TOKENS(toolcore,管检索交付)语义不同:这个管喂进 LLM 的 prompt
        # —— 换 8k/32k 小上下文后端时防超窗 400 / 后端静默截掉 SYSTEM。
        self.max_context_tokens = max_context_tokens

    def answer(self, query: str, user, top_k=None, rerank: bool = False,
               doc_ids=None, doc_type=None, kind=None, strategy=None, extra_legs=None,
               max_context_tokens: int | None = None) -> Answer:
        # N3(Pharos 评审后落地):可选检索过滤/选路透传给 retriever。**按需传**——不设的参数不出现在
        # 调用里,老的窄签名 retriever(单测 mock / smoke)与既有调用(eval)零影响。
        # 动机实证:"数字埋在表里"的题,通用问法下表格块被 MD&A 散文挤出 top-k,kind='table' 一击命中。
        kw: dict = {}
        if doc_ids is not None:
            kw["doc_ids"] = doc_ids
        if doc_type is not None:
            kw["doc_type"] = doc_type
        if kind is not None:
            kw["kind"] = kind
        if strategy is not None:
            kw["strategy"] = strategy
        results = list(self.retriever.search_with_context(query, user, top_k=top_k, rerank=rerank, **kw))
        # smart-ask 多路检索(并集,非替换——decompose 实验证明替换会窄化):每条 leg 一次带自己
        # 过滤/精排参数的检索,按 chunk_id 去重后**追加在主命中之后**(主路排序不受扰动)。
        # 典型用法:数值题补一条 DEFAULT_TABLE_LEG(signals.py),把被散文挤出主窗口的表格捞回来。
        if extra_legs:
            seen = {r["hit"].chunk_id for r in results}
            for leg in extra_legs:
                lkw = {k: leg[k] for k in ("doc_ids", "doc_type", "kind", "strategy", "rerank_top_n")
                       if leg.get(k) is not None}
                for r in self.retriever.search_with_context(query, user, top_k=leg.get("top_k"),
                                                            rerank=bool(leg.get("rerank")), **lkw):
                    cid = r["hit"].chunk_id
                    if cid not in seen:
                        seen.add(cid)
                        results.append(r)
        contexts, meta = [], []
        for r in results:
            hit, ctx = r["hit"], r["context"]
            payload = hit.payload or {}
            if self.acl_check is not None and not self.acl_check(payload.get("acl") or {}, user):
                continue                                               # 防御纵深:注入校验时,无权命中块不进 prompt
            text = (ctx.text if ctx is not None else hit.text) or ""   # big-block 优先;无则降级用命中块(均已授权)
            # ③ 表格/图表 grounding:assemble_big 的 big-block 只取 el.text/caption,不含资产的 asset_content
            #   -> 图表/表里的数(在 chunk.content_raw)被排除,LLM"召回到却答不出"。资产命中时把 content_raw 补回
            #   (eval 实证:正确性失败大量源于此)。资产块自身已过 ACL,补的是同一命中块的授权内容,不越权。
            if payload.get("kind") in ("chart", "table"):
                craw = (payload.get("content_raw") or "").strip()
                # R2#2:短资产数据(单元格/数字,如 "42")总是补回 —— 去空白子串匹配会误伤(如 "42" 恰在散文里被当重复抑制,
                # 重现 ③ 失败)。只对较长 content_raw 做"已在 text 里就不重复喂"的去重(避免重复喂长表 HTML)。
                if craw and (len(_norm(craw)) < 40 or _norm(craw) not in _norm(text)):
                    text = (text + "\n" + craw).strip() if text.strip() else craw
            if not text.strip():
                continue
            dm = payload.get("doc_meta") or {}
            # 范围证据进 prompt(与 SYSTEM 的数值范围约束配套):财报的"分部/子期间"信息往往只在小节标题里
            # (表格正文无一字提及)——不把 section_path 并进 source 行,模型无从知道数字是分部数据,
            # 实测会把分部营收错引申成公司总营收(Pharos N3 案例,根因=范围证据缺失,非模型不听话)。
            src = dm.get("title") or hit.doc_id
            sec = payload.get("section_path") or ""
            contexts.append({"text": text, "source": f"{src} § {sec}" if sec else src})
            meta.append({"chunk_id": hit.chunk_id, "doc_id": hit.doc_id, "title": dm.get("title") or hit.doc_id,
                         "section": payload.get("section_path") or "", "page": payload.get("page_start", 0),
                         "text": text})   # R3.D:title 缺省兜底 doc_id(与喂给 LLM 的 source 一致,溯源不落空串)
        # context 总量软预算:超预算按 contexts 顺序整条截尾(主命中在前、extra_legs 在后,天然优先级),
        # meta 同步截 —— 引用编号在 build 时才生成,截后天然对齐。首条永远保留:单条即超预算时清空会退化成
        # 零召回拒答,信息损失大于超预算(单条自身有 chunker BUDGETS 上界)。est_tokens 按英文除数近似
        # (中文会低估),软预算够用。
        budget = max_context_tokens if max_context_tokens is not None else self.max_context_tokens
        if budget and budget > 0 and contexts:
            total, n_keep = 0.0, 0
            for c in contexts:
                total += est_tokens(c["text"], "")
                if total > budget and n_keep >= 1:
                    break
                n_keep += 1
            contexts, meta = contexts[:n_keep], meta[:n_keep]
        messages = self.pb.build(query, contexts)
        if not contexts:   # R3.E:零召回 -> 确定性返回"信息不足",不把作答权交给 LLM(grounding 退路不再靠模型听话)
            return Answer(text="I don't have enough information in the provided context to answer.",
                          citations=[], n_contexts=0, finish_reason=None, raw_messages=messages)
        raw = self.llm.complete(messages)
        # finish_reason 在 complete 返回后**立即快照**进 Answer:llm 实例属性会被后续调用覆盖
        # (smart-ask 重试被弃用时实例上残留的是被丢弃那轮的值),事后读必错位。
        return Answer(text=raw, citations=self._parse_citations(raw, meta), n_contexts=len(contexts),
                      finish_reason=getattr(self.llm, "last_finish_reason", None), raw_messages=messages)

    @staticmethod
    def _parse_citations(answer_text: str, meta: list[dict]) -> list[Citation]:
        """从答案里抽 [n] 标记,映射回 context 来源(只保留实际引用且编号有效的,去重升序)。"""
        # 与 prompt._neutralize 共用 CITE_RE(宽松容空白/大小写):解析器认的变体中和器必拦(构造性对称),
        # 换非 DeepSeek 后端输出 "[cite: 1]" 之类漂移不再静默丢引用。只认 [cite:n],不抓正文裸 [n](review#1)。
        used = sorted({int(m) for m in CITE_RE.findall(answer_text)})
        out = []
        for n in used:
            if 1 <= n <= len(meta):                  # 越界的 [n](LLM 幻觉编号)丢弃,不映射错来源
                m = meta[n - 1]
                out.append(Citation(marker=n, chunk_id=m["chunk_id"], doc_id=m["doc_id"], title=m["title"],
                                    section=m["section"], page=m["page"], text=m["text"]))
        return out
