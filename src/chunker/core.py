"""Pure chunking core (v2 Heading-Skeleton). Consumes Element[] -> ChunkResult.
No file IO. Ported from the harness chunk_document.py (tested on 77 docs / 5337 headings)."""
from __future__ import annotations

import re
from collections import Counter, defaultdict

from .types import Chunk, ChunkResult, Element, Section

NOISE_KINDS = {"header", "footer", "page_number", "aside_text"}   # round-3: aside_text = page-edge
# watermark / rotated spine (arXiv stamp, 券商书脊), never body — confirmed noise on 11/11 corpus cases
TOC_RE = re.compile(r"(\.{3,}|…)\s*\d+\s*$")
TOC_TAIL_RE = re.compile(r"^\d+(?:\.\d+)*\s+.+\s+\d{1,3}\s*$")
NUM_RE = re.compile(r"^\s*(\d+(?:\.\d+)*)")
YEAR_RE = re.compile(r"^(19|20)\d\d$")
BULLET_RE = re.compile(r"^\s*[-–—•·●○▪◦]\s")   # review F3: bullet glyph = body, never a heading
LAW_SEC_RE = re.compile(r"^\s*(SECTION|SEC\.)\s+\d+", re.I)
MAX_HEADING_LEN = 120

# (min, target, max) token budget per doc_type. Numbers cluster — one default works; the
# real differentiation is the layout MODE (slides/policy keep whole sections).
DEFAULT_BUDGET = (200, 800, 1500)
BUDGETS = {
    "financial_research_zh": (250, 550, 900), "academic_paper": (300, 750, 1100),
    "law": (300, 700, 1200), "financial_report_en": (250, 650, 1000),
    "government": (250, 650, 1000), "research_report": (200, 550, 800),
    "admin_industry": (250, 650, 1000), "tech_report": (300, 750, 1100),
    "form": (150, 400, 800), "brochure": (150, 400, 700), "guidebook": (150, 400, 700),
    "slides_tutorial": (0, 9999, 9999), "policy": (0, 9999, 9999), "news": (200, 550, 800),
}
PAGE_GROUPED = {"slides_tutorial"}        # one slide(page) = one chunk

# FAIL-CLOSED default access policy: a chunk whose document was NOT (yet) wired to a permission
# source is RESTRICTED, never public. Retrieval MUST hard pre-filter on chunk.acl; a chunk with
# `unset=True` / empty `allow` is denied to everyone but an explicit admin/open-mode caller.
RESTRICTED_ACL = {"visibility": "restricted", "allow": [], "unset": True}


def est_tokens(text, lang):
    # HEURISTIC char/divisor. 实测验证(2026-06,Qwen3-VL 真 tokenizer,2927 真实 chunk):英文散文
    # academic/law char/token≈3.85(≈现值 4.0,误差<4%);偏差几乎全来自数字/表格密集文档(财报 5.08/
    # 政府 5.35/中文研报 1.51),是 char-based 启发式的固有局限,单值无法兼顾"散文 vs 数字密集"两簇——
    # 改成均值(en 4.34)反害散文主体。两个误差方向都被兜住:高估 token→块偏小→small-to-big 补(不丢
    # 数据);低估→块偏大但远未及 Qwen3-VL 32k 上限。故保持现值不重标。
    # 'zh' 别名见 seal review:CJK 落 en 除数会 2.35x 低估 -> 截断风险。
    return len(text or "") / (1.7 if (lang or "").lower().startswith(("ch", "zh")) else 4.0)


def _norm(text):
    return re.sub(r"\s+", " ", (text or "").strip())


def _safe_rel(p):
    """SECURITY (seal review): an image_path must be a SAFE RELATIVE ref under the MinerU output root —
    reject absolute / UNC / drive paths, URL schemes (://), and '..' traversal, so a poisoned parse can't
    make the embed stage read an arbitrary file or SSRF off it. Returns the cleaned rel path, or None."""
    import posixpath
    if not isinstance(p, str) or not p.strip():
        return None
    q = p.strip().replace("\\", "/")
    if "://" in q or q.startswith("/") or re.match(r"^[A-Za-z]:", q):
        return None                                  # url scheme / absolute / UNC / windows drive letter
    norm = posixpath.normpath(q)
    if norm == ".." or norm.startswith("../") or norm.startswith("/"):
        return None                                  # traversal escaping the output root
    return norm


def _asset_desc(s, cap=800):
    """An image's VLM-extracted content (OCR text / mermaid / chart data) or alt-text, cleaned for
    use as retrieval text. For mermaid (flowchart/diagram), keep only the node + edge LABELS — the
    real retrieval signal — and drop scaffolding (`graph TD`, `-->`, `style X fill:#f9f`), which is
    zero-value noise in the embedding (review F2). Cap at a word boundary, not mid-token."""
    if not s:
        return ""
    s = re.sub(r"```[a-z]*|```", " ", s)
    if re.search(r"\bgraph\s+(?:TD|TB|LR|RL|BT)\b|\bflowchart\b|--?>", s):
        labels = re.findall(r'[\[\(\{|]\s*"?([^"\[\]\(\)\{\}|]+?)"?\s*[\]\)\}|]', s)
        s = " ".join(x.strip() for x in labels if x.strip())
    s = re.sub(r"\s+", " ", s).strip()
    return s[:cap].rsplit(" ", 1)[0] if len(s) > cap else s


_TR_RE = re.compile(r"<tr[^>]*>(.*?)</tr>", re.S | re.I)
_CELL_RE = re.compile(r"<t[dh][^>]*>(.*?)</t[dh]>", re.S | re.I)
_TAG_RE = re.compile(r"<[^>]+>")


def _table_signal(body, cap=400):
    """表格的检索信号:表头行(前 2 行,财报常见双层表头)+ 其余每行首个非空单元格(行标签),
    去重拼接、按词边界封顶。**为什么**:表格 chunk 的 text 原本只有 caption+footnote 一句——
    列头/行标签(Revenues / Net income / 年份…)才是"这张表回答什么问题"的语义信号,全锁在
    content_raw(不参与 embed/sparse)里,实测导致数字类问题表格块被散文挤出 top-k
    (Pharos TESTING §3 诊断)。只抽标签不抽数据单元格:数字本身无检索语义,徒增噪声。"""
    if not body:
        return ""
    rows = [[_TAG_RE.sub(" ", c).strip() for c in _CELL_RE.findall(r)] for r in _TR_RE.findall(body)]
    parts = []
    for r in rows[:2]:                                   # 表头(可能双层)
        parts.extend(r)
    for r in rows[2:]:                                   # 行标签 = 每行首个非空单元格
        first = next((c for c in r if c), "")
        if first:
            parts.append(first)
    seen, out = set(), []
    for p in parts:
        p = re.sub(r"\s+", " ", p).strip()
        if p and p not in seen:
            seen.add(p)
            out.append(p)
    sig = " | ".join(out)
    return sig[:cap].rsplit(" ", 1)[0] if len(sig) > cap else sig


def _banner_texts(elements):
    """Running banners/watermarks parser mislabeled as content: identical text appearing on
    >= half the pages (review round-2). Page-fraction (not raw count) avoids killing legit
    recurring headings ('SALARIES AND EXPENSES' x4, 'Engine Sensors' x6 are NOT banners).

    Round-3: a banner must NOT end with a colon — colon labels like 'Prompt:' / 'GPT-4V:'
    recur on 60% of pages in dialogue/multimodal papers but are CONTENT (speaker turns),
    not page furniture. (bbox position-stability would be a stronger signal — deferred.)"""
    npages = max((e.page for e in elements), default=0) + 1
    if npages < 4:
        return set()
    pages_of, freq = defaultdict(set), Counter()
    for e in elements:
        if e.kind in NOISE_KINDS:
            continue
        t = _norm(e.text)
        if t and not t.endswith((":", "：")):
            pages_of[t].add(e.page); freq[t] += 1
    return {t for t in freq if freq[t] >= 3 and len(pages_of[t]) >= npages * 0.5}


def _structural_nonheading(txt):
    """Text-shape rejects shared by the leveling pass and the reset-aware pre-pass."""
    return bool(TOC_RE.search(txt) or TOC_TAIL_RE.match(txt)
                or len(txt) > MAX_HEADING_LEN or BULLET_RE.match(txt))


def _leading_number(txt):
    """Parse a leading section/list number -> ('dotted', depth) | ('bare', value) | None.
    Year-leading (2026 ...) is treated as not-a-number (review F1/F4)."""
    m = NUM_RE.match(txt)
    if not m:
        return None
    tok = m.group(1)
    if YEAR_RE.match(tok.split(".")[0]):
        return None
    if "." in tok:
        return ("dotted", tok.count(".") + 1)
    if tok.isdigit():
        return ("bare", int(tok))
    return None


def heading_level(el: Element, doc_type, promote_bare=True):
    """text_level primary (free from parser). DOTTED numbering (2.1) always refines depth;
    a BARE integer promotes to L1 only when the doc's numbering is a monotonic outline
    (`promote_bare`) — otherwise it is a list ordinal and we defer to text_level (review F1)."""
    txt = (el.text or "").strip()
    if _structural_nonheading(txt):
        return None
    if doc_type == "law" and LAW_SEC_RE.match(txt):
        return 1
    if not el.text_level:
        return None
    num = _leading_number(txt)
    if num:
        kind, val = num
        if kind == "dotted":
            return val
        if kind == "bare" and promote_bare:
            return 1
    return int(el.text_level)


def build_sections(headings, doc_id, doc_end):
    """headings: [{idx, level, title, crumb}] in reading order -> section tree (idx ranges)."""
    sections = []
    for i, h in enumerate(headings):
        end = doc_end
        for j in range(i + 1, len(headings)):
            if headings[j]["level"] <= h["level"]:
                end = headings[j]["idx"]; break
        parent = next((headings[k]["idx"] for k in range(i - 1, -1, -1)
                       if headings[k]["level"] < h["level"]), None)
        sections.append(Section(
            sec_id=f"{doc_id}::s{h['idx']}", doc_id=doc_id, level=h["level"], title=h["title"],
            breadcrumb=h["crumb"], start_idx=h["idx"], end_idx=end,
            parent_sec_id=f"{doc_id}::s{parent}" if parent is not None else None))
    return sections


def _sentence_split(text, hi, lang):
    parts = re.split(r"(?<=[。！？.!?])\s+", text)
    out, cur = [], ""
    for p in parts:
        if est_tokens(cur + p, lang) > hi and cur:
            out.append(cur.strip()); cur = ""
        cur += p + " "
    if cur.strip():
        out.append(cur.strip())
    return out or [text]


def assemble_text(blocks, lang, budget):
    """Greedy accumulate consecutive same-section text blocks to a token budget. THE knob."""
    lo, target, hi = budget
    groups, cur, cur_tok = [], [], 0.0

    def flush():
        nonlocal cur, cur_tok
        if cur:
            groups.append(cur); cur = []; cur_tok = 0.0

    for b in blocks:
        bt = est_tokens(b["text"], lang)
        if bt > hi:
            flush()
            for piece in _sentence_split(b["text"], hi, lang):
                groups.append([{**b, "text": piece}])
            continue
        if cur and not b.get("merge") and cur_tok + bt > target:
            flush()
        cur.append(b); cur_tok += bt
    flush()

    merged = []
    for g in groups:
        gt = sum(est_tokens(x["text"], lang) for x in g)
        if merged and gt < lo and sum(est_tokens(x["text"], lang) for x in merged[-1]) + gt <= hi:
            merged[-1].extend(g)
        else:
            merged.append(g)

    return [{"text": "\n".join(x["text"] for x in g), "pages": sorted({x["page"] for x in g}),
             "idxs": [x["idx"] for x in g], "stitched": any(x.get("merge") for x in g)}
            for g in merged]


class Chunker:
    """The chunker component. `chunk(elements, ...)` -> ChunkResult. Stateless config holder."""

    def __init__(self, target=None, min_tokens=None, max_tokens=None, budgets=None, page_grouped=None):
        self._override = None
        if target is not None:
            self._override = (min_tokens or 200, target, max_tokens or 1500)
        self.budgets = {**BUDGETS, **(budgets or {})}
        self.page_grouped = page_grouped if page_grouped is not None else PAGE_GROUPED

    def _budget(self, doc_type):
        return self._override or self.budgets.get(doc_type, DEFAULT_BUDGET)

    def chunk(self, elements: list[Element], *, doc_id: str, doc_type: str | None = None,
              lang: str = "en", doc_meta: dict | None = None, acl: dict | None = None) -> ChunkResult:
        budget = self._budget(doc_type)
        banners = _banner_texts(elements)          # round-2: drop per-page running banners
        kept = [e for e in elements
                if e.kind not in NOISE_KINDS and _norm(e.text) not in banners]

        # reset-aware switch (review F1): bare-integer numbering only promotes to L1 when
        # the doc's bare numbers form a monotonic outline. A restart (1..9,1.. = list usage)
        # means the numbers are list ordinals, not chapters -> defer to text_level.
        bare_seq = []
        for el in kept:
            txt = (el.text or "").strip()
            if not el.text_level or _structural_nonheading(txt):
                continue
            num = _leading_number(txt)
            if num and num[0] == "bare":
                bare_seq.append(num[1])
        promote_bare = all(bare_seq[i] > bare_seq[i - 1] for i in range(1, len(bare_seq)))

        # heading stack -> breadcrumb + deepest-section per element; collect headings
        stack, headings, enriched = [], [], []
        for el in kept:
            lvl = heading_level(el, doc_type, promote_bare)
            if lvl is not None:
                while stack and stack[-1][0] >= lvl:
                    stack.pop()
                title = (el.text or "").strip()
                stack.append((lvl, title, el.idx))
                headings.append({"idx": el.idx, "level": lvl, "title": title,
                                 "crumb": [t for _, t, _ in stack]})
                continue
            el._crumb = [t for _, t, _ in stack]
            el._sec_head = stack[-1][2] if stack else None
            enriched.append(el)

        # emit chunks: assets atomic; text grouped by (section [, page])
        leaves, text_run, run_key = [], [], None

        def section_key(el):
            # group by the DEEPEST SECTION (its heading idx — unique per heading instance), NOT the
            # breadcrumb TEXT (seal review): two same-named sibling sections (repeated subheadings in
            # forms/appendices/multi-party filings) share an identical crumb, which would merge their
            # distinct bodies into one chunk and mis-anchor it to the first section — irreversibly wrong
            # data in the vector store. _sec_head (core:236) is unique per heading instance.
            base = getattr(el, "_sec_head", None)
            return (base, el.page) if doc_type in self.page_grouped else base

        def flush_text():
            nonlocal text_run, run_key
            if not text_run:
                return
            crumb = list(text_run[0].get("crumb") or [])     # breadcrumb from the run's elements, not the key
            for g in assemble_text(text_run, lang, budget):
                leaves.append(self._text_chunk(doc_id, lang, doc_type, g, crumb))
            text_run = []; run_key = None

        for el in enriched:
            if el.kind in ("table", "chart", "image"):
                flush_text()
                ch = self._asset_chunk(doc_id, lang, doc_type, el)
                if ch is not None:               # skip captionless + contentless placeholders (review F3)
                    leaves.append(ch)
            else:   # text/list + previously-dropped kinds: equation/code/page_footnote/ref_text
                txt = el.text or " ".join(el.list_items or [])
                if not txt.strip():
                    continue
                key = section_key(el)
                if run_key is not None and key != run_key:
                    flush_text()
                run_key = key
                text_run.append({"text": txt, "page": el.page, "idx": el.idx,
                                 "merge": el.merge_prev, "crumb": getattr(el, "_crumb", [])})
        flush_text()

        # section tree + stamp each leaf with its enclosing section anchor
        doc_end = (max((e.idx for e in elements), default=-1) + 1)
        sections = build_sections(headings, doc_id, doc_end)
        idx2head = {e.idx: getattr(e, "_sec_head", None) for e in enriched}
        sec_by_head = {s.start_idx: s for s in sections}
        # stamp doc-level metadata + access policy onto every chunk (the chunker STAMPS; ingest
        # EXTRACTS). acl defaults FAIL-CLOSED so an un-permissioned doc is never accidentally public.
        import copy
        dm = doc_meta or {}
        ac = acl if acl else RESTRICTED_ACL
        for n, ch in enumerate(leaves):
            ch.chunk_id = f"{doc_id}#{n:04d}"
            first = ch.source_indices[0] if ch.source_indices else None
            sec = sec_by_head.get(idx2head.get(first))
            ch.section_id = sec.sec_id if sec else None
            ch.section_anchor = [sec.start_idx, sec.end_idx] if sec else None
            ch.doc_meta = copy.deepcopy(dm)      # deep copy -> per-chunk (per-section) override is safe
            ch.acl = copy.deepcopy(ac)
            ch.doc_type = doc_type               # seal review: assemble_big reads this for query-time budget
        return ChunkResult(chunks=leaves, sections=sections, banners=frozenset(banners))

    def assemble_big(self, hit_chunk, result: ChunkResult, elements, target=None,
                     min_tokens=None, max_tokens=None, admit=None):
        from .retrieve import assemble_big as _ab
        b = self._budget(getattr(hit_chunk, "doc_type", None))
        return _ab(hit_chunk, result.sections_by_id(), elements,
                   target=target or b[1], min_tokens=min_tokens or b[0], max_tokens=max_tokens or b[2],
                   banners=result.banners,               # reuse the banner set computed at chunk time
                   acl_index=result.acl_index(), admit=admit)   # SECURITY: gather only same-ACL-as-hit text

    # -- chunk constructors --
    def _text_chunk(self, doc_id, lang, doc_type, g, crumb):
        return Chunk(
            chunk_id="", doc_id=doc_id, kind="text", text=g["text"], content_raw=None,
            breadcrumb=crumb, section_path=" > ".join(crumb), section_id=None, section_anchor=None,
            page_start=g["pages"][0], page_end=g["pages"][-1], source_indices=g["idxs"],
            n_tokens=round(est_tokens(g["text"], lang)), trust="high",
            flags=[f for f in ["multi_page" if len(g["pages"]) > 1 else None,
                               "merge_prev_stitched" if g.get("stitched") else None] if f],
            lang=lang)

    def _asset_chunk(self, doc_id, lang, doc_type, el: Element):
        cap, foot = (el.caption or "").strip(), (el.footnote or "").strip()
        body = el.table_body if el.kind == "table" else el.asset_content
        img = _safe_rel(el.image_path) if el.kind in ("image", "chart") else None   # sanitize (seal #2) + ''->None
        parts = [cap, foot]
        if el.kind in ("image", "chart"):        # fold the image's best description (VLM-extracted
            parts.append(_asset_desc(el.asset_content))   # content/OCR/mermaid, or alt-text) into the
        if el.kind == "table" and (cap or foot or el.table_body):   # 表格:面包屑(分部/子期间等范围信息常只在
            parts.append(" > ".join(getattr(el, "_crumb", [])))     # 小节标题里)+ 表头/行标签(表的语义信号,原本
            parts.append(_table_signal(el.table_body))              # 锁在 content_raw 不可检索;Pharos TESTING §3)。
            # ⚠ 门控 (cap|foot|body):无 caption 无表体的占位表格必须维持 F3 丢弃语义——若让面包屑单独把
            # retrieval 撑成非空,会"救活"幽灵块(text=面包屑、无任何内容),并使同文档后续 chunk id 平移
            # (旧索引 gold/引用全部错位)。实测无门控时全库 7652→7675(+23 幽灵块),门控后恢复 7652。
        retrieval = " ".join(x for x in parts if x).strip()   # retrieval text so the image is findable
        if not retrieval and not body and not img:    # ① a bare picture (img only) is STILL retrievable via
            return None                               # VL image向量化; only a true-empty placeholder is dropped
        flags = []
        if not cap:
            flags.append("captionless")
        if el.kind in ("chart", "image") and body:
            flags.append("vlm_content")
        if img and not retrieval and not body:   # ① pure picture: text is a placeholder -> downstream embeds
            flags.append("image_only")           # the IMAGE itself (skip the sparse/text path for this chunk)
        return Chunk(
            chunk_id="", doc_id=doc_id, kind=el.kind,
            text=retrieval or f"[{el.kind} without caption]", content_raw=body,
            breadcrumb=getattr(el, "_crumb", []), section_path=" > ".join(getattr(el, "_crumb", [])),
            section_id=None, section_anchor=None, page_start=el.page, page_end=el.page,
            source_indices=[el.idx], n_tokens=round(est_tokens(retrieval, lang)),
            trust="low" if "vlm_content" in flags else "high", flags=flags, lang=lang,
            image_path=img)                       # normalized; None for text/table (only图片做图像向量化)
