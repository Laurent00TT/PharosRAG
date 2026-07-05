"""Unit tests on the bundled fixture (no external data). Run: pytest -q"""
import json
import sys
from pathlib import Path


from chunker import Chunker
from chunker.adapters.mineru import from_mineru

FIX = Path(__file__).resolve().parent / "fixtures"


def _elements():
    content = json.loads((FIX / "sample_content_list.json").read_text(encoding="utf-8"))
    layout = json.loads((FIX / "sample_layout.json").read_text(encoding="utf-8"))
    return from_mineru(content, layout)


def _result():
    return Chunker().chunk(_elements(), doc_id="demo", doc_type="academic_paper", lang="en")


def test_noise_filtered():
    # header/footer never become chunks
    res = _result()
    texts = " ".join(c.text for c in res.chunks)
    assert "RUNNING HEADER" not in texts and "footer noise" not in texts


def test_numbering_correction_levels():
    # the core long-bar: '2' is level 1, '2.1'/'2.2' are level 2 (parser flattens; we split)
    res = _result()
    by_title = {s.title: s for s in res.sections}
    assert by_title["2 Methods"].level == 1
    assert by_title["2.1 Data"].level == 2
    assert by_title["2.2 Model"].level == 2
    # and 2.1 is nested under 2 Methods
    assert by_title["2.1 Data"].parent_sec_id == by_title["2 Methods"].sec_id


def test_asset_chunk_and_breadcrumb():
    res = _result()
    tables = [c for c in res.chunks if c.kind == "table"]
    assert len(tables) == 1
    t = tables[0]
    assert "Demo results" in t.text                  # caption is the retrieval text
    assert t.content_raw and "<table" in t.content_raw   # html kept for generation
    assert t.section_id is not None                  # has a section anchor


def test_image_path_passthrough():
    # img_path travels parser -> Element -> Chunk, but ONLY image/chart chunks expose it (we VL-embed
    # pictures only). table -> HTML in content_raw, equation -> LaTeX text, so they carry image_path=None.
    content = [
        {"type": "image", "img_path": "images/fig1.jpg", "image_caption": ["Figure 1: results"],
         "page_idx": 0},
        {"type": "table", "img_path": "images/tab1.jpg", "table_caption": ["Table 1"],
         "table_body": "<table><tr><td>x</td></tr></table>", "page_idx": 0},
        {"type": "text", "text": "A body paragraph long enough to become its own chunk here.",
         "page_idx": 0},
    ]
    els = from_mineru(content, None)
    assert els[0].image_path == "images/fig1.jpg"     # adapter extracts img_path...
    assert els[1].image_path == "images/tab1.jpg"     # ...verbatim for tables too (faithful Element layer)
    assert els[2].image_path is None
    res = Chunker().chunk(els, doc_id="d", doc_type="academic_paper", lang="en")
    img = [c for c in res.chunks if c.kind == "image"]
    assert img and img[0].image_path == "images/fig1.jpg"   # image chunk -> path exposed for VL embed
    tbl = [c for c in res.chunks if c.kind == "table"]
    assert tbl and tbl[0].image_path is None                # table -> HTML payload, no image ref (只做图片)
    txt = [c for c in res.chunks if c.kind == "text"]
    assert txt and all(c.image_path is None for c in txt)


def test_pure_image_survives_for_vl_embedding():
    # ① a picture with ONLY img_path (no caption/footnote/VLM content) must STILL produce a chunk —
    # VL embeds the raw image; dropping it loses the one thing only VL can recall. Flag it image_only.
    content = [
        {"type": "image", "img_path": "images/nocap.jpg", "page_idx": 0},
        {"type": "text", "text": "A surrounding body paragraph with enough words to be a chunk.",
         "page_idx": 0},
    ]
    res = Chunker().chunk(from_mineru(content, None), doc_id="d", doc_type="academic_paper", lang="en")
    imgs = [c for c in res.chunks if c.kind == "image"]
    assert len(imgs) == 1                                 # NOT dropped (was the 30% bug)
    assert imgs[0].image_path == "images/nocap.jpg"       # path retained -> VL can embed the picture
    assert "image_only" in imgs[0].flags                  # downstream: image vector, skip the sparse path
    assert imgs[0].n_tokens == 0                          # text is a placeholder, no real tokens


def test_empty_img_path_normalized_and_true_empty_dropped():
    # ③ '' img_path normalizes to None (one canonical "no image" value); and a true-empty asset
    # (no img, no text, no body) is still dropped — the image_path exception must not resurrect noise.
    captioned = [{"type": "image", "img_path": "", "image_caption": ["Fig 2: a chart"], "page_idx": 0}]
    res = Chunker().chunk(from_mineru(captioned, None), doc_id="d", doc_type="academic_paper", lang="en")
    imgs = [c for c in res.chunks if c.kind == "image"]
    assert imgs and imgs[0].image_path is None            # '' -> None, not '' (no ambiguous empty string)
    empty = [{"type": "image", "img_path": "", "page_idx": 0},
             {"type": "text", "text": "body paragraph with enough words to be its own chunk here.",
              "page_idx": 0}]
    res2 = Chunker().chunk(from_mineru(empty, None), doc_id="d", doc_type="academic_paper", lang="en")
    assert not [c for c in res2.chunks if c.kind == "image"]   # no img + no text + no body -> still dropped


def test_image_path_sanitized():
    # seal review #2: a poisoned img_path (traversal / absolute / UNC / url) must NOT reach the chunk —
    # image_path is the embed stage's only file-read entry (arbitrary-file-read / SSRF otherwise).
    for bad in ["../../etc/passwd", "/etc/passwd", "C:/secrets.txt", "http://evil/x.jpg", "\\\\srv\\x.jpg"]:
        content = [{"type": "image", "img_path": bad, "image_caption": ["fig"], "page_idx": 0}]
        res = Chunker().chunk(from_mineru(content, None), doc_id="t", lang="en")
        img = next(c for c in res.chunks if c.kind == "image")
        assert img.image_path is None, f"unsafe path leaked into chunk: {bad!r}"
    # a normal relative path under the output root survives
    content = [{"type": "image", "img_path": "images/ok.jpg", "image_caption": ["fig"], "page_idx": 0}]
    res = Chunker().chunk(from_mineru(content, None), doc_id="t", lang="en")
    assert next(c for c in res.chunks if c.kind == "image").image_path == "images/ok.jpg"


def test_same_name_sibling_sections_not_merged():
    # seal review (high): repeated subheadings (forms / appendices / multi-party filings, ~15% of docs)
    # share an identical breadcrumb. Their distinct bodies must NOT merge into one chunk (which would put
    # two semantically different sections into one embedding + mis-anchor to the first section).
    content = [
        {"type": "text", "text": "Applicant", "text_level": 1, "page_idx": 0},
        {"type": "text", "text": "Details", "text_level": 2, "page_idx": 0},
        {"type": "text", "text": "first applicant body alpha with enough words to be its own chunk",
         "page_idx": 0},
        {"type": "text", "text": "Applicant", "text_level": 1, "page_idx": 0},      # same-name sibling
        {"type": "text", "text": "Details", "text_level": 2, "page_idx": 0},
        {"type": "text", "text": "second applicant body bravo with enough words to be its own chunk",
         "page_idx": 0},
    ]
    res = Chunker().chunk(from_mineru(content, None), doc_id="t", doc_type="form", lang="en")
    texts = [c.text for c in res.chunks if c.kind == "text"]
    assert not any("alpha" in t and "bravo" in t for t in texts), "sibling bodies merged into one chunk"
    ca = next(c for c in res.chunks if "alpha" in c.text)
    cb = next(c for c in res.chunks if "bravo" in c.text)
    assert ca.section_id != cb.section_id, "two sibling sections mis-anchored to the same section_id"


def test_every_text_chunk_has_section_anchor():
    res = _result()
    text_chunks = [c for c in res.chunks if c.kind == "text"]
    assert text_chunks
    assert all(c.section_anchor is not None for c in text_chunks)


def test_small_to_big_grows():
    res = _result()
    els = _elements()
    small = min((c for c in res.chunks if c.kind == "text"), key=lambda c: c.n_tokens)
    big = Chunker().assemble_big(small, res, els, target=120, min_tokens=40, max_tokens=400)
    assert big.n_tokens >= small.n_tokens            # never smaller than the hit
    assert big.text                                   # produced something


def test_assemble_big_requires_dense_idx():
    # lazy-tree review#1: elements 必须 idx 密集有序,否则取材/ACL 门控按位置索引会取错段+泄漏 -> fail-closed raise
    import pytest
    res, els = _result(), _elements()
    hit = next(c for c in res.chunks if c.kind == "text")
    els[-1].idx = 999                                # 破坏密集(idx != 位置)
    with pytest.raises(ValueError):
        Chunker().assemble_big(hit, res, els)


def test_assemble_big_caps_at_max_tokens():
    # lazy-tree review: max_tokens 是真上限,big-block 超了截断(防 LLM context 溢出)
    from chunker.core import est_tokens
    res, els = _result(), _elements()
    hit = next(c for c in res.chunks if c.kind == "text")
    uncapped = Chunker().assemble_big(hit, res, els, target=99999, min_tokens=5, max_tokens=99999)
    capped = Chunker().assemble_big(hit, res, els, target=99999, min_tokens=5, max_tokens=20)
    assert capped.n_tokens < uncapped.n_tokens       # 封顶确实截短了(fixture 整块 ~37 token)
    assert est_tokens(capped.text, "en") <= 50       # 接近 max=20(留一行容差)


def test_assemble_big_cyclic_parent_no_hang():
    # lazy-tree review#2: 损坏 sidecar 的 cyclic parent_sec_id 不能让 climb 死循环
    res, els = _result(), _elements()
    secs = res.sections
    for s in secs:                                   # 全指向 secs[0]
        s.parent_sec_id = secs[0].sec_id
    secs[0].parent_sec_id = secs[-1].sec_id          # 闭环
    hit = next(c for c in res.chunks if c.kind == "text" and c.section_id)
    big = Chunker().assemble_big(hit, res, els, target=99999, min_tokens=10, max_tokens=400)
    assert big is not None and big.text              # 返回了 = 没死循环(seen 保护)


def test_source_indices_traceable():
    res = _result()
    for c in res.chunks:
        assert c.source_indices and all(isinstance(i, int) for i in c.source_indices)


# --- document metadata + access policy (stamped at chunk time) ---

def test_doc_meta_and_acl_stamped():
    res = Chunker().chunk(_elements(), doc_id="demo", doc_type="academic_paper", lang="en",
                          doc_meta={"title": "T", "source": "sec.gov"}, acl={"allow": ["grp:a"]})
    assert res.chunks
    for c in res.chunks:
        assert c.doc_meta.get("title") == "T" and c.doc_meta.get("source") == "sec.gov"
        assert c.acl.get("allow") == ["grp:a"]


def test_doc_type_stamped_and_lang_zh_alias():
    # seal review should-fix: Chunk now carries doc_type (assemble_big reads it for the query-time budget);
    # lang 'zh' aliases 'ch' so ISO codes don't silently undercount CJK tokens (~2.35x).
    from chunker.core import est_tokens
    assert est_tokens("中文文本测试一下", "zh") == est_tokens("中文文本测试一下", "ch")   # zh divisor == ch
    assert est_tokens("hello world", "en") != est_tokens("hello world", "zh")
    res = Chunker().chunk(_elements(), doc_id="d", doc_type="law", lang="en")
    assert res.chunks and all(c.doc_type == "law" for c in res.chunks)                    # doc_type stamped


def test_acl_fail_closed_by_default():
    # no acl passed -> RESTRICTED, never public
    c = Chunker().chunk(_elements(), doc_id="demo", doc_type="academic_paper").chunks[0]
    assert c.acl.get("unset") is True and c.acl.get("allow") == []


def test_acl_per_chunk_independent():
    res = Chunker().chunk(_elements(), doc_id="demo", doc_type="academic_paper",
                          acl={"allow": ["grp:a"]})
    res.chunks[0].acl["allow"].append("grp:secret")     # per-chunk (per-section) override
    assert res.chunks[1].acl["allow"] == ["grp:a"]       # must NOT leak to siblings


def test_extract_doc_meta_override_and_junk():
    from chunker.meta import extract_doc_meta
    m = extract_doc_meta("missing.docx", source="sec.gov", doc_type="report", author="")
    assert m["doc_id"] == "missing" and m["format"] == "docx"
    assert m["source"] == "sec.gov" and m["doc_type"] == "report"
    assert "author" not in m                              # empty override dropped


def test_extract_doc_meta_refuses_acl_keys():
    # a poisoned manifest must NOT smuggle access-semantic keys into doc_meta (payload-confusion).
    # acl is a SEPARATE channel (Chunker.chunk's acl param); doc_meta is convenience-only.
    from chunker.meta import extract_doc_meta
    m = extract_doc_meta("missing.pdf", source="erp", acl={"visibility": "public"},
                         tenant="other", allow=["*"], visibility="public")
    assert m["source"] == "erp"
    for k in ("acl", "tenant", "allow", "visibility"):
        assert k not in m                                # access keys stripped from doc_meta


def _two_acl_doc():
    # a parent section 'Compensation' with two child subsections: a PUBLIC 'Philosophy' (the hit)
    # and a TIGHTENED 'Executive Detail' holding a secret. small-to-big from the public hit must NOT
    # sweep the secret into big.text (review S3 / F2). Returns (result, elements, hit, SECRET).
    SECRET = "42000000-ceo-secret"
    content = [
        {"type": "text", "text": "Compensation", "text_level": 1, "page_idx": 0},
        {"type": "text", "text": "Philosophy", "text_level": 2, "page_idx": 0},
        {"type": "text", "text": "our pay philosophy is fairness and transparency for everyone here",
         "page_idx": 0},
        {"type": "text", "text": "Executive Detail", "text_level": 2, "page_idx": 0},
        {"type": "text", "text": f"CEO total compensation reached {SECRET} dollars last fiscal year",
         "page_idx": 0},
    ]
    els = from_mineru(content, None)
    res = Chunker().chunk(els, doc_id="comp", doc_type="report", lang="en",
                          acl={"visibility": "restricted", "allow": ["all-staff"]})
    hit = next(c for c in res.chunks if "philosophy" in c.text.lower())
    sib = next(c for c in res.chunks if SECRET in c.text)
    hit.acl = {"visibility": "public", "allow": ["all-staff"]}     # per-chunk: public
    sib.acl = {"visibility": "restricted", "allow": ["execs"]}     # per-chunk: tightened
    return res, els, hit, SECRET


def test_assemble_big_excludes_cross_acl_sibling():
    # the SAFE path (Chunker.assemble_big auto-passes acl_index): the tightened sibling's secret must
    # be excluded, the hit's own (public) text retained, and BigBlock.acl stamped == the hit's acl.
    res, els, hit, SECRET = _two_acl_doc()
    big = Chunker().assemble_big(hit, res, els, target=400, min_tokens=10, max_tokens=2000)
    assert SECRET not in big.text                         # tightened sibling NOT swept in
    assert "philosophy" in big.text.lower()               # the hit's own content survives
    assert big.acl == hit.acl                             # block carries an acl to re-verify


def test_assemble_big_legacy_path_is_unguarded():
    # documents the contract: the pure assemble_big WITHOUT acl_index does NOT gate (single-ACL-doc
    # assumption) and flags it via acl=None. This is exactly the leak the safe path closes — locking
    # it here means a future change that silently re-enables the unguarded default fails this test.
    from chunker import assemble_big
    res, els, hit, SECRET = _two_acl_doc()
    big = assemble_big(hit, res.sections_by_id(), els, target=400, min_tokens=10, max_tokens=2000)
    assert SECRET in big.text                             # unguarded -> would leak (proves the risk)
    assert big.acl is None                                # and is flagged as NOT access-verified


def test_assemble_big_admit_requires_acl_index():
    # seal review #3: admit judges EACH element's acl -> it REQUIRES acl_index. Without it, assemble_big
    # must RAISE, not silently judge every element by the hit's acl and stamp big.acl as verified (leak).
    import pytest
    from chunker import assemble_big
    res, els, hit, _ = _two_acl_doc()
    with pytest.raises(ValueError, match="acl_index"):
        assemble_big(hit, res.sections_by_id(), els, admit=lambda acl: True)


# --- review F1/F3 regressions: the failure modes the fixture could NOT expose ---

def _chunk(content):
    return Chunker().chunk(from_mineru(content, None), doc_id="t", doc_type="news", lang="ch")


def _pages(banner_or_label, with_colon):
    from chunker.adapters.mineru import from_mineru as _fm
    content = [{"type": "text", "text": "Doc Title", "text_level": 1, "page_idx": 0}]
    for p in range(8):
        content += [
            {"type": "text", "text": banner_or_label, "page_idx": p},
            {"type": "text", "text": f"section {p} heading", "text_level": 2, "page_idx": p},
            {"type": "text", "text": f"body paragraph {p} with enough words to make a real chunk here",
             "page_idx": p},
        ]
    return _fm(content, None)


def test_reset_numbering_not_over_promoted():
    # a weekly/list doc: bare numbers RESTART (1,2,1,2) => list ordinals, NOT chapters.
    # must stay at parser text_level (L2), not get promoted to L1 (review F1).
    content = [
        {"type": "text", "text": "Weekly Report", "text_level": 1, "page_idx": 0},
        {"type": "text", "text": "1. 海外AI:", "text_level": 2, "page_idx": 0},
        {"type": "text", "text": "- OpenAI launched a new model this week", "text_level": 2, "page_idx": 0},
        {"type": "text", "text": "more domestic ai news body text here for bulk", "page_idx": 0},
        {"type": "text", "text": "2. 国内AI:", "text_level": 2, "page_idx": 0},
        {"type": "text", "text": "some domestic progress narrative to fill the section", "page_idx": 0},
        {"type": "text", "text": "1. 影视:", "text_level": 2, "page_idx": 1},   # RESET -> proves list usage
        {"type": "text", "text": "2. 游戏:", "text_level": 2, "page_idx": 1},
    ]
    res = _chunk(content)
    l1 = [s for s in res.sections if s.level == 1]
    assert [s.title for s in l1] == ["Weekly Report"]          # only the real title is L1
    by_title = {s.title: s for s in res.sections}
    assert by_title["1. 海外AI:"].level == 2                    # list item NOT promoted to L1
    assert "- OpenAI launched a new model this week" not in by_title   # bullet glyph != heading
    body = " ".join(c.text for c in res.chunks)
    assert "OpenAI launched" in body                           # bullet text survives as body


def test_monotonic_numbering_still_promotes():
    # a deep-dive doc: bare numbers are a MONOTONIC outline (1,2,3) => real chapters.
    # must still promote to L1, with dotted subsection nested under it (no over-correction).
    content = [
        {"type": "text", "text": "Deep Dive Report", "text_level": 1, "page_idx": 0},
        {"type": "text", "text": "1. First pillar of growth", "text_level": 2, "page_idx": 0},
        {"type": "text", "text": "1.1 a sub aspect", "text_level": 2, "page_idx": 0},
        {"type": "text", "text": "narrative body for the sub aspect goes here", "page_idx": 0},
        {"type": "text", "text": "2. Second pillar of growth", "text_level": 2, "page_idx": 1},
        {"type": "text", "text": "3. Third pillar of growth", "text_level": 2, "page_idx": 1},
    ]
    res = _chunk(content)
    by_title = {s.title: s for s in res.sections}
    assert by_title["1. First pillar of growth"].level == 1    # monotonic -> promoted
    assert by_title["1.1 a sub aspect"].level == 2             # dotted refines
    assert by_title["1.1 a sub aspect"].parent_sec_id == by_title["1. First pillar of growth"].sec_id


# --- round-3 regressions: aside_text noise, colon-label false-positive, retrieve consistency ---

def test_aside_text_watermark_not_body():
    # page-edge watermark (aside_text) must NOT be recovered into body (round-3 CR-3)
    content = [
        {"type": "text", "text": "Title", "text_level": 1, "page_idx": 0},
        {"type": "aside_text", "text": "arXiv:1706.03762v7 [cs.CL] 2 Aug 2023", "page_idx": 0},
        {"type": "text", "text": "the real abstract body content goes here for the chunk", "page_idx": 0},
    ]
    body = " ".join(c.text for c in _chunk(content).chunks)
    assert "arXiv:1706.03762" not in body       # watermark dropped
    assert "real abstract body" in body         # real content kept


def test_colon_label_not_flagged_as_banner():
    # 'Prompt:' / 'GPT-4V:' recur on most pages but are CONTENT (speaker turns), not banners (round-3)
    from chunker.core import _banner_texts
    content = [{"type": "text", "text": "Multimodal Paper", "text_level": 1, "page_idx": 0}]
    for p in range(8):
        content += [
            {"type": "text", "text": "Prompt:", "page_idx": p},
            {"type": "text", "text": f"user question {p} about the receipt image shown above", "page_idx": p},
            {"type": "text", "text": "GPT-4V:", "page_idx": p},
            {"type": "text", "text": f"model answer {p} describing the tax amount on the receipt", "page_idx": p},
        ]
    els = from_mineru(content, None)
    assert _banner_texts(els) == set()           # colon labels NOT flagged
    body = " ".join(c.text for c in
                    Chunker().chunk(els, doc_id="t", doc_type="academic_paper", lang="en").chunks)
    assert "Prompt:" in body and "GPT-4V:" in body   # speaker labels preserved


def test_banner_filtered_chunk_and_retrieve():
    # a real running banner (>50% pages, no colon) is removed at BOTH chunk and retrieve time (round-3)
    from chunker.core import _banner_texts
    els = _pages("CONFIDENTIAL DRAFT REPORT", with_colon=False)
    assert "CONFIDENTIAL DRAFT REPORT" in _banner_texts(els)
    res = Chunker().chunk(els, doc_id="t", doc_type="news", lang="en")
    assert "CONFIDENTIAL DRAFT REPORT" not in " ".join(c.text for c in res.chunks)   # chunk-time
    hit = max(res.chunks, key=lambda c: c.n_tokens)
    big = Chunker().assemble_big(hit, res, els)
    assert "CONFIDENTIAL DRAFT REPORT" not in big.text                                # retrieve-time


def test_aside_text_absent_from_bigblock():
    # aside_text must be absent from retrieve-side big-blocks too (round-3 NOISE_KINDS consistency)
    content = [{"type": "text", "text": "Doc Title", "text_level": 1, "page_idx": 0}]
    for p in range(6):
        content += [
            {"type": "text", "text": f"section {p}", "text_level": 2, "page_idx": p},
            {"type": "aside_text", "text": f"WATERMARK-{p}", "page_idx": p},
            {"type": "text", "text": f"body paragraph {p} with several words to fill the section",
             "page_idx": p},
        ]
    els = from_mineru(content, None)
    res = Chunker().chunk(els, doc_id="t", doc_type="news", lang="en")
    assert all("WATERMARK" not in c.text for c in res.chunks)         # chunk-time
    hit = max(res.chunks, key=lambda c: c.n_tokens)
    big = Chunker().assemble_big(hit, res, els)
    assert "WATERMARK" not in big.text                                # retrieve-time


def test_table_retrieval_signal():
    # 表格块检索文本增强(Pharos TESTING §3 诊断):面包屑+表头+行标签进 text,content_raw 不变
    from chunker.core import _table_signal
    from chunker.types import Element
    body = ("<table><tr><td></td><td colspan=\"3\">As of December 31,</td></tr>"
            "<tr><td>2015</td><td>2014</td></tr>"
            "<tr><td>Revenues</td><td>$ 6,779,511</td><td>$ 5,504,656</td></tr>"
            "<tr><td>Net income</td><td>122,641</td><td>266,799</td></tr></table>")
    sig = _table_signal(body)
    assert "Revenues" in sig and "Net income" in sig and "As of December 31," in sig
    assert "6,779,511" not in sig                        # 数据单元格不进检索文本(无语义,徒增噪声)
    # 接到 chunk:caption + 面包屑 + 信号都在 text;表体仍在 content_raw
    els = [Element(idx=0, kind="header", text="x", page=1),
           Element(idx=1, kind="text", text="Segment overview.", text_level=1, page=1),
           Element(idx=2, kind="table", text="", caption="Selected Financial Data", table_body=body, page=1)]
    res = Chunker().chunk(els, doc_id="d", doc_type="financial_report_en", lang="en")
    tch = next(c for c in res.chunks if c.kind == "table")
    assert "Selected Financial Data" in tch.text and "Revenues" in tch.text
    assert "Segment overview" in tch.text                # 面包屑(所属小节)进检索文本 —— 范围/主题信号
    assert tch.content_raw == body                       # 表体原文不动(仍走 content_raw 供 ③ 补回)
    assert _table_signal(None) == "" and _table_signal("<table><td>甲") in ("甲", "")   # 缺失/畸形不崩


def test_placeholder_table_still_dropped():
    # 门控:无 caption 无表体的占位表格维持 F3 丢弃语义(面包屑不得单独救活幽灵块,否则 chunk id 平移)
    from chunker.types import Element
    els = [Element(idx=0, kind="text", text="Overview.", text_level=1, page=1),
           Element(idx=1, kind="table", text="", page=1)]          # 无 cap/foot/body
    res = Chunker().chunk(els, doc_id="d", doc_type="financial_report_en", lang="en")
    assert not [c for c in res.chunks if c.kind == "table"]
