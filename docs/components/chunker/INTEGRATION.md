# 集成指南

## 1. 接进 RAG indexing 管线

```
parse ──(adapter)──► Element[] ──► Chunker.chunk ──► Chunk[] ─► embed(chunk.text) ─► vector store
                                                  └► Section[] ─► section store(供检索)
```

**索引期**(含文档级 metadata + 访问策略):
```python
from chunker import Chunker, extract_doc_meta
from chunker.adapters.mineru import from_mineru

def index_document(doc_id, content_list, layout, doc_type, lang, src_path, acl):
    elements = from_mineru(content_list, layout)
    doc_meta = extract_doc_meta(src_path, doc_type=doc_type, lang=lang,   # 文件 core.xml/PDF info
                                source="...", domain="...")               #  + manifest/源系统覆盖
    # acl 来自源系统(SharePoint/Drive/S3 的权限、租户映射…),不从内容推断
    result = Chunker().chunk(elements, doc_id=doc_id, doc_type=doc_type, lang=lang,
                             doc_meta=doc_meta, acl=acl)
    for c in result.chunks:
        # 文本 chunk 嵌文本;image/chart（尤其带 `image_only` flag 的纯图）应改走「图像向量化」——
        # 用 VL embedder 嵌 image_path 指向的裁切图（相对 MinerU 输出根，需 join 成绝对路径），而非 embed(c.text)。
        # image_only chunk 的 text 只是占位符（n_tokens=0），稀疏路对它无效，只能靠稠密图像向量召回。
        vec = embed(c.text)
        vector_store.add(id=c.chunk_id, vector=vec, payload={
            "doc_id": c.doc_id, "kind": c.kind, "text": c.text,
            "breadcrumb": c.breadcrumb, "section_id": c.section_id,
            "section_anchor": c.section_anchor, "source_indices": c.source_indices,
            "content_raw": c.content_raw, "page_start": c.page_start, "flags": c.flags,
            "doc_meta": c.doc_meta, "acl": c.acl,         # ← 文档级 metadata + 访问策略
            "image_path": c.image_path,                   # ← image/chart 裁切图引用(供 VL 图像向量化;text/table=None)
            "lang": c.lang, "page_end": c.page_end,       # ← assemble_big 查询期必读(token 估算 / slide 开窗)
            "doc_type": c.doc_type,                       # ← assemble_big 据此选查询期 token 预算
        })
    section_store.put(doc_id, [vars(s) for s in result.sections])   # 节树
    element_store.put(doc_id, [vars(e) for e in elements])          # 原始元素(供取大块)
```

> 也可只存 `section_anchor` + 原始元素,查询期再算大块——本组件的 `assemble_big` 要的就是 `sections + elements`。

**查询期(small-to-big),权限是第一道硬过滤**:
```python
def retrieve(query, user, top_k=8):
    # ① ACL 硬预过滤(在向量库 filter 层,fail-closed)——用户拿不到无权 chunk
    acl_filter = (
        "acl.unset != true AND "                                  # 未接权限的文档默认拒绝
        "(acl.tenant == :tenant) AND "                            # 租户隔离
        "(acl.allow ANY IN :principals OR acl.visibility == 'public')"
    )
    hits = vector_store.search(embed(query), top_k, filter=acl_filter,
                               params={"tenant": user.tenant, "principals": user.groups + [user.id]})
    out = []
    for h in dedup_by_section(hits):
        chunk = rebuild_chunk(h.payload)                          # payload -> Chunk(含 doc_meta/acl)
        secs  = {s["sec_id"]: rebuild_section(s) for s in section_store.get(h.doc_id)}
        els   = [rebuild_element(e) for e in element_store.get(h.doc_id)]
        # ② small-to-big 必须 ACL 感知:big-block 按 idx 范围从原始 elements 重新取材,会跨 chunk 边界。
        #    若文档内对某小节单独收紧(铁律3),纯按 idx 取材会把无权兄弟小节的明文捞进 big.text。
        #    建一个 {idx: acl} 映射(本文档所有 chunk payload 的 source_indices->acl),传给 assemble_big:
        acl_index = {i: cp["acl"] for cp in chunk_store.get(h.doc_id)   # 该 doc 全部 chunk payload
                     for i in cp["source_indices"]}
        from chunker import assemble_big
        big = assemble_big(chunk, secs, els, acl_index=acl_index)  # 默认:只取与命中块同 ACL 的 element
        # 想"取调用者一切有权看到的(跨不同但可见的 ACL)",改传 admit= 复用硬过滤同一谓词:
        #   big = assemble_big(chunk, secs, els, acl_index=acl_index,
        #                      admit=lambda acl: acl_admits(acl, user))
        out.append({"title": chunk.doc_meta.get("title"), "source": chunk.doc_meta.get("source"),
                    "breadcrumb": big.breadcrumb, "context": big.text, "hit": chunk.text,
                    "context_acl": big.acl})                       # big.acl: 供出口处二次校验
    return out
```

## 6. 文档级 metadata 与访问控制(权限)

两者都是**文档属性**(对同一文档每个 chunk 一致),由 **ingest 提取**、chunker **盖到每个 chunk**、随 payload 进向量库。**chunker 只盖章,不提取/不判权限**(它 format-无关,不知道文件的标题/来源/权限)。

- **`doc_meta`(便利信息)**:`extract_doc_meta(path, **overrides)` 从 office `docProps/core.xml`(title/author/created/modified)、PDF `/Info` 抽取,合并 manifest/源系统的 `source/doc_type/domain/lang`(覆盖优先,去空/去 "admin" 等脏值)。用途:payload **过滤**("2024 年的英文财报")+ **引用溯源**(title/source/date)。可选只把 `title` 拼进 embed text 做消歧,**其余 payload-only**(全塞会稀释)。
- **`acl`(安全边界,语义完全不同)**:由**源系统**(文档库 ACL / 租户映射)在 ingest 给出,chunker 盖到 `chunk.acl`。**铁律:**
  1. **硬预过滤**:权限在向量库 **filter 层**强制(如上),不是 re-rank、更不是"让 LLM 别提"。无权 chunk **根本不被检索到**。
  2. **Fail-closed**:不传 `acl` → 默认 `RESTRICTED_ACL`(`unset=True`,空 `allow`)——未接权限的文档**默认拒绝所有人**,绝不意外公开。
  3. **每 chunk 独立可覆盖**:`acl` 深拷贝到每个 chunk,敏感小节可在 chunk 后单独收紧(`chunk.acl = stricter`),不影响同文档其他 chunk。
  4. **big-block 必须 ACL 感知(不是"同文档即安全")**:`assemble_big` 按 idx 范围从原始 elements 重新取材,会**跨 chunk 边界**。"同文档"≠"同 ACL"——若用了铁律3 的 per-chunk 收紧,纯按 idx 取材会把无权兄弟小节的明文捞进 `big.text`。**必须传 `acl_index`**({idx: acl},见上 retrieve 示例 / `ChunkResult.acl_index()`):默认只取与命中块**同 ACL** 的 element(fail-closed:未知 idx 排除),或传 `admit=` 用调用者可见性谓词。返回的 `BigBlock.acl` 带回有效 acl 供出口二次校验;**不传 acl_index 即 legacy 无防护模式(假设全文档单一 ACL),`BigBlock.acl=None` 显式标记"未经访问校验"**。便捷封装 `Chunker.assemble_big(hit, result, els)` 已自动从 `result` 建 acl_index,默认安全。
  5. **出口不变量**:任何**返回给用户的文本**(hit / big-block context / doc_meta)在出口处都要能映射回一个 acl 并对 caller 复核——不要信任"上游 filter 已经管住了"。small-to-big(big.text)与 `deny`(query filter)历史上都是绕过 filter 的旁路。

> ACL 的 schema 是自由 dict(`{visibility, allow:[principals], tenant, classification}`),按你的权限模型填;chunker 不解释字段,只忠实盖章 + fail-closed 默认。
> **`deny`(黑名单)不在上面的硬过滤示例里生效**——示例 filter 只读 `unset/tenant/allow/visibility`。要"排除某组",**从 `allow` 移除**即可;确需黑名单语义,必须自己在 filter 里加 `AND NOT (acl.deny ANY IN :principals)`,否则被 deny 的组只要还在 allow(或命中 public)照样检索到。**安全字段"写了不生效"是最毒的契约失败**,别假设 chunker 或示例 filter 替你执行了 deny。

## 2. 写一个新的 parser 适配器

parse 阶段换成别的(Docling / Unstructured / 自研),只需产出 `Element[]`:

```python
from chunker.types import Element

def from_yourparser(parsed) -> list[Element]:
    out = []
    for i, item in enumerate(parsed.blocks):
        out.append(Element(
            idx=i,
            kind=map_kind(item.type),          # -> text|table|image|chart|list|header|footer|page_number
            text=item.text,
            text_level=item.heading_level,      # parser 给的标题级别(没有就 None)
            page=item.page,
            caption=item.caption, table_body=item.html, asset_content=item.vlm_desc,
            merge_prev=item.continues_prev,     # 跨页续接(没有就 False)
        ))
    return out
```
**关键字段**:`kind`、`text`、`text_level`(定级主信号)、`caption`/`table_body`(资产)、`merge_prev`(跨页)。其余可缺省。核心与 `assemble_big` 全部复用,不用改。

**写新 adapter 前先看格式是否适配本模型**(详见 [`ARCHITECTURE.md` §格式与 parser 范围](ARCHITECTURE.md)):

- **PDF(MinerU)**✅ 已验证(77 篇)——`from_mineru` 即是;扫描件 PDF 走 OCR(40 篇 13 语言)。
- **Word(.docx)**✅ 已验证(~50 篇)——**推荐 MinerU 原生 office 后端**(`scripts/parse_office.py`→content_list→`from_mineru`);另有零依赖 fallback `adapters/docx.py`(段落样式→`text_level`,对 ~64% 无 heading style 文档用 bold/format 兜底)。页概念弱→ `page` 近似。
- **PPT(.pptx)**✅ 已验证——推荐 MinerU 原生 office 后端;另有 fallback `adapters/pptx.py`(幻灯片=页,标题→heading、要点→`text`、图→`image`)。
- **Excel(.xlsx)**✅ 已适配——走**独立路径** `table_chunker.py`(TableChunker,方案 A,51 篇验证、单元格 100% 覆盖):按 sheet/行组切 + 列头作上下文;heading-tree 不适用,但**输出同一 Chunk schema**,能被同一 embedder/retriever 消费。见 [`../../methodology/MULTIFORMAT_IMPL.md`](../../methodology/MULTIFORMAT_IMPL.md)。

## 3. 配置

- **token 预算**:`Chunker(target=800, min_tokens=200, max_tokens=1500)` 统一覆盖;或不传按 `doc_type` 用内置 `BUDGETS`(`Chunker(budgets={"my_type": (200,700,1200)})` 扩充)。
- **doc_type 路由(推荐)**:有编号类型(学术/规范)走分型预算即可;**无编号类型把期望放在"通用 parent-child"**,不要指望精确层级。`doc_type` 不传也能跑(走默认预算 + `text_level`)。
- **lang**:`"ch"`/`"en"` 影响 token 估算;中文务必传 `"ch"`。
- **每页一块**:`Chunker(page_grouped={"slides_tutorial","my_slide_type"})`。

## 4. 与原型 harness 的关系

`../scripts/chunk_document.py` + `retrieve_big.py` 是实验原型(绑 MinerU 目录与 manifest);本组件最初是它的解耦、可安装版本(当时 byte-parity 77/77)。**此后经 R1–R3 修复(reset-aware 定级 / 重复横幅守卫 / 内容回收 / aside_text 剔除)已与原型分叉——组件是现在的 source of truth,原型未跟进,仅留作历史参照。**

## 5. 测试

```bash
python -m pytest -q          # 42 项单测(test_core + test_table;自带 fixture,免外部数据;含 docx adapter 用例)
python examples/run_mineru.py            # fixture demo
python examples/run_mineru.py <mineru_dir>   # 真实文档
```
