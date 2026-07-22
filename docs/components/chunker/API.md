# API 参考

`from chunker import Chunker, assemble_big, Element, Chunk, Section, ChunkResult, BigBlock`

## `Chunker`

```python
Chunker(target=None, min_tokens=None, max_tokens=None, budgets=None, page_grouped=None)
```
- `target/min_tokens/max_tokens`:传了 `target` 即用这组 `(min,target,max)` token 预算覆盖所有 doc_type;不传则按 `doc_type` 用内置 `BUDGETS`(`DEFAULT_BUDGET=(200,800,1500)`)。
- `budgets`:`{doc_type: (min,target,max)}`,覆盖/扩充内置预算。
- `page_grouped`:`set[doc_type]`,这些类型按"每页一 chunk"(默认 `{"slides_tutorial"}`)。

### `Chunker.chunk(elements, *, doc_id, doc_type=None, lang="en", doc_meta=None, acl=None) -> ChunkResult`
把 `Element[]` 切成 chunk + 节树。
- `elements`:`list[Element]`(parse 适配器产出)。
- `doc_id`:文档 id(用于 chunk_id / sec_id 前缀)。
- `doc_type`:可选,用于选预算 + `law` 特判;不传走默认预算。
- `lang`:`"en"` 或 `"ch"`(影响 token 估算:en≈chars/4,ch≈chars/1.7)。
- `doc_meta`:文档级 metadata dict,**盖到每个 `chunk.doc_meta`**(payload 过滤+引用)。用 `extract_doc_meta` 取。
- `acl`:访问策略 dict,**盖到每个 `chunk.acl`**(安全;深拷贝,可每 chunk 覆盖)。**不传 → fail-closed `RESTRICTED_ACL`**。检索须硬过滤,见 INTEGRATION §6。

### `extract_doc_meta(path, **overrides) -> dict`(ingest 助手,非 core)
从 office `docProps/core.xml` / PDF `/Info` 抽 title/author/created/modified + format;`**overrides`(source/doc_type/domain/lang)覆盖优先;去空/去 "admin" 等脏值。

### `Chunker.assemble_big(hit_chunk, result, elements, target=None, min_tokens=None, max_tokens=None, admit=None) -> BigBlock`
查询期 small-to-big。预算缺省继承构造时的设置/doc_type。**自动从 `result` 建 `acl_index` 传下去 → 默认 ACL 安全**(只取与命中块同 ACL 的 element,被收紧的兄弟小节不会被捞进 big.text)。`admit=acl->bool` 可换成自定义可见性谓词。等价于模块级 `assemble_big(hit, result.sections_by_id(), elements, acl_index=result.acl_index(), ...)`。

## `assemble_big(hit_chunk, sections_by_id, elements, target=800, min_tokens=200, max_tokens=1500, banners=None, acl_index=None, admit=None) -> BigBlock`
纯函数版。`sections_by_id`:`{sec_id: Section}`(用 `result.sections_by_id()`);`elements`:与切块时同一份(idx 对齐),用于按 anchor 取正文。`banners`:doc 级横幅集(传 `result.banners` 免重算;`None` 时内部重算)——保证大块与 chunk 期一致地剔除每页横幅。
- **`acl_index`**:`{idx: acl}`(用 `result.acl_index()`)。**安全**:big-block 跨 chunk 取材,传它则只取与命中块**同 ACL** 的 element(fail-closed:未知 idx 排除),被 per-chunk 收紧的兄弟小节明文不会泄漏进 `big.text`。不传 → legacy 无防护(假设全文档单一 ACL),`BigBlock.acl=None` 标记未校验。详见 INTEGRATION §6 铁律4。
- **`admit`**:`acl->bool`,自定义可见性谓词(覆盖"同 ACL 等价类"默认)——用于"取调用者一切有权看到的"(跨不同但可见的 ACL),复用硬过滤同一谓词。

## 适配器:`from chunker.adapters.mineru import from_mineru, from_mineru_dir`
- `from_mineru(content_list, layout=None) -> list[Element]`:`content_list` = 解析的 `*_content_list.json`(list);`layout` = `layout.json`(dict,仅 `merge_prev` 需要)。
- `from_mineru_dir(doc_dir) -> list[Element]`:便捷,读一个 MinerU 输出目录。

## 数据 schema(`types.py`,均为 dataclass)

### `Element`(输入)
| 字段 | 类型 | 说明 |
|---|---|---|
| `idx` | int | 阅读序(0 起、连续) |
| `kind` | str | `text\|table\|image\|chart\|list\|header\|footer\|page_number` |
| `text` | str? | 正文/标题文本 |
| `text_level` | int? | parser 给的标题级别提示 |
| `page` | int | 页号 |
| `bbox` | list? | 坐标(本组件不依赖,留给溯源) |
| `list_items` | list[str]? | 列表项 |
| `caption` / `footnote` | str? | 资产标题/脚注(归一化后) |
| `table_body` | str? | 表格 HTML(生成负载) |
| `asset_content` | str? | 图/图表 VLM 内容(低可信) |
| `sub_type` | str? | 如 line chart/flowchart |
| `merge_prev` | bool | 续接前块(跨页) |
| `image_path` | str? | MinerU 裁切图相对路径(images/*.jpg);供 VL 图像向量化;忠实透传,核不碰 IO |

### `Chunk`(输出,嵌入 `text`)
| 字段 | 说明 |
|---|---|
| `chunk_id` | `<doc_id>#0007` |
| `doc_id` / `kind` / `lang` | `kind ∈ text\|table\|image\|chart` |
| `text` | **嵌入这个** |
| `content_raw` | 资产生成负载(表 HTML / VLM 内容);文本为 None |
| `breadcrumb` / `section_path` | 祖先标题链 / `" > "` 拼接 |
| `section_id` / `section_anchor` | 所属节 id / `[start_idx, end_idx]`(供 small-to-big) |
| `page_start` / `page_end` | 页范围 |
| `source_indices` | 来源 Element idx 列表(可溯源) |
| `n_tokens` / `trust` | 估算 token / `high\|low` |
| `flags` | text/图:`captionless` / `vlm_content` / `image_only`(纯图,走图像向量、跳稀疏路) / `multi_page` / `merge_prev_stitched`;table 另有 `nontabular` / `header_only` / `chart_meta` / `cols:N-M` / `sheet:X` |
| `doc_meta` | 文档级 metadata dict(title/source/date/doc_type/...);payload 过滤+引用 |
| `acl` | 访问策略 dict(安全;硬过滤 + fail-closed `RESTRICTED_ACL` 默认) |
| `image_path` | image/chart 裁切图引用(相对 MinerU 输出根);供 VL 图像向量化;text/table=None;**已净化**(拒绝 `../`/绝对/UNC/`://`) |
| `doc_type` | 文档类型(chunk 期盖章);`assemble_big` 据此选查询期 token 预算(缺则 DEFAULT) |

### `Section`(节树)
`sec_id, doc_id, level, title, breadcrumb(含自身), start_idx, end_idx(exclusive), parent_sec_id`

### `ChunkResult`
`chunks: list[Chunk]` · `sections: list[Section]` · `banners: frozenset[str]`(doc 级每页横幅,chunk 期算一次,供 `assemble_big` 复用) · 方法 `sections_by_id() -> dict[str, Section]` · `acl_index() -> dict[int, dict]`(element idx → 其归属 chunk 的 acl;供 `assemble_big` 做 ACL 感知取材;idx 冲突取更严的 acl)

### `BigBlock`(检索产物)
`text, resolved_section, breadcrumb, n_tokens, climbed(往上并几层), anchor, note`

## 最小例

```python
from chunker import Chunker
from chunker.adapters.mineru import from_mineru_dir

els = from_mineru_dir("parsed/mydoc")
res = Chunker(target=800).chunk(els, doc_id="mydoc", doc_type="academic_paper", lang="en")
big = Chunker(target=800).assemble_big(res.chunks[5], res, els)
print(big.n_tokens, big.breadcrumb, big.resolved_section)
```
