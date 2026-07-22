# chunker

一个**可换的 chunking 组件**,用于 RAG indexing 管线的中间一节:

```
ingest → parse → [ chunker ] → embed
```

把 parser 的产物切成**带层级骨架(breadcrumb + 节树)的 chunk**,供 embedding 阶段嵌入与存储;查询期提供 **small-to-big** 大块组装。纯 Python、**零运行时依赖**。核心逻辑经 **77 篇真实文档 + 多轮对抗审核** 验证,自带单测 **42 项**全过(`test_core.py` + `test_table.py`)。

> **五格式已验证**:PDF(MinerU,77 篇)/ 扫描件 PDF(OCR,40 篇 13 语言)/ **Word(.docx)、PPT(.pptx)** 走 MinerU 原生 office 后端(各 ~50 篇;`adapters/docx.py`/`pptx.py` 降为零依赖 fallback)/ **Excel(.xlsx)** 走独立 `table_chunker.py`(网格非文档流,heading-tree 不适用,但输出同一 Chunk schema)。见 [`ARCHITECTURE.md` §格式覆盖](ARCHITECTURE.md) 与 [`../../methodology/MULTIFORMAT_IMPL.md`](../../methodology/MULTIFORMAT_IMPL.md)。
> 设计动机、取舍与三轮对抗审核结论:见 [`ARCHITECTURE.md`](ARCHITECTURE.md) 与 [`../../methodology/LAZY_HEADING_TREE_DESIGN.md`](../../methodology/LAZY_HEADING_TREE_DESIGN.md)(v2)。

---

## 安装 / 运行

零依赖,可直接用 `src/` 上路径运行,或 `pip install -e .`:

```bash
# 方式 A:装成包
pip install -e .            # 之后任意位置 `from chunker import Chunker`

# 方式 B:免安装(脚本里已把 src/ 加进 path)
python examples/run_mineru.py            # 自带 fixture 的端到端 demo
python examples/run_mineru.py <mineru输出目录>   # 跑真实 MinerU 解析结果
python -m pytest -q                       # 单测(用自带 fixture,免外部数据)
```

## 60 秒上手

```python
from chunker import Chunker
from chunker.adapters.mineru import from_mineru   # 或 from_mineru_dir("解析目录")

elements = from_mineru(content_list_json, layout_json)            # ① parse → Element[]
result   = Chunker().chunk(elements, doc_id="d1", doc_type="academic_paper", lang="en",
                           doc_meta={...}, acl={...})   # ② 切块(acl 默认 fail-closed;每 chunk 盖 ACL+doc_meta 供硬过滤/引用)

for c in result.chunks:                  # ③ 给 embedding:嵌入 text,其余当 metadata
    embed(c.text); store(c)              #    c.breadcrumb / c.section_id / c.section_anchor ...
store_sections(result.sections)          #    节树,供检索

big = Chunker().assemble_big(hit_chunk, result, elements)         # ④ 查询期 small-to-big
```

## 组件契约(三个接缝)

| 接缝 | 类型 | 谁负责 |
|---|---|---|
| **输入** | `list[Element]`(归一化 parse 单元) | parser **适配器**(`adapters/mineru.py`) |
| **输出** | `ChunkResult(chunks, sections)` | 本组件 |
| **检索 helper** | `assemble_big(hit, result, elements) -> BigBlock` | 本组件(查询期) |

→ parse 将来组件化时,只需让它吐 `Element[]` 或写个新适配器(见 [`docs/INTEGRATION.md`](INTEGRATION.md)),**核心与检索 helper 全部复用**。

## 目录结构

```
chunker/
├── pyproject.toml                # 可安装(零依赖)
├── README.md                     # 本文件
├── docs/
│   ├── ARCHITECTURE.md           # 设计/数据流/v2 与对抗审核结论
│   ├── API.md                    # 完整 API + 数据 schema 参考
│   └── INTEGRATION.md            # 接进管线 / 写新 parser 适配器 / 配置
├── src/chunker/
│   ├── types.py                  # Element / Chunk / Section / BigBlock(稳定 schema)
│   ├── core.py                   # 纯核心:定级·节树·assemble_text·资产
│   ├── meta.py                   # doc-level metadata + ACL 提取(extract_doc_meta)
│   ├── table_chunker.py          # xlsx 独立路径(TableChunker,网格 → 同一 Chunk schema)
│   ├── retrieve.py               # assemble_big(small-to-big)
│   └── adapters/{mineru,docx,pptx}.py   # parser 适配器(docx/pptx 为零依赖 fallback)
├── examples/run_mineru.py + fixtures/   # 自带 fixture 的 demo(也能跑真实目录)
└── tests/{test_core,test_table}.py      # 单测 42 项(用 fixture)
```

## 它做了什么(一句话)

`text_level`(parser 免费给)为主 + **小数点编号细分**(`2.1` 分层)+ **reset-aware 裸整数提级**(仅当全文编号单调才认大纲,循环重启则当列表放弃)+ 项目符号守卫 + **重复横幅守卫**(每页横幅剔除)→ 单调栈建节树 → 资产原子化 + **内容回收**(公式/脚注/ref 进正文,页边水印剔除)→ 每 chunk 挂 `section_anchor` → 查询期 `assemble_big` 按真实 token 取包围区(**过小往上并祖先 / 并相邻兄弟节**,横幅一致剔除)。

## 诚实定位

定级质量**取决于 parser 的 `text_level` 有多干净**,不取决于 doc_type:parser 给对(干净学术/规范)→ 精确层级、breadcrumb 准;parser 压平但编号单调(深度报告)→ reset-aware 恢复章节;parser 压平且编号循环(周报、封面密集中文研报)→ 诚实退化为"单根 + 扁平 L2"的通用大小切块,**不要指望章节嵌套**。三轮对抗审核(0 refuted)、格式范围与边界详见 [`docs/ARCHITECTURE.md` §7 / §格式 / §适用边界](ARCHITECTURE.md)。
