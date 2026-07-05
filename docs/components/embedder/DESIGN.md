# embedder 设计

> RAG 管线的 embed + 检索组件,接 chunker 的 `Chunk[]`。**本地 4090** 部署:稠密 = Qwen3-VL-Embedding-8B(多模态)、稀疏 = **BM25(jieba 分词 + Qdrant IDF)**、向量库 = Qdrant(dense+sparse hybrid + payload filter)。位置:`src/embedder/`(与 `src/chunker/` 同仓平级包)。

## 1. 在管线里的位置

```
ingest ─► parse ─► chunker ─► [ embedder ] ─► (Qdrant)
                   Chunk[]        │                查询期:
                                  ▼                query ─► embed ─► query_points(hybrid + ACL filter)
                          dense + sparse 向量        ─► chunker.assemble_big(small-to-big,ACL门控)─► LLM
```

组件**只认 chunker 的 `Chunk`**(text/image_path/image_only/acl/doc_meta/source_indices/doc_type/section_anchor/...)。chunker 留的接口在此全部兑现。

## 2. 技术栈(已定 + API 已核实)

| 层 | 选型 | 关键接口 |
|---|---|---|
| 稠密 | **Qwen3-VL-Embedding-8B** | `model.encode([{"text":..}/{"image":path}])`(sentence-transformers);图文同 4096 空间;MRL 降 64–4096;query 加 `prompt=` instruction;本地 ~16–18GB GPU |
| 稀疏 | **BM25**(jieba 分词 + Qdrant `Modifier.IDF`) | 客户端 jieba 分词 → token;doc sparse=词频 tf,query sparse=1;Qdrant 服务端按 IDF 算 BM25 分。**零 GPU、CPU 即可** |
| 向量库 | **Qdrant** | named dense+sparse;`query_points(prefetch=[dense,sparse], query=FusionQuery(RRF), query_filter=Filter)` 服务端 hybrid |

**为何 sparse = BM25 而非 BGE-M3**:三轮调研收敛——① BM25 是业界 production 的 sparse 绝对主流(框架/Qdrant 标配);② 你的中文财报/法律**精确词**(数字/法条编号/型号)正是 BM25 强项(金融文档实证 BM25 超最强商用 dense);③ 零 GPU、与 Qwen3-VL dense 干净解耦;④ BGE-M3 的 sparse 是少数派且中文偏弱(MIRACL-zh 36.3),它的主流价值(dense)又被 Qwen3-VL 替代。**保留 BM25 vs BGE-M3-sparse 在真实精确词查询上的实测**(见 §7 待验证)。

## 3. Qdrant collection

```python
client.create_collection(
    "rag_chunks",
    vectors_config={"dense": VectorParams(size=1024, distance=Distance.COSINE)},   # Qwen3-VL MRL=1024 起步
    sparse_vectors_config={"sparse": SparseVectorParams(modifier=Modifier.IDF)},    # BM25:Qdrant 按 IDF 算分
)
# payload(schemaless dict): chunk_id/doc_id/kind/text/doc_meta/doc_type/section_id/section_anchor/
#   source_indices/page_start/page_end/image_path/flags/lang
#   + ACL: acl_unset(bool) / acl_visibility(str) / acl_tenant(str) / acl_allow(list[str])
# payload index(加速 ACL filter): acl_tenant, acl_visibility, acl_allow, acl_unset
```

`point.vector = {"dense": [...], "sparse": SparseVector(indices=token_ids, values=tf)}`;**image_only chunk 省略 sparse key**(纯图只靠图像稠密向量召回)。

## 4. ACL 硬过滤(安全核心,兑现 chunker 契约)

embed 时把 `chunk.acl` dict 拆进 payload 的 4 个 ACL 字段。检索 filter(`user.tenant` / `user.principals = groups + [uid]`):

```python
Filter(must=[
    FieldCondition(key="acl_unset", match=MatchValue(value=False)),       # fail-closed:未授权文档排除
    FieldCondition(key="acl_tenant", match=MatchValue(value=user.tenant)),# 租户隔离
    Filter(should=[                                                       # 嵌套:(allow ANY OR public)
        FieldCondition(key="acl_allow", match=MatchAny(any=user.principals)),
        FieldCondition(key="acl_visibility", match=MatchValue(value="public")),
    ]),
])
```

- **嵌套 should 是安全关键**:把 `(allow OR public)` 包成 `must` 的子 `Filter`,确定表达"租户 AND (允许 OR 公开)",不依赖顶层 must+should 语义。对应 chunker INTEGRATION §6"括号不可压平"。
- **fail-closed**:`acl_unset==true` 默认排除。**deny 不自动生效**(按契约,确需用 `must_not`)。
- 双层 ACL:检索 filter + 取材 `chunker.assemble_big(admit=acl_admits(user))`。

## 5. 数据流

**索引期**(`embed.py`):
```
for chunk in chunks:
    if "image_only" in chunk.flags:           # 纯图:只稠密(图像向量)
        dense = qwen3vl.encode_image(abs(image_path)); sparse = None
    elif chunk.kind in (image, chart):        # 带 caption 的图:图像稠密 + caption BM25
        dense = qwen3vl.encode_image(abs(image_path)); sparse = bm25_sparse(chunk.text)
    else:                                      # text/table:文本稠密 + BM25
        dense = qwen3vl.encode_text(chunk.text); sparse = bm25_sparse(chunk.text)
    vec = {"dense": dense} | ({"sparse": sparse} if sparse else {})
    client.upsert("rag_chunks", [PointStruct(id, vector=vec, payload=acl_split(chunk)+meta(chunk))])

# bm25_sparse(text): tokens = jieba.cut(text) (中文); 去停用词; token->uint32 稳定 hash;
#                    返回 SparseVector(indices=hashes, values=term_counts)  ← Qdrant IDF modifier 负责打分
```

**查询期**(`retrieve.py`):
```python
qd = qwen3vl.encode_text(query, prompt="Retrieve relevant documents for the query.")
qs = bm25_sparse(query)                          # query 端 values 用 1.0
acl = acl_filter(user)
hits = client.query_points("rag_chunks",
    # ACL filter 必须下推到每个 prefetch:嵌入式 QdrantLocal 在 fusion 下会丢弃顶层 query_filter
    # 的 should 子句(实测 §7 待验证#4),只 prefetch-level filter 才让 should 生效。顶层保留作双保险。
    prefetch=[Prefetch(query=qd, using="dense", filter=acl, limit=50),
              Prefetch(query=qs, using="sparse", filter=acl, limit=50)],
    query=FusionQuery(fusion=Fusion.RRF),
    query_filter=acl, limit=k, with_payload=True).points
for h in dedup_by_section(hits):                  # dedup:section_id=None 不折叠(按 chunk_id),seal#6
    mn, tg, mx = BUDGETS.get(h.doc_type, DEFAULT_BUDGET)   # per-doc_type 预算(slides/policy 整节),seal#8
    # acl_index 默认路径(只取与命中块同 ACL 的原文)而非 admit:big.acl=hit_acl 自然准确,出口可校验,seal#2
    big = chunker.assemble_big(shim(h.payload), secs, els, target=tg, min_tokens=mn, max_tokens=mx,
                               banners=banners, acl_index=acl_index)
    if not acl_admits(big.acl, user):            # 出口二次校验(铁律5):无权则不交付该上下文
        big = None
```

## 6. 模块

| 模块 | 职责 | 状态 |
|---|---|---|
| `config.py` | 模型路径/维度/Qdrant 连接/collection/停用词/sidecar 目录 | ✅ |
| `dense.py` | Qwen3-VL 封装(复用官方 `Qwen3VLEmbedder`,encode_text/image,instruction,MRL 截断,按名锁 4090)— 唯一吃 GPU | ✅ 实测(图文同空间) |
| `sparse.py` | **BM25**:jieba 分词 + 正则保精确串 + token→uint32 稳定 hash → `SparseVector`(doc=tf / query=1) | ✅ 单测 |
| `store.py` | Qdrant collection(sparse modifier=IDF)/ payload index / upsert / **hybrid + ACL 硬过滤**(filter 下推 prefetch) | ✅ ACL 单测 |
| `acl.py` | ACL 客户端逻辑(集中便审计):`acl_split`(索引拆字段)+ `acl_admits`(检索逐元素谓词),与 store 服务端同语义 | ✅ |
| `embed.py` | chunk 分流(image_only→图向量/其余→文字+BM25)+ ACL 拆解 + payload + per-doc sidecar(version+elements/sections/acl_index) | ✅ 端到端 |
| `retrieve.py` | hybrid 召回 + **可选 rerank 精排** + dedup_by_section + 接 `chunker.assemble_big`(ACL 感知 small-to-big) | ✅ 端到端 |
| `rerank.py` | **Qwen3-VL-Reranker-8B** cross-encoder 精排(复用官方 `Qwen3VLReranker`,对 hybrid top-N 重排,按名锁 4090)— 可选、第二个吃 GPU | ✅ 评估 MRR 0.566→0.867 |
| `types.py` | `User`(tenant+principals)/ `Hit` 检索结果契约 | ✅ |

## 7. 决策与取舍

- **sparse=BM25**:业界主流 + 精确词强 + 零 GPU(只 dense 吃 GPU,环境大大简化)。
- **dense MRL=1024 起步**:4096 贵 4x,1024 召回通常几乎不掉;按检索质量再调。
- **中文分词**:jieba 是变量——也可换 Qdrant 1.15+ 服务端 CJK 分词器(`Document` + BM25)。先 jieba 客户端(可控),实测对比。
- **token→uint32**:稳定 hash(可能极低碰撞率,可接受);需 doc/query 同一 hash 函数。
- **est_tokens → 真 tokenizer**:✅ 实测 2927 真实 chunk(Qwen3-VL tokenizer)——散文 char/token≈3.85(≈现值 4.0,误差<4%),偏差全在数字/表格密集文档(财报 5.08/政府 5.35/中文研报 1.51),单值无法兼顾两簇且改均值反害散文;两个误差方向都被 small-to-big + 32k 上限兜住 → **保持现值不重标**(见 core.est_tokens 注释)。
- **第 4 档 oversplit**:retrieve 用 `source_indices` 把列切/行组切散的 xlsx 记录回拼。

**待验证(实测,环境就绪后一次性)**:
1. ✅ **已验证(BM25 确认)**:14 文档/1067 chunk,精确词查询 25(程序化挖稀有串)+ 语义查询 31(agent 生成、避开原词),
   单路 MRR/Recall@10(`scratchpad/eval/`):
   - **精确词**:bm25 **0.738**/0.96 > bgem3 0.584/0.88 ≫ dense 0.149/0.24 —— BM25 在 sparse 主场(编号/型号/金额一字不差命中)**明显赢** BGE-M3;
   - **语义**:dense **0.794**/0.94 ≫ bgem3 0.347 > bm25 0.210 —— 语义是 dense 的活,两 sparse 都弱;
   - **整体**:dense 0.506 > bgem3 0.453 ≈ bm25 0.446(两 sparse 打平)。
   **结论:选 BM25 正确** —— sparse 该负责的精确词 BM25 胜,语义交给 dense(BGE-M3 的语义优势在 hybrid 里冗余),
   而 BM25 零模型/零 GPU/零维护 vs BGE-M3 2.3GB+GPU。注:合成集、规模小,趋势可信、绝对值仅供参考。
2. ✅ **已验证**:Qwen3-VL 图文跨模态召回(`scratchpad/verify_dense.py`,2 张真实论文配图 + 准确/无关描述交叉相似度)。
   文字准确描述 → 对应图相似度 0.74/0.49,→ 非对应图仅 0.37/0.17,无关描述(金毛沙滩)对两图 0.07–0.11。
   **文字 query 能跨模态召回 image_only chunk 成立**,且模型对齐的是图的语义内容(抽象描述也能命中流程图)。
3. ✅ **MRL 已验证**:1024 vs 全维 4096 几乎无损 —— 对角线 0.7388/0.4930 → 0.7363/0.4483,分离度仍健康
   (A=0.34/B=0.30),无关描述始终 ≤0.11。**dense_dim=1024 起步确认**(省 4× 存储/检索)。
   ✅ **RRF 权重已验证**:dense+bm25 加权 RRF 扫描,峰值 w_dense≈0.4 整体 MRR **0.541** > 纯 dense 0.510 > 纯 sparse 0.449
   —— **hybrid 确实优于任何单路**(dense 管语义、sparse 管精确词,互补)。等权(0.50)MRR 0.499 已接近峰值。
   **决定:store 保持 Qdrant 标准 RRF(等权,服务端融合简洁),不改客户端加权** —— 0.04 提升不值客户端融合复杂度,
   且最优 0.4 强依赖查询 exact:semantic 比例(本集≈1:1)会随真实分布漂移;原则:**sparse 权重别给太低**;
4. ✅ **已验证**:Qdrant `Modifier.IDF` 在嵌入式 local mode **支持**(sparse 路返回结果);BUT 实测发现
   **嵌入式 QdrantLocal 在 fusion(RRF)模式下静默丢弃顶层 `query_filter` 的 `should` 子句**(顶层/嵌套
   should 都失效,只 must 等值条件生效)——这会**让 ACL 的 "(allow ANY) OR public" 退化成只过滤 tenant,
   越权文档泄漏(fail-open)**。正解:**filter 下推到每个 `Prefetch(filter=acl)`**,fusion 只融合已过滤结果
   (过滤仍在召回层 = fail-closed 不破坏,且 limit 不被无权结果污染)。已在 `store.py` 落地,
   `tests/test_store.py` 用"越权不可召回"断言守住。诊断脚本:`scratchpad/diag_acl.py`。

## 8. 环境前提(实现/实测要先就绪)

- **GPU 环境**:**仅 Qwen3-VL-8B dense 要 CUDA torch**(当前默认 python 是 CPU torch);BM25 sparse 纯 CPU(jieba),无需 GPU。
- **依赖**:`sentence-transformers transformers>=4.57 torch(cuda) qwen-vl-utils qdrant-client jieba`(去掉 FlagEmbedding)。
- **Qdrant**:嵌入式(`QdrantClient(path=...)`,零 docker)起步。**约束:同一 path 同时只允许一个 client**
  (单进程内 embed+retrieve 必须共享 Store/Dense —— `Embedder/Retriever` 接 `store=`/`dense=` 复用;否则
  `AlreadyLocked` + 重复 load 8B)。规模大了换 server 模式即无此约束、且 payload index 才生效。
- **模型**:Qwen3-VL 官方 8B(~16GB,modelscope 下到 `~/models`)+ 自带 `scripts/qwen3_vl_embedding.py`(dense.py 复用);BM25 无模型。

## 9. 端到端验证(MVP 里程碑)

`scratchpad/e2e.py`:真实论文(acl.long.386,MinerU 输出)走完 parse→`from_mineru`→`Chunker.chunk`(盖 ACL)→
`Embedder.index_document`→`Retriever.search_with_context`。实测:

- **链路**:247 elements → 58 chunks → 58 indexed(0 纯图,6 个 image/chart 带 caption 走文字路)。
- **ACL fail-closed 端到端**:有权(t1/g_research)召回精准(top 0.83 正中 query);**跨租户(t2)= 0 命中,同租户无组(g_other)= 0 命中** —— 无权者在 Qdrant filter 层即拿不到任何 chunk。
- **small-to-big ACL 感知**:`big.acl` 正确盖章,climbed/tokens 合理。
- **图文召回**:描述图的 query → top1 命中 image chunk(此文档走 caption 路;纯图像向量路由 `verify_dense.py` 独立验证:准确描述↔对应图 0.74/0.49,无关 0.07–0.11)。

## 10. 封板对抗 review(进下阶段前,4 维度 finder × 独立验证)

对抗 workflow(ACL安全/数据流/检索契约/鲁棒性 4 finder → 每 finding 独立验证者反驳)挖出 13 confirmed,去重 10 真问题全修 + 回归锁:

| # | 问题 | 严重 | 修复 |
|---|---|---|---|
| 1 | 空 tenant 与空 tenant 用户自匹配 → 租户隔离 fail-open | high | `acl_split` 空 tenant 视同 unset;`acl_admits` 空 tenant 双向拒绝(`test_acl`) |
| 2 | admit 路径 `BigBlock.acl` 低报 big.text 实含的更严 ACL(铁律5) | med | 改用 `acl_index` 默认路径(只取同 ACL)+ 出口 `acl_admits(big.acl,user)` 二次校验(`test_retrieve`) |
| 3 | `point_id=uuid5(chunk_id)` doc_id 复用静默覆盖 | low | 契约文档化(doc_id 全局唯一) |
| 4 | `dense_dim` 改后 `ensure_collection` 早返回 → 维度漂移崩/检索错 | high | 已存在分支断言 size==dense_dim,fail-fast(`verify_seal4`) |
| 5 | BM25 doc tf 对字母数字 token 双计(jieba+正则)→ 不对称扭曲 | med | `tokenize` 只补 jieba 未完整切出的精确串(`test_sparse`) |
| 6 | `dedup_by_section` 把所有 `section_id=None` 折叠成一条 → 丢召回 | high | None 退化按 chunk_id 去重(`test_retrieve`) |
| 7 | `assemble_big` lang 只认 `ch`,`zh` 别名落 en → 中文 token 低估 2.35× | high | 对齐 `est_tokens` 的 `startswith(("ch","zh"))`(chunker 回归) |
| 8 | embedder 走自由函数丢 per-doc_type 预算 → slides/policy 被截 | med | `_ChunkShim` 加 doc_type,按 `BUDGETS` 取预算传入(e2e:1286→800) |
| 9 | sidecar 写非原子 → 崩溃留半截 JSON → 该 doc 永久 assemble 崩 | high | 写 .tmp + fsync + `os.replace`(原子) |
| 10 | sidecar 缺失/损坏无容错 → 一条坏 hit 拖垮整次查询 | high | `_load_sidecar` 显式报错 + `search_with_context` try/except 降级(`test_retrieve`) |

embedder 单测 17 全绿(test_acl/test_sparse/test_store/test_retrieve)+ chunker 回归 + e2e(含 e2e_xlsx)+ verify_seal4。封板就绪。

**封板后打磨**(lazy-tree review C 清单):sidecar schema 版本绑定 —— 写侧 `_write_sidecar` 盖 `SIDECAR_VERSION`(`config.py`),读侧 `_load_sidecar` 在反序列化前校验,不符(含旧 sidecar 无 `version` 字段)抛 `ValueError` 响亮失败。**故意不在 `search_with_context` 的 `(FileNotFoundError, JSONDecodeError)` 降级 catch 内**:缺文件/坏 JSON 是单 doc 瞬态(降级),版本不符是系统性 schema 漂移(所有 sidecar 皆旧),应提示整体重建而非逐条静默降级。改 sidecar 结构(Element/Section 字段、acl_index 编码)时须 `SIDECAR_VERSION += 1`。回归:`test_retrieve` 的写侧盖章 + 读侧不符×2(显式旧版本号 / 缺字段)。
