# Heading-Skeleton RAG —— 设计 v2（经对抗性审核 + 真实数据实证修订）

> v1 想法是"ingest 只留最小编号骨架 + 命中时懒重建层级"。**对抗性审核(4维+逐条独立验证,跑了 77篇/5337标题真实数据)推翻了 v1 的两个命根**,v2 据实证重写。
> 一句话变化:**默认从"懒"翻成"eager-but-cheap";层级从"赌编号"翻成"text_level 为主的多信号融合";pre-TOC 剥离进 ingest;"懒"降级为罕见特例。**

---

## 0. 审核结论(v1 错在哪,为什么 v2 这么改)

实证(`analysis/per_doc_stats.csv` + `parsed/*/content_list.json`):

1. **核心赌注"有编号→零 LLM"只覆盖 ~13%**:全语料仅 10.6% 标题能解析出编号;按类型 academic 0.73,financial/law/policy/slides/brochure ≈0。**无编号是主体不是尾部** → 不能把"默认零 LLM"当卖点。
2. **font_rank 兜底不可靠**:bbox 高度分不开层级(10-K 的 L1/L2 同高);法律/财报的 `SEC./TITLE/DIVISION/Item` 让 num_depth 塌成 depth=1。
3. **"懒"是过早优化**:eager 全树 77 篇共 66.6ms(<1ms/篇),与查询期 StepA 同算法 → 懒省不到东西,反而把零成本工作搬上热路径 + 加缓存失效。
4. **真长板**:编号密集的**学术论文**上,编号分段定级可靠、优于 MinerU 平层级——v2 保留它作为该类的强信号。
5. **我过虑的**(实测纠偏):阅读序 idx<E 0 违例、标题误检 1.3%、section 过大 1% —— 都不是重点;真实常态是 **section 过小(中位 42 token)**。

---

## 1. 总体(v2):eager 多信号骨架,懒只留给特例

```
ingest(每文档一次,确定性、亚毫秒、默认零 LLM):
  解析(MinerU md+layout) → 候选检测 → pre-TOC 剥离 → 多信号定级 → 单调栈建树
  → 把 breadcrumb + section_anchor 折叠进每个 leaf chunk(不内联文本)
query(命中即读,不重建):
  flat 检索命中 → 直接读 chunk 上挂好的 breadcrumb + section_anchor
  → 按目标 token 取包围区(过小则上爬/并兄弟)→ 装配
懒(仅特例):标题数极大(如 news 1335)+ 查询极稀疏 + ingest 预算硬约束 时,才不预建、改命中现建+缓存
```

---

## 2. Ingest:eager 多信号层级骨架(确定性)

### 2.1 候选检测 + pre-TOC 剥离(强制)
- 候选 = MinerU 标过 `text_level` 的元素 ∪ 行首编号正则命中的元素。
- **pre-TOC 剥离(v1 漏掉、必须加)**:用 `(\.{3,}|…|\s)\d{1,3}\s*$` 点引导+尾页码 + `toc_like_headings` 信号,把目录/图表目录区的条目**排除出候选索引**;并对**编号重启**(同 token 多次出现,如每章 1.1)做**分段隔离**,防跨段污染单调栈。

### 2.2 多信号定级(融合,非赌编号)——按优先级
1. **text_level(主信号,免费)**:MinerU 已给,直接用作基线层级。**v1 错在丢弃它**;v2 以它为主。
2. **编号分段(有则最强,做校正)**:解析编号 token 按 `.` 分段 → `2`=1、`2.1`=2。**仅在编号密集文档(学术)凌驾 text_level**;并扩充体裁前缀映射:`SEC./Article/TITLE [罗马]/PART/Item N/第N条/一、二、` → 显式 depth 表(确定性、仍零 LLM)。
   - **守卫**:纯数字 token 若是 19xx/20xx 年份或 >40 的量,**不当编号**(防 `2020 业绩`、`52 Places` 误触发,实测 ~4%)。
3. **font/版式(弱兜底,只做同级 tiebreak)**:**不把 bbox 高度当可比的全局 level**;字号分不开时**显式标 `level_unreliable`,退化为线性 breadcrumb(只给最近上一个标题,不强排父子)**,而不是默默拼错。
4. **冲突裁决**:编号与 text_level 冲突时,编号密集文档信编号、否则信 text_level。

### 2.3 建树 + 折叠进 chunk
- 单调栈建树(微秒级)。把每个 leaf chunk 标注:`breadcrumb`(祖先标题链)+ `section_anchor`(所属节的 idx 区间)+ `level_reliable: bool`。
- **零 LLM**。LLM 不在 ingest 主路径。

### 2.4 LLM 的诚实定位(不是"罕见尾部")
- 无编号多层文档(语料主体)若 text_level 也不可信,**要么接受 text_level 原值(廉价、breadcrumb 可能只 1 层、best-effort)**,**要么离线/ingest 期(非查询热路径)按 doc_type 选择性上一次 LLM 消歧并缓存**。
- **明确**:这是**按 doc_type 路由的可选增强**,不是"绝大多数零 LLM"。成本模型对 `numbered_frac<0.2` 的文档单列。

---

## 3. Query:直接读,不重建

1. flat 检索命中 chunk → 直接读它 ingest 时挂好的 `breadcrumb`(几十 token,白送上下文)。
2. **取包围区(big)按真实 token 量,不靠脆弱的 font 层级**:section_anchor 的 idx 区间 + assemble_text 实算 token。
   - **过小(主路径,中位仅 42 token)→ 沿 breadcrumb 上爬祖父 / 并相邻同级兄弟,直到 ≥ 目标**(用 anchor 区间实算大小驱动,不靠 font 层级)。
   - 过大(罕见,1%)→ 取 hit 邻域窗口。
3. 多命中同 section 去重(类 auto-merge);单孤立 hit 可只给 breadcrumb+hit。
4. `level_unreliable` 的文档:breadcrumb 退化为"最近上一个标题"的线性上下文,**诚实标注 best-effort**。

---

## 4. 存储

- 向量库:leaf chunk(挂 breadcrumb + section_anchor + level_reliable)。
- docstore:每 doc 元素流(供取包围区文本 + 上爬)。
- **无需查询期缓存层**(eager 已预建);懒特例才需缓存。

---

## 5. 可选:跨引用扩展(差异化,默认关)
ingest 廉价正则抽 `见附录G/表5/§2.3` 存边;查询时若 hit 区含与 query 相关引用,额外拉被引目标 section。补"±相邻够不到附录G"的洞。

---

## 6. "懒"什么时候才真的赢(收窄后的适用前提)
**同时满足**才上懒,否则 eager-cheap:
1. 语料**高频局部更新**(单文档秒级改写/版本爆炸)使 eager 预算总和也变贵;**且**
2. 命中**极稀疏长尾**(绝大多数 section 永不被命中,eager 预建被浪费);**且**
3. 单文档**标题数极大**(如 news 1335、form-p17 193;本语料仅 3/77 篇)。
本语料三条都不满足 → **默认 eager**。

---

## 7. 失败模式(经实证更新:✅真问题 / ❌我过虑了)

- ✅ **无编号是主体**(74% 文档 frac<0.2)→ text_level 为主、LLM 按类型路由(§2.2/2.4)。
- ✅ **font_rank 不可分级** → 不当全局 level,退线性 breadcrumb(§2.2.3)。
- ✅ **法律/财报编号塌缩** → 体裁前缀 depth 表(§2.2.2)。
- ✅ **混标度** → 统一以 text_level 为基线,编号做校正,避免 num_depth 直接 vs font_rank(§2.2)。
- ✅ **TOC 污染 + 编号重启** → ingest pre-TOC 剥离 + 分段隔离(§2.1)。
- ✅ **section 普遍过小** → 上爬/并兄弟为主路径,实 token 驱动(§3.2)。
- ✅ **dotted 误触发年份/量** → 守卫(§2.2.2)。
- ❌ 阅读序 idx<E:实测 0 违例,**非问题**,不再防御。
- ❌ 标题误检:实测 1.3%,不设复杂兜底(扫描件做一次正则规整即可)。
- ❌ section 过大裁窗口:实测 1%,降为次要。

---

## 8. 一句话(v2 定位)
**不是"懒重建层级",而是"eager 多信号(text_level为主+编号校正+TOC剥离)建廉价骨架,把 breadcrumb/section_anchor 顺手挂进 chunk;查询直接读、按真实 token 取包围区(过小则上爬)"。** 编号分段定级在学术类是强长板;"懒"只留给巨标题数+超稀疏+高更新的罕见特例。这版本质上收敛到了成熟系统(Knowhere/parent-child)的 ingest-时结构化——对抗审核把一个聪明但窄的想法,逼成了一个诚实、按数据分型的设计。

## 9. 参数
`target=800, min=200, max=1500` token;编号年份守卫 19xx/20xx;LLM 按 doc_type(numbered_frac<0.2 类型)离线可选,不入查询热路径。
