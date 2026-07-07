# 04 企业级 ACL 与安全模型

> **本篇导读**
> 这篇讲 pharos 如何在 RAG 里做权限:从 chunker 的 ACL 盖章、embedder 的召回层硬过滤与出口复核、small-to-big 的等价类取材,到服务层的三种身份模式与不泄密细则——以及"防线纵深里如何证明某一层自身有效"的测试方法论。
> **面试权重:高。** "多租户 RAG 怎么做权限"是企业级 RAG 面试的必问题,而"嵌入式向量库在 fusion 下丢 filter 子句"这种实测抓到的 fail-open 是罕见的一手素材。
> **前置阅读**:建议先了解 small-to-big 检索的基本形态(索引小块、交付大块);评估口径见 [07 评估方法论](07-evaluation.md)。

---

## 1. 概念底座:为什么 RAG 里的权限比看上去难

不依赖本项目,先把问题本身讲清楚。

**RAG 把原有的权限边界打碎了。** 传统文档系统里,权限挂在"文件"上:你没权限,打不开这个文件,故事结束。RAG 的第一步却是把文件切成几百个 chunk、编码成向量、混进一个大库——然后用"语义相似度"而不是"权限"来决定给用户看什么。这带来一个反直觉的危险:**无权内容与查询越相关,越容易被召回**。一个普通员工问"高管薪酬方案",向量检索会忠实地把 HR 机密文档排在第一名——除非有人拦住它。

**主流方案光谱**(按隔离强度从弱到强):

| 方案 | 做法 | 问题 |
|---|---|---|
| **事后过滤**(post-filtering) | 先按相似度召回 top-k,再按权限筛掉无权的 | ① 无权结果污染 limit:top-10 里 8 条被筛掉,用户只拿到 2 条;② 过滤代码有 bug 时直接泄漏(fail-open);③ "先拿到再丢弃"本身就是一次越权读 |
| **召回层过滤**(pre-filtering) | 把权限编码成向量库的 metadata filter,检索时下推 | 依赖向量库 filter 的正确性与表达力;权限模型复杂时 filter 构造容易出错 |
| **物理隔离** | 每租户/每权限域一个独立 collection 或索引 | 最强隔离,但运维成本 O(租户数),跨域共享文档(如全员公告)要重复存 |

**RAG 特有的三个绕过面**,是这个领域比"数据库行级权限"更难的地方:

1. **上下文扩展会绕过检索过滤。** small-to-big、"取回命中块周边原文"这类机制,在硬过滤*之后*从原始文档重新取材——过滤挡住了"检索到无权 chunk",挡不住"从有权 chunk 出发把无权邻居的原文捞进上下文"。
2. **by-id 直读接口。** 凭 chunk_id / doc_id 直接取内容的工具(expand、get_document)根本不走检索路径,检索层的 filter 对它无效。
3. **错误信息泄露存在性。** "404 不存在" 与 "403 无权" 的区别本身就是信息:攻击者可以用它枚举出"有一份我看不到的文档叫 X"。

最后两个通用概念,贯穿全篇:

- **fail-open vs fail-closed**:权限信息缺失/损坏/语义不明时,默认放行还是默认拒绝。安全系统的铁律是 fail-closed——所有"配置不完整"都应该表现为"看不到"或"起不来",而不是"全都能看到"。
- **身份(AuthN)与授权(AuthZ)分层**:"谁在问"和"能看什么"是两个正交问题。前者由服务边界解决(API key、SSO),后者由数据层解决(ACL 过滤)。混在一起做的系统,换一种接入方式(HTTP → MCP → CLI)就要重写一遍权限。

---

## 2. Pharos 怎么做

### 2.0 全景:一条 ACL 的一生

```
manifest 声明 acl ──► chunker 盖章(缺省=RESTRICTED,fail-closed)
                        │  每 chunk 深拷贝一份 acl;acl_index() 记录 {元素idx → acl}
                        ▼
                embedder 索引期:acl_split 拆成 4 个可过滤字段进 Qdrant payload
                        │  原始 acl 也存 payload(出口复核用);acl_index 存 sidecar
                        ▼
        ┌── 第一道闸:召回层硬过滤 acl_filter(下推到每个 prefetch)
        │
        ├── 第二道闸:small-to-big 取材门控(acl_index 同 ACL 等价类)
        │
        └── 第三道闸:出口逐条 acl_admits 复核(所有交付路径,含 by-id 直读)
                        ▲
   服务层身份(keys/legacy/open)每请求解析出 User{tenant, principals} 喂给上面三道闸
```

分层原则来自 [DESIGN.md D5/D12](../DESIGN.md):**身份在服务层,ACL 在检索层**。服务层只回答"谁在问",把它变成一个 `User(tenant, principals)`;"能看什么"的兑现收敛在 embedder 一处。这条分层让 HTTP、MCP 适配器、stdio 直连三个入口共享同一个 fail-closed 模型,而不是各写一套。

### 2.1 索引期:盖章、拆解、防走私

**盖章 fail-closed。** `Chunker.chunk` 不传 acl 时,默认盖 `RESTRICTED_ACL = {"visibility": "restricted", "allow": [], "unset": True}`——一份没人声明权限的文档,谁都看不到,而不是谁都看得到([src/chunker/core.py:36](../../src/chunker/core.py#L36)、[src/chunker/core.py:340](../../src/chunker/core.py#L340))。每个 chunk 拿到的是 `deepcopy`([src/chunker/core.py:348](../../src/chunker/core.py#L348)),允许 per-chunk 单独收紧而互不污染。

**acl 与 doc_meta 分通道。** doc_meta 是"过滤+引用"的便利通道,绝不是安全通道。`extract_doc_meta` 用 `_ACL_KEYS` 黑名单拒绝 manifest 把 `acl/tenant/allow/visibility` 等权限语义键走私进 doc_meta([src/chunker/meta.py:18](../../src/chunker/meta.py#L18)、[src/chunker/meta.py:62-63](../../src/chunker/meta.py#L62-L63))——否则某个下游若误把 doc_meta 当权限读,一份投毒的 manifest 就能伪造策略。

**acl_split:把自由 dict 拆成可过滤字段。** 索引期 [src/embedder/embed.py:69](../../src/embedder/embed.py#L69) 调 `acl_split` 把 chunk.acl 拆成 4 个 payload 字段:`acl_unset / acl_tenant / acl_allow / acl_visibility`([src/embedder/acl.py:14-26](../../src/embedder/acl.py#L14-L26))。这里有个被对抗审查抓过的细节(seal#1):**空串 tenant 一律标 `unset=True`**——如果放行空 tenant,它会与"空 tenant 的用户"自匹配,租户隔离形同虚设、只剩 public 一道把关,即 fail-open。无法安全归属租户的文档默认拒绝;真要全局公开必须显式声明。

### 2.2 第一道闸:召回层硬过滤,以及"嵌入式 fusion 丢 should"的坑

**filter 结构。** `acl_filter` 构造的是:`must=[acl_unset==False, acl_tenant==user.tenant, 嵌套Filter(should=[allow∩principals, visibility==public])]`([src/embedder/store.py:60-81](../../src/embedder/store.py#L60-L81))。注意 `(allow OR public)` 用的是**嵌套 Filter**,不是把 should 平铺到顶层——顶层 must+should 混用的语义在不同引擎/版本间不确定,"括号不可压平"是显式契约。doc_ids/doc_type/kind 等可选过滤一律追加进 must:与 ACL 是 AND,只收窄、绝不放宽。

**本篇最重要的坑:嵌入式 QdrantLocal 在 RRF fusion 模式下会静默丢弃顶层 `query_filter` 的 should 子句。** hybrid 检索的结构是 dense/sparse 各一个 Prefetch 先召回、顶层 `FusionQuery(RRF)` 融合。实测(诊断脚本最小复现)发现:fusion 模式下顶层 filter 只剩 must 等值条件生效,`(allow OR public)` 整个消失——ACL 退化成只过滤 tenant,**同租户内无权文档照常被召回,fail-open**。而单路直查(无 fusion)不踩这个坑,所以只测 dense 单路会完全漏掉它。

修复不是绕开 fusion,而是把 ACL filter **下推到每一个 Prefetch**([src/embedder/store.py:118-120](../../src/embedder/store.py#L118-L120)):fusion 只融合已过滤的结果,过滤仍然发生在召回层,limit 不被无权结果污染;顶层 `query_filter` 保留作双保险([src/embedder/store.py:121-123](../../src/embedder/store.py#L121-L123))。阶段 D 迁到 Qdrant server 模式时,没有假设 server 行为与嵌入式相同,用 raw fusion 探针(绕过出口复核直接断言 server 原始输出)重验了一遍——`test_server_fusion_no_should_leak_raw` 常驻回归。

这个故事的通用教训:**托管你安全语义的基础设施组件,其行为差异(嵌入式 vs server、版本升级)本身就是安全面**。"filter 传进去了"不等于"filter 生效了",必须用泄漏断言实测。

### 2.3 同语义双实现:acl_admits 与 acl_filter 互为镜像

ACL 语义集中在一个不到 40 行的文件里([src/embedder/acl.py](../../src/embedder/acl.py)):`acl_split` 是索引期拆解,`acl_admits` 是检索期的客户端等价谓词([src/embedder/acl.py:29-36](../../src/embedder/acl.py#L29-L36)),二者与服务端 `store.acl_filter` **严格同语义**。为什么必须同语义?因为 small-to-big 按 idx 范围从 sidecar 重取原文,判定"这个元素能不能进 big-block"用的是 `acl_admits`——如果它比服务端 filter 松哪怕一点,它就是硬过滤的旁路。这个同构关系有专门单测锁死(`tests/engine/test_acl.py::test_split_admits_same_semantics`),改一边不改另一边会当场红。

### 2.4 第二道闸:small-to-big 的 acl_index 等价类取材

这是 chunker 侧安全审计的核心结论,审计原话是"**生产闭合、检索敞开**":两条 chunk 生产路径都 fail-closed 了,但 `assemble_big` 在硬过滤*之后*按 idx 范围从原始 elements 重新取材,会跨 chunk 边界——**同文档 ≠ 同 ACL**。命中一个 public 小节,small-to-big 会把同区间内被 per-chunk 收紧的兄弟小节明文捞进 big.text。旧契约"big-block 只在同文档内取材故不越权"被实测证伪。

修复是三件套:

1. **acl_index**:`ChunkResult.acl_index()` 建 `{element idx → 产出它的 chunk 的 acl}`,同一 idx 出现在两个 chunk 时取更严的(`_stricter` 按 openness 比较,[src/chunker/types.py:7-19](../../src/chunker/types.py#L7-L19)、[src/chunker/types.py:93-104](../../src/chunker/types.py#L93-L104))。它随 sidecar 落盘([src/embedder/embed.py:116-117](../../src/embedder/embed.py#L116-L117))。
2. **等价类门控**:`assemble_big` 默认只取 `acl_index[i] == hit_acl` 的元素——与命中块 ACL **完全相等**才放行,未知 idx fail-closed 排除([src/chunker/retrieve.py:102-105](../../src/chunker/retrieve.py#L102-L105))。等价类判等(list 顺序敏感)刻意偏保守:方向是安全,代价是极端情况少取一点材。
3. **出口可校验的 acl 标记**:`BigBlock.acl` 记录组装文本的有效 ACL;legacy 无防护模式显式 `acl=None`,含义是"这段文本**未经访问校验**"([src/chunker/types.py:116-118](../../src/chunker/types.py#L116-L118))。另有一个防误用设计:传 `admit=`(调用者自定义可见性谓词)时必须同时传 acl_index,否则直接 raise([src/chunker/retrieve.py:94-98](../../src/chunker/retrieve.py#L94-L98))——因为没有逐元素 acl 时 admit 只能按 hit_acl 判,却会把 big.acl 盖成"已验证",构成静默跨 ACL 泄漏。

效果实测(真语料 government 文档,568 元素/80 chunks,人为收紧其中一个 chunk):legacy 路径 3/79 个 big-block 泄漏收紧内容的明文,修复后 **0/79,且 78/79 个块仍正常长大**——安全没有以召回为代价。还有一条反向的"契约锁"测试:`test_assemble_big_legacy_path_is_unguarded` 断言 legacy 路径**必然泄漏**——未来谁静默把无防护模式改成默认,测试会红。用测试锁死一个已知漏洞的形态,这个手法本身值得记住。

配套的完整性断言:`assemble_big` 与 sidecar 加载都强制 `elements[i].idx == i`(密集有序,[src/chunker/retrieve.py:118-120](../../src/chunker/retrieve.py#L118-L120)、[src/embedder/retrieve.py:89-90](../../src/embedder/retrieve.py#L89-L90))——取材和 ACL 门控都按位置索引,稀疏/乱序的 sidecar 会把无权元素错配成可见,宁可响亮失败。

### 2.5 第三道闸:出口二次校验(所有交付路径)

第一道闸挡检索,第二道闸挡取材,但还有三类路径可能绕过它们:by-id 直读、fusion 引擎行为差异、sidecar 组装。所以**所有裸 hit/payload 交付路径在出口逐条 `acl_admits` 复核**,这是 mode 无关的硬兜底(INTEGRATION 铁律5):

- `hybrid_search` 返回前对每个 point 的嵌套 acl 复核,acl 缺失视为 `{}` → 拒([src/embedder/store.py:125-129](../../src/embedder/store.py#L125-L129));
- `list_documents` scroll 逐条复核——scroll 路径下嵌套 should 的行为未单独验证过,fail-closed 不赌([src/embedder/store.py:131-158](../../src/embedder/store.py#L131-L158));
- `get_by_chunk_id` 凭 `uuid5(chunk_id)` O(1) 直取,**完全绕过了 acl_filter,必须复核**;无 point 与无权统一返回 None,不区分"不存在"与"无权",防探测([src/embedder/store.py:160-172](../../src/embedder/store.py#L160-L172));
- `search_with_context` 对组装出的 big-block 校验 `big.acl`,`acl=None`(未经校验)一律拒绝交付,状态标 `single_chunk_acl`([src/embedder/retrieve.py:157-160](../../src/embedder/retrieve.py#L157-L160));
- `expand` 是三道闸的完整缩影:get_by_chunk_id 复核 → assemble_big 逐元素 `==hit_acl` 门控 → 出口 `acl_admits(big.acl)`([src/embedder/retrieve.py:244-259](../../src/embedder/retrieve.py#L244-L259))。

一个精细的反例说明这套系统不是"处处加锁"那么无脑:`_load_sidecar` 的 doc 级预检**只给 doc_id 直读工具用**(get_document/get_outline 传 user 触发);hit 驱动路径(_assemble/expand)**刻意不传 user**([src/embedder/retrieve.py:76-83](../../src/embedder/retrieve.py#L76-L83)、[src/embedder/retrieve.py:186-197](../../src/embedder/retrieve.py#L186-L197))——因为 `_stricter` 会把命中块 idx 的 acl 在 acl_index 里改写成更严兄弟块的 acl,doc 级预检在 hit 路径会误伤本该交付的 public 命中(B2 对抗评审实证)。**fail-closed 不是"到处多拦一道"**,拦错了位置会制造 false-deny,而 false-deny 的修复冲动(放松谓词)才是未来泄漏的种子。

### 2.6 "禁出口闸仍 0 召回":如何证明防线纵深里的某一层自身有效

这一小节单独讲,因为它是这套安全设计里最有方法论价值的一步。

**防线纵深有个测试悖论:兜底会掩盖第一道防线的回归。** 假设出口复核(第三道闸)工作正常,那么无论召回层下推(第一道闸)是否正确,端到端测试都是"0 泄漏"——第一道闸悄悄坏掉(比如某次重构又把 filter 从 prefetch 挪回顶层),所有测试照样绿。R5.M2 对抗审查指出的正是这个:既有的越权断言在出口闸在场时 trivially pass,证明不了"prefetch 下推本身挡住了越权"这个真命题。

解法是**隔离测试**:[eval/acl_regression.py](../../eval/acl_regression.py) 第 5 段把出口 `acl_admits` monkeypatch 成恒 True——**故意废掉兜底**——然后以五种身份 × 四篇文档的 sentinel 原文做精确查询,断言跨租户/无权内容仍然 0 召回([eval/acl_regression.py:123-138](../../eval/acl_regression.py#L123-L138)):

```python
_store.acl_admits = lambda acl, user: True   # 废出口复核
# ... 断言 hybrid(RRF fusion)下泄漏集合仍为空 → 证明 prefetch 下推本身有效
```

整个 acl_regression 的证明结构值得完整记住,它有四层互相咬合的断言:

1. **sentinel 精确查询**:把每篇文档的独有原文当 query 去搜——哪怕精确匹配也必须 0 召回,证明"ACL 硬过滤先于相关性",而不是靠"碰巧搜不到"侥幸通过;
2. **授权正向**:有权身份必须能搜到自己该看的——防"全拒假过"(一个把所有人都拒了的系统同样能通过泄漏测试);
3. **直读面覆盖**:get_document 无权 → PermissionError、expand 跨 ACL → None;
4. **禁出口闸隔离**:如上,单独证明第一层自身有效。

同一思想在阶段 D 换 Qdrant server 时再次使用:`test_server_fusion_no_should_leak_raw` 绕过出口复核直接断言 server 的 fusion **原始输出**不含无权 point——证明"嵌入式丢 should"的坑在 server 上未复发,且不是靠出口闸掩盖。**安全断言必须测真命题:每一道防线都要有一个"其它防线不在场"的测试。**

### 2.7 服务层:三种身份模式 + fail-closed 铁律

检索层的一切都以 `User(tenant, principals)` 为输入,服务层的职责是权威地生产它([docs/DESIGN.md D10](../DESIGN.md))。三种模式:

- **keys**(团队,默认推荐):`PHAROS_KEYS_FILE` 指 JSON,每请求 `X-API-Key` 解析成 `Identity{name, tenant, principals, admin}`;未知/缺失一律 401,且不泄"key 是否存在过"([src/pharos/service.py:148-164](../../src/pharos/service.py#L148-L164));
- **legacy**:单 `PHAROS_API_KEY` 门槛,启动绑定单身份;
- **open**:都不设,仅限回环地址。

fail-closed 三级,全部在**启动期**响亮失败而不是运行期静默放行:

1. **tenant 未设** → toolcore 每个工具入口先查 `user.tenant`,空则返回 `no_identity` 空结果([src/pharos/toolcore.py:215-216](../../src/pharos/toolcore.py#L215-L216));
2. **绑非回环地址而非 keys 模式** → `create_app` 直接 SystemExit 拒绝启动([src/pharos/service.py:93-95](../../src/pharos/service.py#L93-L95))。这条叫"**部署即授权**":能连上端口的人 = 该身份的全部可见内容,所以不允许把整库以单身份/无鉴权形态裸奔到局域网;
3. **keys 文件任何格式错**(短 key、缺 name/tenant、name 重复、name 含 `|`、key 重复)→ SystemExit,绝不静默降级([src/pharos/identity.py:29-64](../../src/pharos/identity.py#L29-L64))。

keys 模式下 `_current_user` 按解析出的身份**逐请求现建**引擎 User([src/pharos/service.py:128-133](../../src/pharos/service.py#L128-L133))——身份从 HTTP 头流到 Qdrant filter 的整条链没有任何进程级共享状态,alice 和 bob 的并发请求各自带各自的 tenant 进检索栈(test_team.py 有断言身份真流到了引擎)。

### 2.8 会话隔离:一个输入校验背后的命名空间证明

per-session 去重(已交付的段落下次只回指针)是便利功能,但在多用户下它成了信息边界:A 用户取过的段落,绝不能让 B 用户被误标 `already_returned`(B 从没收到过正文)。设计是:

- 去重 **opt-in**:请求带 `X-Pharos-Session` 头才启用;不带 = 不去重(curl 一次性调用不该有跨调用状态)。MCP 适配器每进程生成一个 uuid 作会话头([src/pharos/mcp_adapter.py:26](../../src/pharos/mcp_adapter.py#L26)),stdio 下"进程=会话"天然成立;
- 登记键 = `f"{身份名}|{会话id}"`([src/pharos/service.py:204-208](../../src/pharos/service.py#L204-L208)):**不同用户即使伪造相同会话 id 也互不可见**;
- SessionRegistry 是有界 LRU(64 会话),逐出只丢去重便利、不影响正确性([src/pharos/sessions.py:16-32](../../src/pharos/sessions.py#L16-L32))。

现在回头看 identity 校验里两条看似琐碎的规则:name 禁止含 `|`、name 必须唯一([src/pharos/identity.py:51-58](../../src/pharos/identity.py#L51-L58))。它们不是风格洁癖,是**可推导的**:登记键是 `name|sid` 拼接,若 name 含 `|`,则 `("a", "b|c")` 与 `("a|b", "c")` 同键——命名空间碰撞;若 name 重复,两个不同身份共享去重命名空间——串味。面试里能从"一个输入校验"反推出"命名空间无歧义证明",是展示设计密度的好素材。

### 2.9 不泄密细则:错误、探针、日志

权限系统的最后一公里是"失败时不多说话":

- **无权与不存在同响应**:`_safe_doc_call` 把 PermissionError 映射成 `no_access`,文案统一"无访问权限**或不存在**"([src/pharos/toolcore.py:198-206](../../src/pharos/toolcore.py#L198-L206));store 层 get_by_chunk_id 同款(None 不区分两种情况)。同时注意它把 FileNotFoundError/ValueError 映射成 `config_error` 而**不是** no_access——sidecar 损坏是"可见但索引坏了",掩盖成无权会把运维问题伪装成权限问题,方向相反的两种错误绝不合并;
- **异常不裸抛给 agent**:sidecar 版本漂移的 ValueError 曾会把含绝对路径的异常文本冒泡给 agent(info_leak),修复后降级为 `single_chunk_degraded`,细节只进服务端日志([src/embedder/retrieve.py:153-156](../../src/embedder/retrieve.py#L153-L156));
- **探针信息边界**:/healthz、/readyz 未鉴权(编排/nginx 要无 key 探活),所以它们的响应体是侦察面——readyz 异常不回 `str(e)`(否则泄内网 qdrant/inference host:port),collection 缺失只回 `collection_missing` 不回集合名([src/pharos/service.py:224-256](../../src/pharos/service.py#L224-L256));
- **日志隐私**:绝不落盘 key 本体(只记身份 name),query 默认截断 120 字、可整体关闭,截断在入队前单点执行;
- **聚合信息也要 gate**:/v1/stats 在 keys 模式下 admin key 才可读——"谁在高频查什么端点"这类聚合模式本身就是信息([src/pharos/service.py:258-268](../../src/pharos/service.py#L258-L268))。

---

## 3. 为什么这么设计:被否决的备选

| 备选 | 否决理由 | 证据 |
|---|---|---|
| **召回后客户端过滤** | fail-open 窗口(过滤代码有 bug 即泄漏)+ limit 污染(top-k 被无权结果占位)。放召回层才是 fail-closed:无权内容根本不进候选 | 嵌入式 fusion 丢 should 事件本身就是"客户端兜底不可靠"的实证——最后挡住它的是 prefetch 下推,不是出口闸 |
| **顶层 must+should 平铺**(不用嵌套 Filter) | 顶层混用 must+should 的语义在引擎间/版本间不确定,"括号不可压平"是显式契约 | fusion 丢 should 实测;嵌套结构 + 下推后 acl_regression 65 断言 0 泄漏 |
| **small-to-big 用 `admit=acl_admits(user)`(取一切用户可见元素)** | seal#2:admit 路径下 `big.acl` 只能盖 hit_acl,低报 big.text 实含的更严内容,出口校验形同虚设;等价类取材让 `big.acl = hit_acl` 天然准确、出口可校验 | 真语料泄漏 3/79 → 0/79,78/79 块照常长大(召回基本不损) |
| **实现 deny 语义** | 与 chunker INTEGRATION 契约一致:要排除某组,从 allow 移除。deny 是"最毒的契约失败"候选——字段存在但静默不生效比没有更危险,直接从 schema 删掉并显式警告 | 契约层决策(见 [INTEGRATION.md](../components/chunker/INTEGRATION.md)) |
| **MappingProxyType 包 RESTRICTED_ACL 防篡改**(安全 review 的建议) | `deepcopy(mappingproxy)` 在 Py3.12 抛 TypeError,会把 fail-closed 的默认路径变成崩溃路径;已有的 deepcopy 隔离实测扛住五类污染攻击 | 对 review 建议也要对抗验证,不是照单全收 |
| **SSO/OIDC、数据库存密钥、key 热加载** | 当前团队规模不值得引入 IdP 依赖;文件+重启(秒级)足够;热加载增加状态一致性面 | [DESIGN.md D10](../DESIGN.md) 否决清单 |
| **每租户独立 collection(物理隔离)** | 单 collection + filter 下推在当前规模下已被回归证明 0 泄漏,物理隔离的运维成本(迁移/多副本/评估基线 × 租户数)不成比例;但它仍是隔离强度更高的升级路径 | 规模取舍,未一票否决 |

一句话概括设计哲学:**把安全语义收敛到最少的地方(acl.py 一个文件 + 三道闸),然后用"能变红的测试"锁死每一处**,而不是把权限检查洒满代码库。

---

## 4. 实战复盘:本轮对抗审查中与 ACL/安全相关的条目

写作前的这轮对抗审查(34 confirmed)按纪律分流:行为无关的健壮性修复直接落地,改变输出内容或需 GPU 验证的延期。安全相关条目恰好三种结局都有,各是一个教学点。

### 已修:healthz 信息收敛(fixes_applied #8)

- **症状**:`/healthz` 未鉴权可读 `collection`、`llm_model` 等字段;而同文件的 `/readyz` 因评审 sec-2 结论刻意**不回集合名、异常不回 str(e)**。
- **根因**:同一个信息边界被两个端点执行成两种标准——readyz 辛苦建立的"未鉴权探针不泄内部信息"被 healthz 直接架空。这类"不一致"比单点缺陷更值得警惕:它说明边界没有被当成一条系统性纪律,而是逐端点各自为政。
- **修法**:healthz 收敛为最小 liveness 响应 `{status, service, version, tenant_bound, uptime_s}`([src/pharos/service.py:211-222](../../src/pharos/service.py#L211-L222)),敏感字段挪进 admin-gated 的 `/v1/stats`([src/pharos/service.py:264-267](../../src/pharos/service.py#L264-L267));docs/API.md 同步。
- **测试**:healthz 响应体负向断言(不含 collection/llm_model),防回归。

同批还有一条安全相邻的 #7:stats 键此前用原始 URL 路径,**未鉴权的 401 请求也计量**——网络上任何人可用随机路径把进程内存慢性撑大(低速 DoS)。修法是改用路由模板做键、未匹配归并固定桶,键集合天然有界([src/pharos/service.py:185-191](../../src/pharos/service.py#L185-L191));`test_stats_unauthorized_requests_bounded` 锁死。

### 延期:ACL 感知路径下 big-block 系统性丢失标题(deferred chunker#0)

- **确认的事实**:heading 元素不进任何 chunk 的 source_indices,因此不在 acl_index 里;等价类门控对未知 idx fail-closed 排除 → 生产默认路径的 big.text **不含任何标题行**。同一根因 get_document 已打过补丁(own-section 可见的小节额外纳入其标题,[src/embedder/retrieve.py:212-232](../../src/embedder/retrieve.py#L212-L232)),但 assemble_big/expand 路径未同步。
- **为什么确认了却不马上改**:修复会改变 big-block 内容 → 检索交付与评估数字都会动 → 必须 bump SIDECAR_VERSION、重建索引、重跑 GPU eval 才能落地。在写作窗口内"顺手修"意味着已发布的评估基线失真。**"确认 → 排期 → 附修法草图"本身是工程判断**:fail-closed 的方向性错误(漏出而非泄入)是质量缺陷不是安全漏洞,可以等一个正规的重建窗口。
- **教学点**:这是 fail-closed 的**代价面**。方向选对了(标题被漏出,而不是受限标题被泄入),但代价真实存在——LLM 收到的 context 缺小节边界信号,climb 到父节时兄弟正文连成一片。安全默认值的质量成本要被显式管理,而不是假装不存在。

### refuted:readyz 绕过 Store._lock(service#0)

分析师报"/readyz 直接访问嵌入式 Qdrant client,绕过 Store._lock,与业务并发竞争非线程安全对象"。对抗验证**反驳**了它:嵌入式 QdrantLocal 的不安全仅源于写路径(np.append 重绑数组),而 serve 进程运行期**零写路径**(索引在独立 CLI 进程,文件锁互斥),`collection_exists` 只是进程内 dict 成员读,读+读在声明的并发模型内安全。留档为"若未来引入在线写端点需改走带锁方法"。——**对抗审查的价值不只在抓 bug,也在挡住"看起来更安全"的无效修复**;每一次不必要的加锁都是可用性税。

---

## 5. 面试怎么讲

### 30 秒版(电梯稿)

> 我在 RAG 系统里做过完整的企业级 ACL:权限过滤放在**召回层**而不是召回后,filter 下推到向量库的每个 prefetch;因为 small-to-big 会在硬过滤之后从原文重新取材,我们设计了 acl_index 等价类门控,再加所有交付路径的出口二次校验——三道闸的深防御。期间实测抓到过嵌入式 Qdrant 在 RRF fusion 下**静默丢弃 filter 的 should 子句**,ACL 退化成 fail-open,这个修复还带出一个测试方法论:把出口兜底 monkeypatch 成恒 True 做隔离测试,证明第一道防线自身有效,而不是被兜底掩盖。真语料回归:跨租户/无权内容 0 召回,small-to-big 泄漏 3/79 → 0/79 且召回基本不损。

### 3 分钟版(结构化展开)

1. **问题定性**(30s):RAG 把文件权限打碎进向量库,而相似度检索天然"偏爱"无权内容;且 RAG 有三个特有绕过面——上下文扩展、by-id 直读、错误信息泄露存在性。所以单点过滤不够,要防线纵深。
2. **分层**(30s):身份在服务层(API key → Identity,三种模式,非回环强制多身份鉴权、配置错拒绝启动),授权在检索层(User{tenant, principals} → 硬过滤)。两层正交,三个入口(HTTP/MCP/stdio)共享同一模型。
3. **三道闸**(60s):① 召回层 filter 下推每个 prefetch——这里讲嵌入式 fusion 丢 should 的 fail-open 实测与修复;② small-to-big 的 acl_index 等价类取材——"同文档≠同 ACL",数据点:真语料泄漏 3/79→0/79、78/79 块照常长大;③ 出口逐条复核,by-id 直读"无权与不存在同响应"防探测。
4. **证明方法**(45s):防线纵深的测试悖论——兜底掩盖第一层回归。acl_regression 的四层断言:sentinel 精确查询证明"过滤先于相关性"、授权正向防全拒假过、直读面覆盖、**禁出口闸隔离测试**;换 server 模式时用 raw fusion 探针重验不假设行为相同。
5. **收尾**(15s):fail-closed 是有代价的(ACL 路径 big-block 丢标题,已确认、排期修),我们显式管理这个代价而不是掩盖它——这句话通常能把面试官引向你准备好的诚实边界。

---

## 6. 追问预演

**Q1:为什么过滤一定要放召回层?召回后过滤加个大 buffer 不行吗?**
要点:三个不可修复的缺陷——① fail-open:过滤代码任何 bug 都直接泄漏,而召回层过滤下 bug 表现为"少召回"(fail-closed);② limit/buffer 是猜的:无权文档占比不可控,top-50 可能全是无权的;③ "先取回再丢"本身是越权读,审计/合规过不去。关键词:fail-open vs fail-closed、limit 污染。可补:pharos 连出口复核都有,但它的角色是兜底不是主防线。

**Q2:fusion 丢 should 这个坑怎么发现的?怎么确认是引擎行为不是你们代码错?**
要点:验证 BM25 的 Modifier.IDF 支持性时顺带跑了 hybrid 越权断言,发现同租户无权文档被召回;写**最小复现脚本**把变量隔离到"fusion 模式 × should 子句"组合(单路直查不踩、must 等值条件生效)→ 定位为嵌入式 QdrantLocal 的 fusion 实现丢顶层 query_filter 的 should。修复选择"下推到 prefetch"而不是"绕开 fusion",因为过滤必须留在召回层。关键词:最小复现、变量隔离、行为差异即安全面。

**Q3:三道闸会不会互相掩盖?你怎么知道每一道都在工作?**
要点:这正是 R5.M2 审查抓的问题——出口闸在场时,召回层的回归测试 trivially pass。解法是隔离测试:monkeypatch 出口 `acl_admits` 恒 True,断言仍 0 泄漏 → 证明 prefetch 下推自身有效;server 迁移时用 raw 探针绕过出口直接看引擎原始输出。普适原则:**每道防线配一个"其它防线不在场"的测试**。另一半是防"全拒假过":授权正向断言有权身份必须召回得到。

**Q4:small-to-big 为什么会绕过硬过滤?等价类取材为什么比"取一切用户可见元素"更安全?**
要点:big-block 按 idx 范围从原始 elements 取材,跨 chunk 边界,"同文档≠同 ACL"。admit(用户可见)方案的致命点是 big.acl 无法准确表达混合内容的有效 ACL(只能盖 hit_acl,低报更严内容),出口校验被架空;等价类(==hit_acl)让 big.acl 天然准确、出口可校验,未知 idx fail-closed。数据:3/79→0/79,78/79 照常长大。关键词:出口可校验性(verifiability)优先于取材完整性。

**Q5:"无权"和"不存在"为什么必须同响应?区分开对用户不是更友好吗?**
要点:区分即泄露存在性——攻击者可枚举 chunk_id/doc_id 拿到"库里有一份我看不到的 X"。对可信运维,细节走服务端日志;对不可信 agent/客户端,统一 no_access/None。注意反方向的例外:sidecar 损坏映射成 config_error 而非 no_access——把运维故障伪装成权限问题会把人引向错误的排障路径,"不泄密"不等于"所有错误都装作无权"。

**Q6:为什么不用 SSO/OIDC、不做 RBAC、不做数据库行级权限?**
要点:按规模取舍,且留了升级路径。身份层是可替换的薄层(keys 文件 → 未来换 IdP 不动检索层,因为分层正交);授权模型是 tenant + principals∩allow + public,够覆盖"部门/项目组"粒度;拒绝的是运维成本(IdP 依赖、热加载的状态一致性),不是拒绝概念。反问自己那句:如果明天要接 SSO,改哪几行?答案是 identity 模块换实现,`_current_user` 契约不变——说明分层是真的。

**Q7:多副本/换存储后端时,这套 ACL 怎么保证不回归?**
要点:① ACL 语义单点(acl.py)+ 双实现同构单测;② acl_regression 是端到端回归(真 Chunker+Embedder 灌 2 租户合成库,五身份矩阵),换后端后重跑;③ 对新后端不做行为假设——server 模式专门加 raw fusion 探针;④ 会话去重这类跨副本状态被显式声明为"便利不是正确性",轮询下降级到 1/N 是可接受的(正确性不依赖它)。

**Q8:你这套设计现在已知的弱点是什么?**(顺接下一节)

---

## 7. 动手实验

### Lab 1(CPU):亲手复现"嵌入式 fusion 丢 should"的 fail-open

前置:仓库已 `pip install -e .`(需要 qdrant-client、jieba;WSL 下 `conda activate navikb`)。原理见 §2.2;脚本用 :memory: 库、随机 dense 向量(不需要 GPU/模型),五个 point 覆盖 public/受限/跨租户/unset 四类 ACL:

```bash
python - <<'PY'
import random
from qdrant_client import models
from embedder.config import EmbedConfig
from embedder.sparse import doc_sparse, query_sparse
from embedder.store import Store
from embedder.types import User
DIM = 8; _v = lambda: [random.random() for _ in range(DIM)]
A = lambda t, a, vis, u=False: {"tenant": t, "allow": a, "visibility": vis, "unset": u}
def pt(i, text, acl):
    sp = {"acl_tenant": acl["tenant"], "acl_allow": acl["allow"],
          "acl_visibility": acl["visibility"], "acl_unset": acl.get("unset", False)}
    return models.PointStruct(id=i, vector={"dense": _v(), "sparse": doc_sparse(text)},
        payload={"chunk_id": f"c{i}", "doc_id": f"d{i}", "kind": "text", "text": text, "acl": acl, **sp})
s = Store(EmbedConfig(qdrant_path=":memory:", dense_dim=DIM, collection="t", prefetch_limit=20))
s.ensure_collection()
s.upsert([pt(1, "公开 营收", A("t1", [], "public")),
          pt(2, "hr 营收",  A("t1", ["g_hr"], "restricted")),
          pt(3, "fin 营收", A("t1", ["g_fin"], "restricted")),
          pt(4, "t2 营收",  A("t2", ["g_hr"], "restricted")),
          pt(5, "unset 营收", A("t1", [], "restricted", True))])
acl = s.acl_filter(User("t1", ["g_hr"])); qs = query_sparse("营收")
leak = s.client.query_points("t",
    prefetch=[models.Prefetch(query=_v(), using="dense", limit=20),
              models.Prefetch(query=qs, using="sparse", limit=20)],
    query=models.FusionQuery(fusion=models.Fusion.RRF),
    query_filter=acl, limit=10, with_payload=True).points
print("只顶层filter(不下推):", sorted(p.payload["chunk_id"] for p in leak))
ok = s.client.query_points("t",
    prefetch=[models.Prefetch(query=_v(), using="dense", filter=acl, limit=20),
              models.Prefetch(query=qs, using="sparse", filter=acl, limit=20)],
    query=models.FusionQuery(fusion=models.Fusion.RRF),
    query_filter=acl, limit=10, with_payload=True).points
print("filter下推每个prefetch:", sorted(p.payload["chunk_id"] for p in ok))
PY
```

**预期**:第一行(只靠顶层 query_filter)会混入 `c3`——同租户但用户不在 allow 组的文档被召回,嵌套 should 被 fusion 静默丢弃、只剩 must(tenant/unset)生效,即 fail-open;第二行(下推 prefetch)只剩 `c1/c2`,0 泄漏。你亲眼看到的就是 [store.py:103-105](../../src/embedder/store.py#L103-L105) 注释描述的坑,以及为什么顶层 filter 只能当双保险。

### Lab 2(CPU):用 pytest 走一遍三道闸的守护测试

```bash
# ① ACL 语义双实现同构 + fail-closed 拆解
python -m pytest -q tests/engine/test_acl.py -v
# ② small-to-big 三态:等价类挡住跨 ACL 兄弟 / legacy 路径"必泄漏"契约锁 / admit 必须配 acl_index
python -m pytest -q tests/engine/test_core.py -k "cross_acl or legacy_path or admit_requires" -v
# ③ store 层:硬过滤 / by-id 复核(无权与不存在同 None)/ list_documents 作用域
python -m pytest -q tests/engine/test_store.py -k "acl" -v
# ④ 服务层:keys 校验(含 '|'/重名拒启)/ 伪造 session id 跨用户隔离 / 日志不落 key
python -m pytest -q tests/test_team.py -v
```

重点读 `test_assemble_big_legacy_path_is_unguarded`(用测试锁死已知漏洞形态)和 `test_keys_mode_session_isolated_across_users`(两用户伪造相同 session id 互不可见,验证 `身份名|` 前缀)。全程无 GPU、无网络。

### Lab 3(GPU/WSL):端到端 ACL 回归,含"禁出口闸"隔离段

前置:WSL + `conda activate navikb`(需要 Qwen3-VL 模型与 4090,脚本会现建 2 租户合成库、真向量灌入):

```bash
conda activate navikb && python eval/acl_regression.py
```

**预期**:65 断言全过,退出码 0。重点看第 5 段输出——`[禁出口闸]` 前缀的断言在 `acl_admits` 被打成恒 True 后仍 0 泄漏,这就是 §2.6 讲的"证明第一道防线自身有效"。

---

## 8. 诚实边界

面试中主动承认这些,比被问出来强得多:

1. **ACL 感知路径下 big-block 丢失全部标题**(已确认、延期修)。fail-closed 方向正确(漏出而非泄入),但 LLM 收到的 context 缺小节边界信号,只有 breadcrumb 部分补偿;get_document 已修,assemble_big/expand 待同一窗口(bump SIDECAR_VERSION + 重建 + GPU eval)统一落地。话术:"这是 fail-closed 的显式代价,我们选择先记账排期、不在评估基线外偷改输出。"
2. **权限模型是粗粒度的**:tenant + 组∩allow + public,无 deny、无字段级/行级权限、无时间窗授权。deny 是刻意不做(静默不生效比没有更毒),但"用户被移出组后已缓存的会话"这类撤销时效问题没有专门机制——keys 文件轮换靠重启,秒级但非实时。
3. **嵌入式模式的 scroll/should 行为未穷尽验证**:list_documents 对 scroll 路径的嵌套 should 是"不赌 + 逐条复核"策略,即靠出口闸而非验证过的第一道防线——这是已声明的保守处理,不是证明过的等价。
4. **查询向量缓存是用户无关的**(ACL 在召回后才施加),这是安全的;但意味着"用户 A 的查询热身了用户 B 的缓存"——计时侧信道意义上的信息量极小,未做专门评估。
5. **威胁模型有边界**:防的是"合法接入者的越权读"与"配置不完整的静默裸奔",不防拿到服务器 shell 的攻击者(sidecar JSON 与 Qdrant payload 都是明文)、不防投毒的解析产物(仅 img_path 做了路径净化)、密钥文件安全依赖文件系统权限(chmod 600 尽力而为,Windows 上更弱)。
6. **禁出口闸测试跑在 GPU 回归里而非 CI**:acl_regression 需要真向量,CPU CI 覆盖的是各层单测;"每次提交都重证第一道防线"目前做不到,靠变更纪律(动 store/acl 必跑)兜底。

---

*锚点行号核验于 2026-07-07(对抗审查修复落地后的当前代码);若后续重构导致行号漂移,以链接文件内同名符号为准。*
