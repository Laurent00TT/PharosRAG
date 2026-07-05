# generator 设计

> RAG 的生成层(G):**LLM 无关脚手架**(prompt 组装 / 引用 / ACL 出口 / grounding,LLM 接口可插拔)
> + 真后端 **`OpenAICompatibleLLM`**(DeepSeek V4 Flash via OpenAI-compat,已端到端实测)。

## 1. 链路

```
query + user
  → Retriever.search_with_context(rerank?)        # embedder:hybrid(+rerank)+ ACL 硬过滤 + small-to-big + 出口校验
  → contexts[](每段已授权,带来源元数据)
  → PromptBuilder.build                            # 编号引用 + grounding 系统指令
  → LLMClient.complete(messages)                   # 可插拔(MockLLM / 真 LLM)
  → 解析答案里的 [n] → Citation[]                   # 映射回 chunk 来源,越界丢弃
  → Answer(text, citations, n_contexts)
```

## 2. LLM 无关脚手架

- **`LLMClient` 协议**(`llm.py`):只一个 `complete(messages: list[Message]) -> str`。messages 用 OpenAI-compatible
  `[{role, content}]` —— 本地 vLLM/Qwen、OpenAI/Claude API 都兼容,真后端实现这一个方法即可插入。
- **依赖注入**:`Generator(retriever, llm)` 不 import embedder/任何 LLM SDK(duck typing),核心纯 stdlib、零运行时依赖。
- **`MockLLM`**:从 prompt 的 context 编号回显一个带 `[n]` 的答案,无真模型即可验证整条脚手架(单测用)。
- **`OpenAICompatibleLLM`**(真后端):任何 OpenAI 兼容端点通吃——DeepSeek / GLM 代理 / 本地 vLLM,**型号/端点/思考开关全走配置,不写死**(换后端=改 base_url+model)。三个评估旋钮 `model`(flash/pro)、`thinking`(on/off)、`reasoning_effort`(high/max);思考链走 `reasoning_content` 与答案 `content` 分离(不污染 `[cite:n]`),默认 thinking off(读-合成任务非重推理,且省延迟/成本);key 从 `.env` 读勿提交。需 `pip install openai`。

## 3. 三条不变量(设计要点)

| 不变量 | 怎么做 | 为什么 |
|---|---|---|
| **grounding 防幻觉** | system prompt 强制"只用 context、无依据说信息不足、禁外部知识"+ 每论断标 `[cite:n]` + **数值范围约束**(分部/子期间数字不得引申为总体);context 的 source 行带**小节面包屑**给约束供范围证据 | 不引入额外模型,靠约束;LLM 跑偏时退到"信息不足"而非编造。实测教训:**约束没有证据就是空文**——分部信息只在 section_path 里,不喂进 prompt 模型无从判断(修复实录见 pharos TESTING §3) |
| **引用溯源** | context 1-based 编号;答案 `[cite:n]` 正则解析回 chunk(doc/title/section/page);**`[cite:n]` 与正文裸 `[n]` 隔离** | 答案可核查来源(企业 RAG 刚需);隔离防溯源造假 |
| **ACL 出口** | 只消费 `search_with_context` 返回(已硬过滤+出口校验);不引入新内容 | 喂 LLM 的每段都已授权;ACL 在检索层守住,生成层不破坏 |

## 4. 决策与取舍

- **OpenAI-compatible messages**:最通用的 LLM 接口形态,本地与各家 API 通吃,可插拔成本最低(`messages_to_dicts` helper 给真后端复用转换)。
- **grounding 靠 prompt 不靠模型**:不引入 NLI/事实校验模型;简单、零额外依赖,代价是依赖 LLM 遵循指令(真 LLM 接入后需实测幻觉率)。
- **引用标记 `[cite:n]` 而非裸 `[n]`(对抗 review#1,RAG 攻击面)**:检索正文常含脚注/参考文献 `[n]`,共用标记会被 LLM 照抄、解析误映射成错来源(溯源造假),甚至恶意 chunk 主动塞 `[1]` 把读者引向攻击者选定来源。`[cite:n]` 在自然文本几乎不出现,与正文 token 隔离。
- **越界 `[cite:n]` 丢弃而非报错**:LLM 可能幻觉出 `[cite:99]`;丢弃比映射错来源安全,比崩溃稳。
- **ACL 出口防御纵深(可选 `acl_check`)**:默认信任 retriever 已硬过滤;但 Generator 解耦设计可复用到别的 retriever,注入 `acl_check(acl,user)` 后对每个命中块二次 fail-closed 校验,不假设 retriever 一定安全。
- **`context=None` 降级用命中块 text**:sidecar 丢/损时不丢答案(命中块单 chunk 已授权);small-to-big 上下文缺失但有命中内容。
- **独立包(非 embedder/generate.py)**:关注点分离——embedder=检索,generator=生成。Generator 通过依赖注入接 Retriever,两包解耦。

## 5. 状态 + 待办

- ✅ 生成层完成 + 单测 **16 passed** + 端到端(真 Retriever + DeepSeek/MockLLM 全链路,ACL 贯穿)。封板 5 类全修(引用 token 污染→`[cite:n]` 隔离等);**R3 对抗评审再修 6**(passage `[cite:n]` 注入中和、finish_reason 透出、thinking 按后端门控);**③ 表格/图表 grounding**(资产命中补回 content_raw)。
- ✅ **真 LLM 后端**:`OpenAICompatibleLLM`(DeepSeek V4 Flash)。smoke test 真调用双向实测——in-context 给数字+正确 `[cite:1]`;out-of-context 答"信息不足"不瞎编(`examples/smoke_deepseek.py`,免 GPU/检索)。选 API 而非本地 LLM:DeepSeek 极廉(整套评估几分钱)+ 1M context + OpenAI 兼容,且把 4090 留给 embedding 16G + reranker 16G。
- ✅ **端到端 RAG 问答评估已闭环**(`eval/`,72 gold / 异厂 Claude 裁判,不让 generator 自评):**忠实度 ≈1.0**(R5 订正——早先 0.83 是裁判 context 被 CTX_CAP 截断的 eval bug)、正确性 single 0.847,已定论 closed pipeline 为默认。详见 [OVERVIEW §7](../../OVERVIEW.md)。
