# generator

RAG 的生成层(RAG 的 **"G"**):检索 → LLM 答案合成,带 **引用溯源 + ACL 出口 + grounding 防幻觉**。
`parse → chunk → embed → retrieve → **generate**`。

## 设计:LLM 无关 + 依赖注入

- **`LLMClient` 协议**:`complete(messages) -> str`(OpenAI-compatible `messages`)。可插拔——本地 Qwen/vLLM、各家 API、Claude 都能实现这一个方法接进来。
- **`Generator(retriever, llm)`**:编排 检索 → prompt → LLM → 解析引用。retriever 由 embedder 提供(`Retriever`)。
- 核心**零运行时依赖**(纯 stdlib);`MockLLM` 供离线/无真模型时验证,`OpenAICompatibleLLM` 提供 OpenAI-compatible API 真后端(DeepSeek V4 Flash 实测)。

## 三条关键不变量

- **grounding 防幻觉**:prompt 强制"只用提供的 context、无依据就说信息不足、禁外部知识",且每个论断标 `[cite:n]`(不引入额外模型)。
- **引用溯源**:context 按 1-based 编号喂入,答案里的 `[cite:n]` 解析回 chunk(doc_id/title/section/page);**越界丢弃**,不映射错来源。引用标记用 **`[cite:n]` 而非裸 `[n]`**——检索正文常含脚注/参考文献的 `[n]`,共用会被 LLM 照抄、误映射成错来源(**溯源造假**),`[cite:n]` 与正文 token 隔离(对抗 review 挖出的 RAG 攻击面)。
- **ACL 出口**:Generator **只消费 `retriever.search_with_context` 的返回**(hits 已过 Qdrant ACL 硬过滤、context 已过出口校验),不引入新内容 → 喂进 LLM 的每段都是 user 有权看到的。`context=None`(sidecar 丢/损)时降级用命中块 text(单 chunk 同样已授权),不丢答案。

## 模块

| 模块 | 职责 |
|---|---|
| `types.py` | `Message` / `Citation` / `Answer` |
| `llm.py` | `LLMClient` 协议 + `MockLLM` + `OpenAICompatibleLLM`(真 API 后端:thinking 按 base_url 门控 / finish_reason 透出) |
| `prompt.py` | `PromptBuilder`(grounding 指令 + 编号引用) |
| `generate.py` | `Generator`(编排 + 引用解析 + ACL 出口) |

## 用法

```python
from embedder import EmbedConfig, Retriever, User
from generator import Generator, MockLLM

ret = Retriever(EmbedConfig())
gen = Generator(ret, MockLLM())          # 真 LLM:把 MockLLM 换成实现了 complete(messages) 的客户端
ans = gen.answer("台积电的资本开支规划?", User(tenant="t1", principals=["g_research"]), rerank=True)
print(ans.text)                          # 带 [n] 引用的答案
for c in ans.citations:
    print(f"[{c.marker}] {c.title} / {c.section} (doc={c.doc_id})")
```

## 状态

生成层完成并过 **R3 对抗评审(修 6)+ ③ 表格/图表 grounding 修复**(资产命中补回 `content_raw`,见 `generate.py`)。单测 **16 passed**
(prompt/generate,含 ACL 出口降级、grounding 退路、越界引用丢弃、passage `[cite:n]` 注入中和、短资产数据总补回)。真 LLM 后端
`OpenAICompatibleLLM` 已接入;端到端 RAG 评估已闭环(72 gold / 异厂 Claude 裁判,**忠实度 ≈1.0**、正确性 0.847,详见 [OVERVIEW §7](../../OVERVIEW.md))。

## 未决(非阻塞)

- **跨文档综合弱**:multi_cross 正确性 0.00(n=5)—— 拿到两篇块也合不出对比,是综合难题不是检索,边际收益低未追(见 [OVERVIEW §7](../../OVERVIEW.md))。
- (真 LLM 后端 / 端到端评估已完成,见「状态」。)
