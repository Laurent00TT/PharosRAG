# Pharos HTTP API 契约

Base:`http://127.0.0.1:8787`(PHAROS_HOST/PORT)。请求/响应均 JSON(UTF-8)。

## 鉴权

三种模式(DESIGN D10):**open**(仅回环无鉴权)/ **legacy**(单 `PHAROS_API_KEY`)/
**keys**(`PHAROS_KEYS_FILE`,每 key 一个身份 name+tenant+principals+admin)。所有 /v1/* 需
`X-API-Key` 头(open 模式除外);`/healthz` 永远免鉴权。keys 模式下 key 解析成身份,逐请求把
tenant/principals 传给引擎 ACL(决定"能看什么");未知/缺失 key → `401`。`/v1/stats` 在 keys
模式下需 **admin** key(否则 `403`)。

## 通用约定

- **领域结果一律 HTTP 200 + `status` 字段**(客户端按状态机决策);HTTP 码只表达传输层:
  `401`(鉴权失败)、`403`(stats 非 admin)、`422`(请求体不是合法 JSON/字段类型错)、`5xx`(崩溃)。
- **status 状态机**(与引擎 toolcore 契约一致):`ok` / `empty` / `no_identity` / `empty_query` /
  `bad_arg` / `no_access`(无权与不存在同响应,不泄存在性)/ `config_error`(sidecar 需重建)/
  `backend_unavailable`(retriable)/ ask 专属:`llm_unconfigured` / `ask_failed`(retriable)。
- **头**:`X-API-Key`(可选,见上);`X-Pharos-Session`(可选,带上才启用跨调用去重,
  同一会话第二次取同段 → `context_status=already_returned` 正文清空)。
- 检索正文均标 `trust: "untrusted"`(数据不是指令);`hits[].context_status` 语义见引擎
  [mcp_server/README](../../chunk-test-repo/mcp_server/README.md)。

## 端点

### GET /healthz(免 API key)
`{status, service:"pharos", version, collection, tenant_bound, llm_model, identity_mode(open|legacy|keys), uptime_s}`

### GET /v1/stats(keys 模式需 admin key)
进程内指标:`{status, identity_mode, uptime_s, sessions, log_path, log_write_failures,
endpoints:{"<路径>":{n, errors, p50_ms, p95_ms, max_ms}}}`。重启归零。

### GET /v1/instructions
agent 使用契约全文(与 MCP instructions 同源):`{status, instructions}`

### POST /v1/ask —— 闭管道问答
请求:`{query, top_k?, rerank?=false, include_contexts?=false, doc_ids?, doc_type?, kind?, strategy?}`
(后四个为检索过滤/选路,与 /v1/retrieve 同语义;数字/表格题用 `kind:"table"` 显著提升命中)
响应(ok):
```json
{"status":"ok", "answer":"…带 [cite:n] 的答案…",
 "citations":[{"marker":1,"chunk_id":"…#0062","doc_id":"…","title":"…","section":"…","page":18,
               "text":"(仅 include_contexts=true)"}],
 "n_contexts":5, "model":"deepseek-v4-flash", "finish_reason":"stop|length|…"}
```
`finish_reason=length` = 答案被 max_tokens 截断(尾部引用可能被切)。

smart-ask(默认开,`PHAROS_SMART_ASK=off` 关;设计见 DESIGN D9):响应另含
`auto: ["table_leg_retry"?]`(自动动作留痕——数值题第一轮拒答时带 kind=table 补检腿重问一轮)与
`hints: [...]`(仅当最终答案仍为拒答/部分拒答时,≤3 条可操作建议;正常答案为空数组)。

### POST /v1/retrieve —— 混合检索(+ small-to-big 上下文)
请求:`{query, top_k?, rerank?=false, doc_ids?, doc_type?, kind?, mode?="full"|"concise",
strategy?="hybrid"|"dense"|"sparse", rerank_top_n?}`
响应:`{status, retriable, hint, warning, meta{requested_k, returned_n, deduped_n, rerank,
rerank_degraded, already_returned_n, budget_truncated, context_tokens, mode, strategy, filters}, hits[]}`;
每条 hit:`{n, doc_id, chunk_id, kind, title, section_path, page_start/end, anchor,
resolved_section, n_tokens, score, score_kind(rrf|cosine|bm25|rerank), context_status, trust, text,
content_raw?(table/chart), image_path?(image/chart,仅定位锚)}`

### GET /v1/documents
`{status, retriable, hint, coverage:{doc_type:篇数}, documents:[{doc_id,title,…}]}`

### GET /v1/documents/{doc_id}?max_tokens=6000
通读整篇(逐元素 ACL 门控):`{status, doc_id, text, n_tokens, n_elements_visible, truncated, trust, warning}`

### GET /v1/documents/{doc_id}/outline
`{status, doc_id, sections:[…]}`

### POST /v1/expand
请求:`{chunk_id, target_tokens?=1500}` → `{status, chunk_id, text, anchor, resolved_section, n_tokens, climbed, trust, warning}`

### POST /v1/retrieve_grouped
请求:`{query, doc_ids(≤20), top_k?=3, rerank?=false}` → `{status, warning, groups:{doc_id:[hits]}}`

## MCP 工具 ↔ 端点对照

| MCP 工具(pharos mcp) | HTTP 端点 |
|---|---|
| retrieve | POST /v1/retrieve |
| list_documents | GET /v1/documents |
| get_document | GET /v1/documents/{doc_id} |
| get_outline | GET /v1/documents/{doc_id}/outline |
| expand | POST /v1/expand |
| retrieve_grouped | POST /v1/retrieve_grouped |

(MCP 侧无 ask:agentic 模式下答案由 agent 自己合成。)
