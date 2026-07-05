# Pharos 路线图

已交付能力见 [README](../README.md) 与各设计/实测文档;本文只记**往哪走**与**明确不做**。
(已落地功能的动机/负结果/实测留档:检索智能见 [TESTING §3](TESTING.md) + [COMPONENT_NOTES](COMPONENT_NOTES.md);
团队服务面见 [DESIGN D10-D12](DESIGN.md) + [OPERATIONS](OPERATIONS.md)。)

## 候选(按优先级)

- **P1 stats 持久化 + logrotate**:`/v1/stats` 现为进程内、重启归零;请求日志单文件。团队长期运行需
  指标落盘(或接 Prometheus)+ 日志滚动。先观察实际增长速率再决定重量级。
- **P1 密钥吊销审计**:keys 文件编辑 + restart 即吊销,但无"谁在何时被吊销"的审计线索。加一条
  吊销日志 + `pharos keys list/revoke` 子命令。
- **P2 per-key 速率限制**:当前无限流,吞吐天花板 ~3.2 req/s 下单个重度用户可饿死他人。按 key
  令牌桶,超限返回结构化 `rate_limited`。
- **P2 解析编排**:`pharos parse <pdf|docx|xlsx…>` 调 MinerU(在线 API tokens 已有)→ 直接 ingest
  新文档,让"加一篇文档"成为一条命令。解析现已在本仓(`scripts/parse_batch.py`/`parse_office.py`/
  `mineru_client.py`,见 [scripts/README](../scripts/README.md));`pharos parse` 子命令化仍待做。
- **P2 表格向检索**:表格题当前 检索 0.750 / 正确 0.688(88 题基线);4 个检索 miss + 2 个大表读数错
  是对称标尺(TESTING §3)。候选:表格块 embed 增强、跨语言查询辅助。
- **P3 LLM 有界重试**(COMPONENT_NOTES N6):DeepSeek 偶发 5xx 现打成 `ask_failed`;观察实际频率后
  决定是否加服务端有界重试(权衡:重试放客户端语义更清晰)。
- **P3 会话内并发去重竞态**(IMPLEMENTATION §3 观察项):单会话并发工具调用时 returned_keys 交错;
  串行调用下无实害,出现实害再加 per-session 锁。

## v2 方向(规模驱动,不预支)

- **Qdrant server 模式**:嵌入式单客户端锁是水平扩展/多副本/滚动升级的天花板;迁 server 模式后解锁
  (EmbedConfig 换 url 即迁移)。这是超出小团队规模后的第一块骨牌。
- **多副本 + 负载均衡**:守护进程当前单点;Qdrant server 化后才谈得上多 pharos 副本。
- SSE/streaming ask;简单 Web UI。

## 明确不做

- **HTTPS/公网终结**:内网信任边界 + API key;要远程走 tailscale 之类隧道,不在应用层做 TLS。
- **SSO/OIDC**:当前规模不值得引入 IdP 依赖;keys 文件 + 重启的轮换足够。
- **跨文档综合难题的检索增强**:评估显示 multi_cross 是研究性问题(仓内 eval 已定论),不追。
