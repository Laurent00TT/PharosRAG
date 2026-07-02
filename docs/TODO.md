# Pharos 规划(TODO)

## v0.1(本版)—— 已完成

- [x] 引擎重构:mcp_server 拆 toolcore(transport 无关工具语义层,引擎 `7fbf709`)
- [x] FastAPI 守护进程:独占 Qdrant + GPU,/healthz + /v1/ask + 6 检索端点
- [x] MCP 薄适配器(零 GPU,毫秒启动,结构化降级),.mcp.json 接 Claude Code
- [x] per-session 去重(X-Pharos-Session,有界 LRU;兑现引擎 B5.A 预告)
- [x] 可选 API key 门槛;ACL 身份启动绑定 fail-closed
- [x] CLI:serve / mcp / index / ask / health
- [x] `pharos index`(index_real 参数化:corpus/dest/collection/ACL/only/limit)
- [x] 25 项 CPU 单测 + 引擎回归 + GPU 冒烟(全绿,见 TESTING.md)
- [x] 全套文档(README/DESIGN/IMPLEMENTATION/API/TESTING/TODO/COMPONENT_NOTES)
- [x] 对抗评审 + confirmed 修复

## v0.2 候选(按优先级)

- **P1 systemd/自启**:WSL 里给 `pharos serve` 配 systemd unit(或 Windows 任务计划唤 WSL),
  开机即有;现在要手动起。
- [x] ~~P1 /v1/ask 支持检索过滤~~ **已落地(2026-07-02)**:Generator.answer 可选 kwargs 按需透传,
  `pharos ask --kind table` 等;动机与实证见 COMPONENT_NOTES N3(Netflix 营收案例)。
- [x] ~~分部数字被错引申成总体~~ **已修(2026-07-03)**:SYSTEM 窄靶数值范围约束 + source 行并入
  小节面包屑(根因=约束与范围证据双缺,详见 TESTING §3);同裁判 72 题回归零损失(忠实度 +0.028)。
- [x] ~~P2 表格块检索文本增强~~ **已落地(2026-07-03,引擎 2bd97a5)**:面包屑+表头+行标签进
  检索文本(存在性门控保 chunk id 稳定);两索引已重建;动机案例中文 kind=table 直接答对。
  回归取舍与完整实录见 TESTING §3。
- **P2 gold 集补表格题**(增强回归暴露的测量盲区):现 72 题 gold 采样自散文块,表格可答题型
  近乎为零——表格相关改动在该 gold 上只会显示成本(冗余 gold 位移)不显示收益(动机案例类)。
  做法:gen_gold 对 kind=table 且 content_raw 非空的块定向采样 10-15 题(问表内数值/对比),
  与现 gold 合并;之后表格向改动才有对称的验收标尺。
- **P2 观测**:请求日志落盘(query/耗时/status 计数),/healthz 加 model_loaded 与索引统计。
- **P2 解析编排**:`pharos parse <pdf|docx|xlsx…>` 调 MinerU(在线 API tokens 已有)→ 直接
  ingest 新文档,不再依赖引擎仓预解析产物。
- **P3 LLM 有界重试**(COMPONENT_NOTES N6):先观察 ask_failed 实际频率。
- **P3 会话内并发去重竞态**(IMPLEMENTATION §3 观察项):单会话并发工具调用时 returned_keys
  交错;agent 串行调用下无实害,出现实害再改(per-session 锁)。

## v2 方向(规模驱动,不预支)

- Qdrant server 模式(多客户端解锁,EmbedConfig 换 url 即迁移)
- 多身份:per-API-key principal 映射(单实例多用户)——当前"多身份=多实例"够用
- SSE/streaming ask;简单 Web UI

## 明确不做

- 公网暴露/HTTPS 终结(个人内网系统;要远程就 tailscale 之类隧道,不在应用层做)
- 跨文档综合难题的检索增强(评估显示 multi_cross 是研究性问题,引擎仓已定论不追)
