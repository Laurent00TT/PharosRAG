# Pharos 运维手册

> 面向:运维本服务的人(可能不是开发者)。所有容量/RTO 数字均为真实实测,非估算。
> 架构与设计动机见 [DESIGN.md](DESIGN.md)(服务面 = D10-D12)。

## 1. 部署拓扑

```
Windows 登录 ─► 启动文件夹 PharosWSL.vbs(静默保活 WSL,防空闲休眠)
                    └► WSL(Ubuntu)systemd ─► pharos.service(Restart=always)
                            └► pharos serve @ 127.0.0.1:8787(独占 ~/rag_real + GPU)
消费方:pharos ask / curl /(每人自己的)MCP 薄适配器 —— 全部经 HTTP + X-API-Key
```

## 2. 日常命令

| 动作 | 命令 |
|---|---|
| 看状态 | `systemctl status pharos` |
| 看日志(服务) | `journalctl -u pharos -f` |
| 看日志(请求) | `tail -f ~/pharos_logs/requests.jsonl` |
| 重启 | `sudo systemctl restart pharos`(秒级;首次检索后模型才热) |
| 健康检查 | `curl http://127.0.0.1:8787/healthz`(免鉴权) |
| 指标 | `curl -H "X-API-Key: <admin key>" http://127.0.0.1:8787/v1/stats` |
| 停(建库/备份前必须) | `sudo systemctl stop pharos`(嵌入式 Qdrant 单客户端锁) |

## 3. 身份与密钥(D10)

- **发一个新身份**:`python -m pharos keys new <名字> --tenant demo [--principals g_a,g_b] [--admin]`
  → key 只打印一次,安全转交;然后 `sudo systemctl restart pharos` 生效。
- **吊销**:编辑 `PHAROS_KEYS_FILE` 删掉该条目 + restart。**轮换** = 吊销 + 重发。
- keys 文件(默认 `~/pharos.keys.json`):chmod 600、**绝不进 git**;格式错(含重名 name、name 含 `|`)
  服务会拒绝启动(响亮,不静默降级)。⚠ `keys new` 自动 chmod 600 **仅在 WSL/Linux 有效**;若在原生
  Windows 侧放置该文件,POSIX 权限位无效,需自行用 NTFS ACL 收紧(本服务部署在 WSL,默认路径 `~` 即 WSL home,无此问题)。
- **团队成员接入 MCP**(Claude Code):各自的 `.mcp.json` 里给适配器传 `PHAROS_API_KEY=<自己的 key>`
  与 `PHAROS_URL=http://<服务机>:8787`。
- **暴露到局域网**:`PHAROS_HOST=0.0.0.0` —— 服务会**强制要求 keys 模式**,否则拒绝启动
  (部署即授权,不允许整库裸奔)。内网信任边界 + key;HTTPS 明确非目标(要远程走隧道)。

## 4. 容量(实测,2026-07-04,4090 / 77 篇 / 7652 chunk)

| 并发客户端 | 检索 p50 | 检索 p95 | 吞吐 | 错误 |
|---|---|---|---|---|
| 1 | 408ms | 528ms | 2.5 req/s | 0 |
| 2 | 659ms | 750ms | 2.9 req/s | 0 |
| 5 | 1.57s | 1.82s | 3.0 req/s | 0 |
| 10 | 3.07s | 3.26s | 3.2 req/s | 0 |
| 3(含 20% ask) | 检索 352ms / ask 4.1s | — | 2.3 req/s | 0 |

**解读**:检索段全局串行(嵌入式 Qdrant + GPU 前向),吞吐天花板 ~3.2 req/s,排队随并发**线性**、
零错误、可预测;**ask 的 LLM 段不持锁**——混合负载下 ask 在飞时检索 p50 仍 352ms(D8 设计实锤)。
**容量判断:≤10 人同时活跃体验可用(p50 ≤3s);更大规模 → v2 迁 Qdrant server 模式。**
复测:`python scripts/bench.py --key <key> --clients 5 --n 10`。

## 5. 备份 / 恢复(演练实录 2026-07-04)

**要备份的全部状态**:`~/rag_real`(qdrant+sidecar)+ `~/pharos.keys.json` + **两个 `.env`**
(pharos 仓 + 引擎仓——`.gitignore` 忽略,git 恢复不回来,漏了它服务会 fail-closed 空跑 / ask 无 LLM key)
+ 两仓 git(代码/文档)。请求日志(~/pharos_logs)按需。

```bash
sudo systemctl stop pharos                       # 单客户端锁:必须先停
mkdir -p ~/backups                               # 新机上 ~/backups 未必存在,先建
tar czf ~/backups/pharos_$(date +%F).tar.gz \
    -C ~ rag_real pharos.keys.json \
    -C /mnt/c/Users/11541/Desktop/projects/pharos .env \
    -C /mnt/c/Users/11541/Desktop/projects/chunk-test-repo .env
sudo systemctl start pharos
```

**实测(2026-07-04)**:备份 27MB / 3 秒;恢复(解包→备用目录→新实例拉起→77 篇可见 + 检索验证)
**RTO = 33 秒**。⚠ 该 33 秒是**热模型**下测得(演练紧接主服务停机、dense 模型仍在显存);
**冷启动**(重启机器/换机)首次检索会触发模型 lazy load(§7),真实 RTO 到"可服务" ≈ 33s + 20s~2min。
灾难兜底:备份全丢也可从 `parsed/` 语料重建(`pharos index`,77 篇约 15-30 分钟,keys 文件需重发)。
建议:cron 每周一备 + 保留 4 份;**每季度跑一次恢复演练**(照上面步骤,别让 runbook 变纸面)。

## 6. 观测(D11)

- **请求日志** `~/pharos_logs/requests.jsonl`,每行:`{ts, ep, user(身份名,绝不含 key), http,
  ms, query(截 120 字,PHAROS_LOG_QUERIES=off 可关), status/n/auto/n_citations/refusal}`。
  单文件追加;体积大了用 logrotate 或直接 mv 走(服务不持句柄常开)。
- **/v1/stats**(keys 模式 admin-only):每端点 n/errors/p50/p95/max、uptime、会话数、日志写失败计数。
  重启归零(设计内)。
- 值得盯的信号:`ask` 的 `refusal=true` 占比升高(语料覆盖不足或检索退化)、`errors` 非零、
  `log_write_failures` 非零(磁盘问题)。

## 7. 故障排查

| 症状 | 原因 | 处置 |
|---|---|---|
| 服务起不来,日志见 "already accessed" | 别的进程占着索引(手动 serve/建库脚本) | 找到并停掉;永远只让 systemd 实例碰 ~/rag_real |
| 服务起不来,日志见 keys 文件报错 | keys JSON 格式/字段错(fail-closed 设计) | 修文件或 `pharos keys new` 重建;别绕过 |
| 全部请求 401 | keys 模式下 key 不在文件里 / 改文件没 restart | 核对文件 + `systemctl restart pharos` |
| 某用户看不到任何文档 | 该身份 tenant 与建库 tenant 不符(fail-closed) | 核对 keys 文件里的 tenant(示例库建库 tenant=demo) |
| 首个检索 20s-2min | dense 模型 lazy 加载 | 正常;重启后第一查慢,之后毫秒级 |
| 一段时间后服务消失 | WSL 空闲休眠(无客户端连接) | 确认启动文件夹 PharosWSL.vbs 在;临时:开个 wsl 窗口 |
| ask 全部 llm_unconfigured | DEEPSEEK_API_KEY 缺失/过期 | 补 .env(pharos 或引擎仓)+ restart |
| GPU OOM | eval/建库与服务同时跑 rerank | 压测/eval 前 `systemctl stop pharos`(见 TESTING) |
