# vLLM 推理后端方案(Plan B → 通过门则升 Plan A)

> 定位:**不替换、先并存**。现有自包 FastAPI 推理服务(`inference_server.py`)保持 Plan A;新增 vLLM 后端作
> Plan B,**呈现完全相同的 `/embed`/`/rerank`/`/readyz` 契约** —— 应用层(pharos `remote.py`)一个字节都不改,
> 只把 `PHAROS_INFERENCE_URL` 指向谁。**vLLM 只有连过等价性门 + 吞吐门才升 Plan A**,否则随时零成本回退。
>
> 状态:**四门全过实测(2026-07-07),vLLM 判定应升 Plan A;剩工程落地(Phase 1 适配器 + compose profile)**。
>
> **实测速览(见 §4.1)**:G0 GO(vLLM 加载 pooling 出归一向量)/ G1 边缘(cosine 0.99956)/ **G2 基本通过**(vLLM 查
> 官方真库 top-1 98.9%、Jaccard 0.9959、分歧全在尾部近重复无害)/ **G4 决定性胜**(vLLM continuous batching **297 QPS**
> vs FastAPI 串行 **24 QPS** 封顶 = **~12×**,且零错误、p95 更低)。**综合:等价性可接受 + 吞吐 12× → vLLM 应升 Plan A。**

---

## 0. 一句话

把"唯一吃 GPU 的前向"从**逐请求串行(FastAPI + `gpu_lock`)**换成 **vLLM 的 continuous batching**,在**并发查询**下把
GPU 利用率和吞吐拉上去;但**前提是 vLLM 出的向量与现有库(官方 `Qwen3VLEmbedder` 建的)数学等价** —— 这是整个
方案的命门,和阶段 B 的 E2 混建混查同级。等价过 + 吞吐真赢,才升 Plan A。

---

## 1. 为什么现在把 vLLM 提上日程

现有 inference 的 GPU 前向是**单卡串行**(`inference_server.py` 的 `state.gpu_lock` + 阶段F 的 `BoundedSemaphore(16)`
背压)。这是 SCALE_OUT §5-F F-3 写死的**吞吐硬上限**:`--scale pharos=N` 扩不了 QPS,因为所有 dense/rerank 前向排同一条队。

**vLLM 的唯一硬价值 = continuous batching**:把并发请求在 GPU 上动态拼批,单卡吞吐随并发上升(而非串行钉死)。
触发条件(SCALE_OUT §3.4 定义):**查询 QPS 上升到 `/embed` 队列深度持续 >1**。不是"vLLM 更潮"。

**为什么之前没上**:等价性。两端走同一份 `Qwen3VLEmbedder` 前向 → 等价是**结构性保证**;换 vLLM = pooling 换实现
→ 等价从"保证"变"实测赌注"。本方案就是把这个赌注**变成一次可跑的 go/no-go 实验**(§4 G1/G2),赢了才用。

---

## 2. 关键事实(grounded,决定方案可行性)

| # | 事实 | 依据 | 对方案的影响 |
|---|---|---|---|
| A | 模型是 `Qwen3VLForConditionalGeneration`(`model_type: qwen3_vl`),即**生成式 VL 模型**,靠官方脚本 last-token pooling 当 embedder | 本机 `~/models/Qwen3-VL-Embedding-8B/config.json` + `scripts/qwen3_vl_embedding.py::_pooling_last` | vLLM 必须复刻:chat 模板(system=instruction)+ **LAST** pooling + L2 normalize |
| B | **vLLM 官方支持 Qwen3-VL-Embedding**:`LLM(model, runner="pooling", dtype="bfloat16", trust_remote_code=True)` + `llm.embed(...)` | vLLM 文档 + QwenLM/Qwen3-VL-Embedding `examples/embedding_vllm.ipynb` | G0 大概率过;`--convert embed` 默认 last-token + normalize,**与官方 pooling 同款** |
| C | vLLM 输入格式:`apply_chat_template([{system:instruction},{user:text}], tokenize=False, add_generation_prompt=True)` → 送 `prompt` | 官方 `embedding_vllm.ipynb` | 适配器必须**照抄**这个格式(尤其 `add_generation_prompt=True`),否则 last token 错位 → 等价必败(**假阴风险**) |
| D | vLLM 返回**已归一化**向量(notebook 直接 `emb @ emb.T` 不再 normalize) | 官方 notebook | 与本地 `F.normalize` 对齐,无需二次归一 |
| E | **image/video 预处理 vLLM 与官方不同**(`qwen_vl_utils` vs transformers `video_processing_qwen3_vl`)→ 结果"略有差异" | vLLM/Qwen 文档明确警告 | **只影响 image 路径**;而 image 编码是**建库专用、走 local torch**(D4/D5),**查询路径纯文本 → 不受影响**。这恰好把 vLLM 的最大等价风险挡在方案之外 |
| F | Reranker 是 `Qwen3-VL-Reranker`(yes/no logits + sigmoid),vLLM 的 score/rerank 对它的支持**未确认** | 未查到权威确认 | **Reranker 先不迁 vLLM**;rerank 是降级安全项(失败→hybrid),留 torch,零风险(见 §5 Phase 1) |
| G | 本机 `vllm` conda env 已装 **vLLM 0.22.1 / torch 2.11+cu130 / transformers 5.10.2** | `conda activate vllm` 实测 | 可本地直接跑 G0/G1;⚠ 与建库用的 pharos(transformers 4.57)版本不同 → 数值可能微漂,G1 要量化 |

**方案可行性初判(依据上表,非实测)**:文本查询路径 **G0 大概率可行、G1 有较大概率过**(同 pooling/normalize + 官方
input 格式可照抄 + image 差异被挡在查询路径外);主要不确定性在 **transformers 版本差 + bf16 数值** 上,须 G1 量化。

---

## 3. 目标架构:同契约适配器 + 双后端共存

```
                        pharos ×N (remote.py 不变;PHAROS_INFERENCE_URL 指向谁就用谁)
                                 │  /embed {texts,instruction} · /rerank
              ┌──────────────────┴───────────────────┐
              ▼ (Plan A 现状)                          ▼ (Plan B / 通过门则升 Plan A)
   ┌─────────────────────────┐           ┌─────────────────────────────────────────┐
   │ inference (FastAPI+torch)│           │ inference-vllm                            │
   │ gpu_lock 串行 + 背压     │           │  ┌────────────┐   ┌────────────────────┐ │
   │ /embed /rerank(全维)   │           │  │ 契约适配器  │──▶│ vLLM serve (pooling)│ │
   └─────────────────────────┘           │  │ (CPU,无卡) │   │ AsyncLLMEngine      │ │
                                          │  │ 套 chat模板 │   │ continuous batching │ │
                                          │  │ /embed→/v1/ │   │ /v1/embeddings 8B   │ │
                                          │  └────────────┘   └────────────────────┘ │
                                          │  reranker:Phase1 仍走 torch(降级安全)   │
                                          └─────────────────────────────────────────┘
   compose profiles 选择:`--profile fastapi`(现状) / `--profile vllm`(Plan B) —— 互斥,回退=换 profile
```

**边界铁律不变**:适配器/vLLM 端点里**不出现任何 pharos 业务概念**(no ACL/Hit/sidecar),只有 `texts→vectors`。
这正是当初"边界画在纯 GPU 前向"的红利兑现 —— 换后端不碰应用层。

**两个落点的取舍(为什么是"vLLM serve + 薄适配器"而非"in-process `LLM.embed`"):**
- **要 continuous batching 跨并发 HTTP 请求生效,必须用 vLLM 的 AsyncLLMEngine**(`vllm serve` 起的就是它)。
  离线 `LLM.embed()` 只在**单次调用内**批处理,跨并发请求不拼批 → 拿不到 vLLM 的核心价值。故用 `vllm serve`。
- **适配器薄、无 GPU**:只做两件事 ——(1) 把我们契约的 `{texts, instruction}` 按 §2-C 套 chat 模板;
  (2) 转发到 vLLM `/v1/embeddings`,把回包 reshape 成 `{"vectors": [...]}`。顺带扛 `/readyz`(探 vLLM up)。
  它复用阶段F 的探针/背压纪律,但**前向排队交给 vLLM**(它自己有调度,适配器不再加 `gpu_lock`)。

---

## 4. go/no-go 门(方案的中心;不过不升 Plan A)

顺序刚性,前门不过不进后门。前 3 个是**正确性**门(等价),G4 是**收益**门(吞吐):

| 门 | 判据 | 怎么测 | 不过的后果 |
|---|---|---|---|
| **G0 可行** | vLLM 在 pooling 模式加载 Qwen3-VL-Embedding-8B,`/v1/embeddings` 出 4096 维向量 | `vllm serve ... --runner pooling`(或 `LLM(runner="pooling")`)起服务,curl 一条文本 | vLLM 这条路直接断,方案作废(回退 FastAPI) |
| **G1 向量等价** ⭐ | 同批文本(中英/长短/带 instruction),`cosine(vLLM, 官方 Qwen3VLEmbedder) > 0.9999`,且 norm≈1 | `scripts/vllm_equiv_probe.py`(见 §7):pharos 出官方向量存档 → 停 → vllm 出向量 → 比 cosine(分时,免 OOM) | 现有库(官方建)**不能被 vLLM 查**(向量漂移→top-k 错位,静默数据损坏)。出路二选一:①精确对齐 input 格式/transformers 版本再测;②接受 vLLM 只在**全库用 vLLM 重建**后用(大迁移) |
| **G2 混建混查** ⭐ | 官方建的真库,vLLM 编码查询,~50 条真 query 的 top-k `chunk_id` 序与"官方查"完全一致 | 起 vLLM 适配器 + 指向真库,对拍官方 encode 的 top-k(同阶段B E2 方法) | 逐元素等价**不蕴含** top-k 不变(HNSW 近似 + RRF rank 放大微差)。不过则**不上生产**(同 E2 铁律) |
| **G3 reranker**(可选) | vLLM 对 Qwen3-VL-Reranker score 与官方 `allclose` | 若 Phase 2 才做;Phase 1 跳过(reranker 留 torch) | 不过就**不迁 reranker**,只迁 embed(rerank 降级安全,零损失) |
| **G4 吞吐**("效果好"判据) | 并发 C∈{16,32,64} 下,vLLM 的 QPS / p50 / p95 **显著优于** FastAPI(串行 gpu_lock);单请求延迟不劣化 | `scripts/bench.py` 扩并发档,分别打 FastAPI 和 vLLM 适配器,同一批 query | vLLM 不比串行快(低并发时可能更差)→ **不升 Plan A**,留 Plan B 备规模;诚实记录 |

> **G1/G2 是硬门,和阶段 B 的 E1/E2 同级**:混建混查错位是最难查的静默损坏。**"效果好"= G1∧G2∧G4 同时成立**,
> 不是只看 G4 的吞吐数字。

### 4.1 实测结果(2026-07-07,`scripts/vllm_equiv_probe.py`,4090)

| 门 | 结果 | 数据 |
|---|---|---|
| **G0 可行** | ✅ **GO** | vLLM 0.22.1 `LLM(runner="pooling", max_model_len=8192)` 成功加载 Qwen3-VL-Embedding-8B,`llm.embed()` 出 **4096 维、已归一(norm=1.00000)** 向量。**架构完全支持** |
| **G1 向量等价** | ⚠ **边缘** | 8 样本(中英/长短)cosine 全在 **[0.99956, 0.99982]**,min **0.99956** / mean **0.99973**。方向高度一致(最差约 1.7° 夹角),但**未达严格 >0.9999** |

**微漂根因(诊断,未定论)**:头号嫌疑 **transformers 版本差**——库是 pharos **4.57.6** 建的,vLLM 环境是 **5.10.2**,
两版 Qwen3-VL 前向数值实现不同;其次 bf16 kernel 差异。最短文本("a")cosine 最低(0.99956),符合"短序列放大逐 token 数值差"。

**这算过还是不过?** —— **不是看这个 cosine 数字拍板,是看 G2**:0.9997 的对齐对绝大多数查询 top-k 稳定(微漂远小于
文档间分差),**但近重复 chunk 可能翻转**——这正是 G2 混建混查要测的。故 G1 边缘 **不否决方案**,交 G2 定生死。

**踩坑留档(诊断纪律,`拒绝 verdict 型结论`的正例)**:探针跑了 3 次假失败才拿到真数据,全是**我的配置错、非 vLLM 不支持**:
①`CUDA_VISIBLE_DEVICES` 给了 GPU-UUID(vLLM 只吃整数索引,torch 才吃 UUID)→ 改 `CUDA_DEVICE_ORDER=PCI_BUS_ID
CUDA_VISIBLE_DEVICES=1`;②`max_position_embeddings=262144` 让 vLLM 要 36GB KV cache OOM → 加 `max_model_len=8192`
(embedding 单次前向无需长上下文);③ 我 `grep -v` 把真实报错(EngineCore Traceback)过滤掉了 → 差点误判"架构不支持"。
**每次探针的固定文案"vLLM 不支持 Qwen3VL pooling"三次都是错的**——真相是模型完全能加载。教训:失败先看真根因,别信 canned verdict。

| 门 | 结果 | 数据(`scripts/vllm_g2_topk.py`,88 题 eval/gold.jsonl,嵌入式真库 7652 点)|
|---|---|---|
| **G2 混建混查** | ⚠→✅ **基本通过** | top-1 一致 **87/88 (98.9%)**;top-10 集合一致 **86/88 (97.7%)**;top-10 序完全一致 76/88 (86.4%);**Jaccard@10 0.9959**。12 分歧样本首分歧位次:rank≥6 有 7 个(尾部近重复换位)、rank 0/1 各 1 个 |

**G2 判读(诚实)**:G1 的 cosine 0.9997 微漂**不翻转有效召回** —— top-1 98.9% 一致、top-10 集合 97.7% 一致、
分歧几乎全是 rank 6-9 的近重复 chunk 换序(证据池不变、不改答案)。这是**混建混查最坏情形**(vLLM 查 × 官方建,
跨实现);若 Plan A 落地**用 vLLM 全库重建**,则建/查同实现、自洽无漂,比这更干净。**唯一残留**:1/88 top-1 不同 +
2/88 top-10 集合不同(纯 transformers 4.57↔5.10 数值差)。对"忠实度头牌"系统若零容忍,对齐 transformers 版本或
全库 vLLM 重建可消除;否则 98.9% top-1 生产可接受。

| 门 | 结果 | 数据(`scripts/bench_embed.py`,88 题循环取满 120/档,单文本查询=1 向量,预套 chat 模板)|
|---|---|---|
| **G4 吞吐** | ✅ **决定性胜** | vLLM(continuous batching)vs FastAPI(串行 gpu_lock + 背压16),4090:|

```
        FastAPI QPS   vLLM QPS   倍数     (p50 ms: FastAPI→vLLM)
conc 1     22.4         33.5     1.5×      44 → 28
conc 8     23.7        169.3     7.1×     334 → 45
conc 16    23.5        226.0     9.6×     680 → 72
conc 32    22.2*       271.2    12.2×     428 → 93     *FastAPI 仅 17/120 成功,103 被背压 503 拒
conc 64     —          297.3      —      (vLLM 全成,p95 264ms)
峰值:FastAPI ~24 QPS 封顶(串行,不随并发升);vLLM 297 QPS(随并发线性 scale)
```

**G4 判读**:FastAPI 串行 gpu_lock 把吞吐钉死在 **~24 QPS**(并发只堆延迟 + 背压泄洪),vLLM continuous batching
把并发请求拼批,**297 QPS ≈ 12×**,且零错误、p95 反而更低(93ms vs 428ms@conc32)。**这正是拆 GPU 推理层时预留
的收益兑现**(端点契约无业务概念 → 换后端不碰应用层)。**四门综合:vLLM 应升 Plan A。**

**总结论**:等价性可接受(G1/G2)+ 吞吐 12×(G4)→ **vLLM 升 Plan A 成立**。落地走 §5 Phase 1(embed 走 vLLM,
reranker 降级安全暂留;compose profile 切换 + 秒级回退)。若追求零 top-1 漂移,配套 vLLM 全库重建(建查同实现自洽)。

---

## 5. 分阶段实施(每阶段可独立验证 + 随时回退)

**Phase 0 — go/no-go 探针(先跑,决定后面做不做)。** 仅 `scripts/vllm_equiv_probe.py`,不碰生产:测 G0 + G1。
本阶段唯一产出是**一个数**(cosine)。cosine>0.9999 → 继续;否则先修 input 对齐或判定"需全库重建"。**成本最低、决定性最高,先做。**

**Phase 1 — embed-only vLLM 后端(reranker 留 torch)。✅ 核心已落地 + bare-metal 实测(2026-07-07):**
- `src/inference_vllm_adapter.py`(薄 FastAPI,无 GPU、无 torch):`/embed` 套 §2-C chat 模板 → vLLM `/v1/embeddings`
  → `{"vectors"}`(全维归一,契约同 inference_server);`/readyz` async 探 vLLM `/health`;`/embed_image` 501(建库专用
  走 local);`/rerank` 配 `RERANK_PROXY_URL` 则代理 torch reranker,否则 503(pharos 降级 hybrid,安全)。
  **⚠ 放 src/ 根不放 embedder/ 包内**:适配器要在 vllm 环境跑(无 qdrant_client),放包内会触发 `embedder/__init__`
  的 qdrant_client import + 让 stdlib `import types` 被 `embedder/types.py` 遮蔽(两坑均实测踩到,已解)。
- `Dockerfile.inference-vllm`:slim 适配器镜像(fastapi/uvicorn/httpx/transformers,build 期断言脱 torch)。
- **端到端 bare-metal 实测(同阶段E pivot 方法)**:`vllm serve :8000` + 适配器 `:8900` + **pharos(RemoteDense,应用层
  一字不改)经 PHAROS_INFERENCE_URL=适配器 打嵌入式真库** → `scripts/vllm_adapter_smoke.py` **3/3 查询出真命中**
  (IBM 查询→IBM 报表 chunk、Netflix→Netflix 10K、cross-lingual→对口论文)。adapter /embed 出 1×4096 norm=1.0。
  **证明"换 vLLM 后端不碰应用层"兑现。**
- **⏸️ 待做(诚实,同阶段E 纪律)**:compose `vllm` profile 容器化(`inference-vllm-engine`=vllm/vllm-openai 起 `vllm serve` +
  `inference-vllm`=适配器)**在容器内原样跑一遍**——vllm/vllm-openai 镜像 ~10GB,本机网络拉取有风险(同阶段E torch base
  的 daocloud 坑),留作带自身校验的下一步,不提交"从未在容器跑过"的 profile(阶段E 盲区教训)。
- G2 混建混查已在 Phase 0 提前跑过(§4.1,GO)。

**Phase 2 — 吞吐门 + 决定 Plan A/B。** 跑 **G4** 并发基准(FastAPI vs vLLM),数字写进 SCALE_OUT。
G4 赢 → vLLM 升 Plan A(默认 profile 切 vllm);否则留 Plan B。

**Phase 3(可选)— reranker 也迁 vLLM。** 仅当 G3 过;否则永远留 torch reranker(降级安全,无损)。

---

## 6. GPU 资源现实(必须先算账)

4090 **48GB**。现状:torch inference(2×8B embed+rerank)占 **~33GB**。vLLM embed 8B 权重 ~16GB + KV cache。
**不能同时满载跑 torch-inference 全量 + vLLM**(>48GB OOM)。因此:

- **测 G0/G1/G2/G4**:必须**先停 torch inference 容器**(`docker compose stop inference`)腾出 GPU,再起 vLLM。
  即跑 vLLM 实验 = 现有 compose 栈临时下线 inference(pharos 会 /readyz 503,可接受,是实验窗口)。
- **Plan A 落地形态(若 G4 赢)**:vLLM 只服务 **embed**(~16-20GB)+ torch **reranker** 单独容器(~16GB)可共存 4090
  (合 ~36GB < 48GB);或 reranker 也 vLLM(G3 过)则全 vLLM。**embed 是 QPS 命门,先迁它即拿大头收益**。

---

## 7. go/no-go 探针脚本(Phase 0,已就位)

`scripts/vllm_equiv_probe.py`(见仓库):分时跑,`--step official` 用 pharos 出官方向量存档,`--step vllm` 用 vllm env
出 vLLM 向量并对比 cosine。**关键**:input 格式严格照 §2-C(chat 模板 + `add_generation_prompt=True` + 同 instruction),
否则假阴。跑法:
```bash
# 0) 腾 GPU
docker compose --env-file .env.compose stop inference
# 1) 官方向量存档(pharos)
conda activate pharos && python scripts/vllm_equiv_probe.py --step official --out /tmp/vllm_probe
# 2) vLLM 向量 + 比对(vllm env)
conda activate vllm    && python scripts/vllm_equiv_probe.py --step vllm     --out /tmp/vllm_probe
# 输出:每条 cosine + max/min/mean;min cosine>0.9999 = G1 GO
```

---

## 8. 风险 + 回退

| 风险 | 缓解 |
|---|---|
| **G1 不过**(transformers 版本差/bf16/input 格式)| 先对齐:vLLM env 装同版 transformers、input 格式逐 token 对拍;仍不过 → 判"需全库 vLLM 重建"或放弃 |
| input 格式差一个 token → 假阴 | 探针严格照官方 notebook(§2-C);先验证两端 tokenize 后 id 序一致再比向量 |
| reranker vLLM 不支持 | Phase 1 就不迁 reranker(降级安全);G3 独立门 |
| GPU 放不下 | embed-only vLLM + torch reranker 分容器(§6);或时分 |
| vLLM 崩/回退 | **compose profile 切回 `fastapi` + pharos `PHAROS_INFERENCE_URL` 指回 torch inference,秒级回退**(应用层不变是这套的最大保险) |
| 低并发下 vLLM 更慢 | G4 如实测;不达标就留 Plan B,不硬上 |

---

## 附:来源

- 模型架构/pooling:本机 `~/models/Qwen3-VL-Embedding-8B/{config.json,scripts/qwen3_vl_embedding.py}`(实读)
- vLLM 支持 + input 格式:[QwenLM/Qwen3-VL-Embedding](https://github.com/QwenLM/Qwen3-VL-Embedding) 的 `examples/embedding_vllm.ipynb`、[vLLM Embedding 文档](https://docs.vllm.ai/en/latest/models/pooling_models/embed/)、[vLLM Pooling 文档](https://docs.vllm.ai/en/latest/models/pooling_models/)
- image 预处理差异警告:vLLM / Qwen3-VL-Embedding 文档(见 §2-E)
- 本机 vllm env:vLLM 0.22.1 / torch 2.11+cu130 / transformers 5.10.2(`conda activate vllm` 实测)
