# NaviKB on a single RTX 4090 48G — step-by-step deployment journal

A reproducible, all-online (search **and** ingest resident simultaneously)
deployment of the full NaviKB model stack on **one RTX 4090 48G**, on
**Windows 11 + WSL2**, behind a **China network with a Clash TUN proxy**.

This is the real journey: each step lists what to do, **why**, and the
gotcha it cost (so you don't pay it again). Validated end-to-end on
2026-06-07: ingest of a 17-page PDF → 115 chunks across all 4 channels,
and `/search_docs` returning RRF-ranked results — with every model online.

> Companion files in this directory: `README.md` (terse reference),
> `start_everything.sh` (one-shot bring-up), `_status.sh`, `stop_all.sh`,
> and all `*_start_*.sh` / setup scripts referenced below.

---

## 0. Target & hardware

| | |
|---|---|
| GPU (serving) | **RTX 4090 48G** (`nvidia-smi` index **1**) |
| GPU (display) | RTX 5070 Laptop 8G (index **0**) — *not* used for serving |
| OS | Windows 11 + WSL2 (Ubuntu 24.04) |
| Constraint | all models online; search + ingest concurrent (user hard req) |
| Network | Clash Verge, TUN + fake-ip (China) |

The 4090 is Ada (sm89) → **hardware FP8 supported**, which is what makes the
"3× 8B VL models on 48G" plan possible. Reference budgets came from the
sibling `knowledge-base/docs/48gb-*.md` (written for a remote 96G card; we
adapted: remote→local, SSH tunnel→WSL localhost, BF16→FP8).

---

## 1. Verify the WSL2 GPU foundation FIRST

```powershell
wsl -d Ubuntu -- bash -lc 'nvidia-smi --query-gpu=index,name,memory.total --format=csv; ls /usr/lib/wsl/lib/libcuda.so'
```
Expect both GPUs visible and `libcuda.so` present. **Gotcha:** torch's default
device order is *fastest-first* (4090→cuda:0), but `nvidia-smi`/PCI order puts
the 4090 at **index 1**. Every serving script therefore pins:
`CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1`. Without this vLLM grabs
`cuda:0` = the 8G 5070 and OOMs instantly.

---

## 2. Miniconda in $HOME (no sudo)

WSL user has no passwordless sudo, and vLLM/mineru need conflicting
torch/transformers → **isolated conda envs**, installed under `~/miniconda3`
(no system packages touched). Script: `01_setup_vllm_env.sh`.

**Gotcha (conda 26 ToS):** `conda create -c conda-forge` resolves the channel
*name* to `conda.anaconda.org`, which (a) hangs through the TUN and (b) hits a
Terms-of-Service gate. Fix: use the **full tuna URL** as the channel and
disable notices:
```bash
conda config --set number_channel_notices 0
conda create -y -n vllm --override-channels \
  -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge python=3.12
```

---

## 3. The China-network reality (read before any download)

Diagnosed empirically (`_netcheck.sh`):

| Channel | From WSL (direct) | Verdict |
|---|---|---|
| tuna (pip/conda) | ✅ | use for all pip/conda |
| modelscope | ✅ | use for all models that have a MS mirror |
| aliyun pip | ❌ timeout | avoid |
| hf-mirror.com file CDN | ❌ timeout | unreachable from WSL |
| huggingface.co | ❌ timeout | unreachable |

**HuggingFace is only reachable via the Windows-side Clash proxy:**
```powershell
curl.exe -x http://127.0.0.1:7897 -L -o <file> https://hf-mirror.com/<repo>/resolve/main/<file>
```
Why this and nothing else: direct connections are intercepted by the TUN
(fake-ip) and time out; `Invoke-WebRequest`/.NET HttpClient on **Windows
PowerShell 5.1 don't follow HTTP 308** (which hf-mirror returns); `curl.exe`
*direct* fails the schannel TLS handshake. Only `curl.exe` **through the proxy**
(CONNECT tunnel + end-to-end TLS + `-L`) works. `download_hf_repo.ps1` and
`download_milco_curl.ps1` automate this with `-C -` resume + per-file retries
(the link is flaky; retries are mandatory).

> To find the Clash controller (to toggle `allow-lan` etc.): the configured
> `external-controller` port is often remapped — the real one is a port owned
> by the mihomo core process (here `127.0.0.1:39798`, secret `set-your-secret`).
> `allow-lan` did **not** help WSL reach the proxy (Windows firewall blocks the
> vEthernet inbound), so we download HF on the Windows side instead.

---

## 4. vLLM env + the Triton C-compiler fix

`bash 01_setup_vllm_env.sh` → `vllm 0.22.1 / torch 2.11.0+cu130 /
transformers 5.10.2`. Verify torch sees the 4090 (sm89) and the inductor
canary (`import torch._inductor.select_algorithm`) imports — on this 4090 it
does, so the `knowledge-base` "duplicate template" patch was **not** needed.

**Gotcha (no compiler):** first `vllm serve` crashes in a Triton JIT kernel
with `Failed to find C compiler`. WSL has no gcc. Fix without sudo:
```bash
conda install -y -n vllm -c <tuna-conda-forge-url> gcc gxx
bash _setup_cc.sh    # symlinks gcc/cc/g++/c++ -> x86_64-conda-linux-gnu-gcc
```
(conda-forge only ships the prefixed compiler name; Triton looks for plain `gcc`/`cc`.)

---

## 5. Models & the FP8 decision

Download (`02_download_models.sh bulk`, all via modelscope — reliable):
`Qwen/Qwen3-VL-Embedding-8B` (BF16), `Qwen/Qwen3-VL-Reranker-8B` (BF16),
`Qwen/Qwen3-VL-8B-Instruct-FP8` (official FP8), MinerU2.5. Into
`~/navikb-serving/models/`.

**FP8 = vLLM `--quantization fp8` on the BF16 model at load — NOT offline
llmcompressor.** We tried `llmcompressor` offline first; `AutoModel.from_pretrained`
loaded the embedding model with **random weights** ("Some weights … newly
initialized" for the entire model) → garbage FP8. vLLM's load-time dynamic FP8
is correct, simpler, and avoids a whole quant env + dependency conflict
(llmcompressor pins transformers<5, vLLM wants 5.x). The `quant` env and
`quantize_*.py` here are **abandoned dead-ends, kept only as a record.**

---

## 6. Bring up the 3 vLLM models (order + budget tuning)

Start **sequentially** (never profile rerank+gen concurrently — they fight for
VRAM and you get negative-KV). Use `_wait.sh <port>` between each.

```
embed (8003, util 0.26)  →  rerank (8005, util 0.28)  →  gen (8006, util 0.28)
```

- **embed** (`_start_embed.sh`): `--runner pooling --quantization fp8
  --chat-template <model>/chat_template.jinja` (vision channel 400s without the
  template). ~11.5G.
- **rerank** (`_start_rerank.sh`): pooling + `--quantization fp8` + the 3
  must-haves (`--chat-template`, `classifier_from_token ["no","yes"]`,
  `is_original_qwen3_reranker:true`). util 0.24 was too low (negative KV) →
  **0.28**. ~12.5G.
- **gen** (`_start_gen.sh`): the hard one. Generation hits flashinfer JIT paths
  that need **nvcc** (absent), and `vllm_flash_attn` isn't installed. Fixes:
  ```bash
  export VLLM_USE_FLASHINFER_SAMPLER=0      # native sampler, no JIT
  export VLLM_ATTENTION_BACKEND=TORCH_SDPA  # pure-PyTorch attn, no flashinfer/nvcc
  # drop --kv-cache-dtype fp8 (that path forces flashinfer)
  # bound the vision profile or profile_run OOMs (negative KV):
  --max-num-seqs 2 --max-num-batched-tokens 8192 \
  --mm-processor-kwargs '{"max_pixels": 401408, "min_pixels": 50176}' --enforce-eager
  ```
  util 0.28 ≈ 13G. Three models resident ≈ **40G / 49G**, ~9G free for mineru.

Budget lesson: gpu-memory-utilization is per-process fraction of **total**;
the real killer is the **vision profile_run** activation (default max-num-seqs
× big image). Bound it, don't just raise util.

---

## 7. sparse (MILCO) — CPU + offline sub-tokenizers

MILCO runs on **CPU** (0 VRAM) — `_start_sparse.sh` sets `SPARSE_DEVICE=cpu`
**and** `CUDA_VISIBLE_DEVICES=""` (milco.py hardcodes `self.to("cuda")` and
ignores the device setting otherwise).

**The MILCO hidden dependency:** at encode time `milco.py` loads two *sub*
tokenizers by HF id — `naver/splade-v3` (BERT vocab) and
`BAAI/bge-m3-unsupervised` (XLM-R vocab). Their vocabularies define the sparse
indices, so they can't be substituted. Fix (offline):
1. Download splade-v3 tokenizer via `download_hf_repo.ps1 naver/splade-v3` → WSL.
2. bge-m3 tokenizer already ships *inside* MILCO's own dir.
3. `_sparse_fix.sh` rewrites `models/milco-650m/config.json` so
   `lsr_encoder_checkpoint` → local splade dir and
   `multilingual_encoder_checkpoint` → MILCO's own dir; start with
   `HF_HUB_OFFLINE=1`.

MILCO itself is HF-only → fetched with `download_milco_curl.ps1` (curl-proxy +
resume). Verify: `/embed_query` returns a sparse vector (nnz≈50).

---

## 8. mineru — its own env, pipeline mode

mineru pins old transformers/torch → **its own conda env** (`03_setup_mineru_env.sh`,
`mineru[pipeline,vlm]`). `_start_mineru.sh` pins the 4090.

Gotchas:
- `ModuleNotFoundError: six` → `pip install six` into the mineru env (missing dep).
- **`hybrid` parse mode deadlocks in WSL** (multiprocessing.spawn) — a tiny PDF
  hangs >3 min. Set `MINERU_PARSE_MODE=pipeline` (navikb sends the mode per
  request). This was the difference between hang and success.
- First `/parse` downloads mineru's ~2.15G model and can exceed navikb's httpx
  client timeout — but the model stays **resident** in the server, so a retry
  succeeds (`_ingest_retry.sh`). mineru self-manages its model cache; the
  `mineru-vlm` dir from the bulk download is unused (deletable).

---

## 9. Qdrant — server mode (required)

all-online ⇒ worker and tool_server write concurrently ⇒ **Qdrant local-file
mode (single-writer) is impossible**. Run the server:
```powershell
# download the linux binary via the proxy, then in WSL:
tar -xzf qdrant.tar.gz -C ~/navikb-serving/qdrant && bash _start_qdrant.sh
```
`.env`: `QDRANT_URL=http://localhost:6333`.

---

## 10. Control plane (navikb env) + the ext4 rule

`04_setup_navikb_env.sh` → `navikb` conda env (torch 2.8 CPU, transformers
4.57.6), editable install of the repo with `[server]`.

**The ext4 rule:** the control plane's SQLite coordination (server↔worker) must
**not** live on `/mnt/c` (9p locking is unreliable → "database is locked",
which breaks the all-online design). Runtime dir = `~/navikb-serving/runtime/`
on ext4, holding `.env` (from `env.local-48g`) + data dirs. serve/worker `cd`
there. **Gotcha:** SQLite won't create parent dirs — `mkdir -p qdrant_data
storage/images storage/source_docs logs` first, or you get "unable to open
database file".

`.env` extras for this topology: `INGEST_USE_GPU_MINERU=false` (use the
persistent mineru server on 8101 instead of per-PDF spawn),
`MINERU_PARSE_MODE=pipeline`, `EMBEDDING_VERSION=...v02`.

Create the admin (serve refuses to start with an empty users table):
```bash
bash _create_admin.sh   # prints the API key ONCE
```

Then `_start_serve.sh` (8000) and `_start_worker.sh`.

---

## 11. End-to-end validation

```bash
bash _ingest.sh /mnt/c/path/to/file.pdf          # mineru -> 4 channels -> Qdrant
# search (embed + sparse + rerank + HyDE/multi-query -> RRF):
TOKEN=kb_admin_...
curl -s http://localhost:8000/search_docs -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' -d '{"query":"...","top_k":5}'
```
Success looks like: `… N pages, M chunks stored` and `vision_pages_…v02` /
`desc_pages_…v02` collections populated in Qdrant; search returns ranked hits.

---

## 12. Run / status / stop / persist

```bash
bash start_everything.sh   # detached full-stack bring-up (after reboot)
bash _status.sh            # health of all 8 services + 4090 VRAM
bash stop_all.sh           # stop everything
```
**Persistence:** services run under whatever launched them. For unattended
operation, run `start_everything.sh` from a kept-open WSL terminal, or wire it
into WSL systemd / a Windows scheduled task (`wsl -d Ubuntu -- bash …/start_everything.sh`).

---

## 13. Gotcha index (root-cause → fix)

| Symptom | Root cause | Fix |
|---|---|---|
| vllm uses 8G GPU / OOM | torch fastest-first puts 4090 at cuda:0 | `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1` |
| `conda create` hangs / ToS error | `-c conda-forge` → anaconda.org via TUN | full tuna URL + `--override-channels` |
| HF downloads all fail | TUN fake-ip + PS5.1 no-308 + schannel | `curl.exe -x http://127.0.0.1:7897 -L` + `-C -` + retries |
| `Failed to find C compiler` | no gcc for Triton JIT | `conda install gcc gxx` + symlink `gcc/cc/g++` |
| FP8 model = garbage embeddings | offline `AutoModel` loads random weights | vLLM `--quantization fp8` at load |
| `Could not find nvcc` (gen) | flashinfer sampler+attention JIT | `VLLM_USE_FLASHINFER_SAMPLER=0` + `VLLM_ATTENTION_BACKEND=TORCH_SDPA` |
| negative KV cache | vision profile_run too big / util too low | bound `--max-num-seqs` + `max_pixels`; raise util |
| sparse uses GPU / 500 | milco.py forces cuda; loads HF sub-tokenizers | `CUDA_VISIBLE_DEVICES=""`; rewrite config to local tokenizer paths + `HF_HUB_OFFLINE=1` |
| mineru `No module named six` | missing dep | `pip install six` in mineru env |
| mineru parse hangs forever | hybrid spawn deadlock in WSL | `MINERU_PARSE_MODE=pipeline` |
| mineru first parse times out | model downloads during parse | retry; model stays resident |
| "database is locked" / "unable to open db" | SQLite on /mnt/c 9p; missing dirs | runtime on ext4; `mkdir -p` data dirs first |
| serve refuses to start | empty users table | create admin first |

---

## 14. Known limitations (honest)

- **Quality not benchmarked.** All three big models are FP8 and gen's
  `max_pixels` is reduced; the system *works*, but retrieval/description quality
  vs the BF16 / 96G reference is **unmeasured** (the `knowledge-base` 48G doc's
  3 gating POCs — embed-FP8 recall, rerank-FP8 ordering, 8B-desc quality — were
  not run with `judge_eval`). If you care about quality, that's the next step.
- VRAM headroom at ingest peak is ~5–7G; watch it if you raise utils or
  `max_pixels`, or run a bigger description model.
- The offline `quant` env + `quantize_*.py` are abandoned (kept as record).
