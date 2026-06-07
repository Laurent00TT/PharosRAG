# NaviKB — local 4090-48G all-online deployment

Runs the **full NaviKB model stack on a single RTX 4090 48G**, with **search and
ingest available simultaneously** (all models resident). Everything runs in
**WSL2 (Ubuntu)**; the Windows git repo on `C:` is untouched and used via an
editable install.

## Topology

| Service | Port | Env | VRAM (4090) | Notes |
|---|---|---|---|---|
| embed — Qwen3-VL-Embedding-8B | 8003 | `vllm` | ~11.5G | BF16 + `--quantization fp8` (dynamic) |
| rerank — Qwen3-VL-Reranker-8B | 8005 | `vllm` | ~12.5G | BF16 + `--quantization fp8`, `TORCH_SDPA` |
| gen VLM — Qwen3-VL-8B-Instruct-FP8 | 8006 | `vllm` | ~13G | description + agent + query-rewrite; `TORCH_SDPA`, flashinfer-sampler off |
| sparse — MILCO-650m | 8004 | `vllm` (CPU) | 0 (CPU) | `CUDA_VISIBLE_DEVICES=""`; sub-tokenizers (splade-v3 + bge-m3) loaded from local paths |
| mineru — MinerU2.5 | 8101 | `mineru` | ~6G (transient) | `MINERU_PARSE_MODE=pipeline` (hybrid deadlocks in WSL) |
| qdrant | 6333 | binary | — | server mode (REQUIRED for concurrent search+ingest) |
| serve (FastAPI + /mcp) | 8000 | `navikb` | — | control plane |
| worker (ingest queue) | — | `navikb` | — | control plane |

Peak ~42G/49G during ingest. All on localhost.

## One-time setup (already done)

1. `bash 01_setup_vllm_env.sh` — Miniconda (`~/miniconda3`, no sudo) + `vllm` env.
   Then `conda install -n vllm -c <tuna-cf> gcc gxx` + `bash _setup_cc.sh` (Triton needs `gcc`/`cc`).
2. `bash 03_setup_mineru_env.sh` ; `bash 04_setup_navikb_env.sh` (control plane) ; `bash _start_qdrant.sh` source via `05_setup_qdrant.sh` (binary already extracted to `~/navikb-serving/qdrant/`).
3. Models in `~/navikb-serving/models/` — embed/rerank/gen/mineru via modelscope (`02_download_models.sh bulk`); MILCO + splade-v3 via `download_milco_curl.ps1` / `download_hf_repo.ps1` (curl through the Clash proxy — see Gotchas).
4. `bash _sparse_fix.sh` (point MILCO's two sub-tokenizers at local paths).
5. `bash _create_admin.sh` (admin token printed once).

## Daily bring-up / after reboot

```bash
wsl -d Ubuntu -- bash /mnt/c/Users/11541/Desktop/projects/navikb/deploy/local-48g/start_everything.sh
```
Brings up all 8 services detached (survives the terminal). Status / stop:
```bash
bash _status.sh      # health of all services + 4090 VRAM
bash stop_all.sh     # stop everything
```

## Use it

```bash
TOKEN=kb_admin_...          # from _create_admin.sh (reset: manage_users.py reset-key admin)
# search (4-channel + rerank):
curl -s http://localhost:8000/search_docs -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' -d '{"query":"...","top_k":5}'
# ingest one PDF end-to-end (mineru -> 4 channels -> Qdrant):
bash _ingest.sh /mnt/c/path/to/file.pdf
```
Windows access: `http://localhost:8000` (WSL2 forwards localhost) ; models at `\\wsl.localhost\Ubuntu\home\tiantian\navikb-serving\models`.

## Gotchas (root-caused during bring-up)

- **HF is unreachable directly on this machine.** Only path that works:
  `curl.exe -x http://127.0.0.1:7897 -L` (Windows side, through the Clash mixed-port).
  modelscope + tuna work directly from WSL; aliyun / hf-mirror-direct / hf.co do not.
- **CUDA pinning:** 4090 = `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1` (index 0 is the 5070 8G).
- **FP8 = vLLM `--quantization fp8` on the BF16 model** — NOT offline llmcompressor (it loads embedding weights as random).
- **gen needs** `VLLM_USE_FLASHINFER_SAMPLER=0` + `VLLM_ATTENTION_BACKEND=TORCH_SDPA` (no nvcc for flashinfer JIT) + bounded vision profile (`--max-num-seqs 2 --mm-processor-kwargs max_pixels=401408`).
- **mineru:** `MINERU_PARSE_MODE=pipeline` (hybrid deadlocks via multiprocessing.spawn in WSL); first /parse downloads its model (~2.15G) and may exceed navikb's client timeout — retry, the model stays resident (`_ingest_retry.sh`).
- **Persistence:** services run under whatever launched them. For unattended operation use `start_everything.sh` from a kept-open WSL terminal, WSL systemd, or a Windows scheduled task running the wsl command.
