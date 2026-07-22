#!/usr/bin/env python3
"""vLLM ↔ 官方 Qwen3VLEmbedder 文本向量等价性 go/no-go 探针(docs/VLLM_PLAN.md Phase 0 / G0+G1)。

**决定 vLLM 方案生死的一个数**:cosine(vLLM 出的查询向量, 官方脚本出的向量)。>0.9999 = G1 GO
(vLLM 能查现有库);否则现有库不能被 vLLM 查(向量漂移→top-k 错位),须先对齐或全库重建。

**为什么分时两步**:官方(pharos, transformers 4.57)与 vLLM(vllm env, transformers 5.10)在不同 conda 环境,
且 8B 双加载会 OOM。故 `--step official` 先出向量存档 → `--step vllm` 再出向量并比对。

**等价前提(照抄官方 recipe,否则假阴)**:两端都用 `[{system:instruction},{user:text}]` 会话 +
`apply_chat_template(add_generation_prompt=True, tokenize=False)`(实读官方 `qwen3_vl_embedding.py::_preprocess_inputs`
与 vLLM `examples/embedding_vllm.ipynb` 一致)。instruction 末尾非标点则补 '.'(官方 format_model_input 规则)。

跑:
  # 0) 腾 GPU(停 torch inference 容器)
  docker compose --env-file .env.compose stop inference
  # 1) 官方向量(pharos)
  conda activate pharos && python scripts/vllm_equiv_probe.py --step official
  # 2) vLLM 向量 + 比对(vllm env)
  conda activate vllm    && python scripts/vllm_equiv_probe.py --step vllm
"""
from __future__ import annotations

import argparse
import os
import sys
import unicodedata

import numpy as np

# 与库一致:dense_dim=1024 是生产截维;但等价性先比**全维 4096**(截维是确定性后处理,全维等价则截维必等价)。
EMB_MODEL = os.path.expanduser("~/models/Qwen3-VL-Embedding-8B")
DEFAULT_INSTRUCTION = "Represent the user's input."          # 官方 wrapper default_instruction
QUERY_INSTRUCTION = "Retrieve relevant documents for the query."   # pharos EmbedConfig.query_instruction

# 覆盖:中/英、长/短、特殊字符、带不同 instruction。查询路径是纯文本,故只测 text。
SAMPLES = [
    ("Netflix 2015 年的营收和净利润分别是多少?", QUERY_INSTRUCTION),
    ("What was IBM's total revenue growth year over year?", QUERY_INSTRUCTION),
    ("混合 中英文 test 123 !@#$% 特殊字符与标点。", QUERY_INSTRUCTION),
    ("a", QUERY_INSTRUCTION),
    ("公司在报告期内的现金流量表显示经营活动产生的现金流净额同比上升,主要由于应收账款周转加快以及"
     "存货管理优化,同时投资活动现金流出增加反映了资本开支扩张。", QUERY_INSTRUCTION),
    ("quarterly earnings per share diluted", DEFAULT_INSTRUCTION),
    ("董事会关于利润分配的决议", DEFAULT_INSTRUCTION),
    ("supply chain risk and mitigation strategy", DEFAULT_INSTRUCTION),
]


def _system_instruction(instr: str) -> str:
    """复刻官方 format_model_input:instruction 末尾非标点(Unicode P*)则补 '.'。"""
    instr = (instr or DEFAULT_INSTRUCTION).strip()
    if instr and not unicodedata.category(instr[-1]).startswith("P"):
        instr = instr + "."
    return instr


def _conversation(text: str, instr: str) -> list:
    """两端共用的会话结构(text-only)。"""
    return [
        {"role": "system", "content": [{"type": "text", "text": _system_instruction(instr)}]},
        {"role": "user", "content": [{"type": "text", "text": text}]},
    ]


def _out_path(out_dir: str) -> str:
    return os.path.join(out_dir, "official_vecs.npz")


def step_official(out_dir: str) -> None:
    """pharos:用官方 Qwen3VLEmbedder 出全维向量,存档。这是**权威**(现有库就是它建的)。"""
    import torch
    scripts = os.path.join(EMB_MODEL, "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    from qwen3_vl_embedding import Qwen3VLEmbedder

    print(f"[official] 加载 {EMB_MODEL} (bf16)…", flush=True)
    model = Qwen3VLEmbedder(model_name_or_path=EMB_MODEL, torch_dtype=torch.bfloat16)
    vecs = []
    for text, instr in SAMPLES:
        emb = model.process([{"text": text, "instruction": instr}], normalize=True)  # (1, 4096) 已归一
        vecs.append(np.asarray(emb.float().cpu().numpy()[0], dtype=np.float32))
    arr = np.stack(vecs)
    os.makedirs(out_dir, exist_ok=True)
    # 只存数值数组(texts/instrs 已在 SAMPLES 常量,无需持久化)-> 读侧 allow_pickle=False,无反序列化风险。
    np.savez(_out_path(out_dir), vecs=arr)
    print(f"[official] 存档 {arr.shape} -> {_out_path(out_dir)};norm 范围 "
          f"[{np.linalg.norm(arr,axis=1).min():.5f}, {np.linalg.norm(arr,axis=1).max():.5f}]", flush=True)


def step_vllm(out_dir: str) -> None:
    """vllm env:vLLM pooling 出向量,与官方存档比 cosine → G0/G1 判定。"""
    p = _out_path(out_dir)
    if not os.path.exists(p):
        raise SystemExit(f"未找到官方存档 {p};先在 pharos 跑 --step official。")
    ref = np.load(p)                # allow_pickle=False(默认):只读数值数组,无反序列化风险
    ref_vecs = ref["vecs"]

    # G0:vLLM 能否在 pooling 模式加载本 VL embedding 模型
    try:
        from vllm import LLM
    except Exception as e:
        raise SystemExit(f"[G0 FAIL] import vllm 失败:{e}(确认 conda activate vllm)")
    print(f"[vllm] LLM(runner='pooling') 加载 {EMB_MODEL}…(首次可能较慢)", flush=True)
    try:
        llm = LLM(model=EMB_MODEL, runner="pooling", dtype="bfloat16", trust_remote_code=True,
                  enforce_eager=True,          # 排除 cudagraph 变量,专注等价性
                  max_model_len=8192,          # ⚠ 关键:模型 max_position_embeddings=262144 会让 vLLM 要 36GB KV cache
                  #                               而 embedding 是单次前向、无需长上下文;官方 wrapper 也 MAX_LENGTH=8192。不设必 OOM。
                  gpu_memory_utilization=0.90)
    except Exception as e:
        # ⚠ 不下"架构不支持"的 verdict —— 原样打出真实报错自行定位(前两次"失败"实为 device UUID / KV cache 配置,非架构问题)。
        raise SystemExit(f"[vLLM 初始化失败] {type(e).__name__}: {e}\n"
                         f"→ 看上方 EngineCore Traceback 真实根因(可能是显存/max_model_len/版本,未必是架构不支持)。")

    tok = llm.get_tokenizer()
    prompts = []
    for (text, instr) in SAMPLES:
        s = tok.apply_chat_template(_conversation(text, instr), tokenize=False, add_generation_prompt=True)
        prompts.append({"prompt": s})
    outputs = llm.embed(prompts)
    vllm_vecs = np.stack([np.asarray(o.outputs.embedding, dtype=np.float32) for o in outputs])

    # 维度对齐检查(全维应 4096)
    if vllm_vecs.shape[1] != ref_vecs.shape[1]:
        print(f"[warn] 维度不一致 vLLM={vllm_vecs.shape[1]} 官方={ref_vecs.shape[1]}"
              f"(若 vLLM 未归一/截维不同,cosine 仍可比方向)", flush=True)
    d = min(vllm_vecs.shape[1], ref_vecs.shape[1])

    # 逐条 cosine(两端都应已归一;保险起见显式归一再点积)
    def _cos(a, b):
        a = a[:d] / (np.linalg.norm(a[:d]) + 1e-12)
        b = b[:d] / (np.linalg.norm(b[:d]) + 1e-12)
        return float(np.dot(a, b))

    print("\n=== G1 向量等价(cosine per sample)===")
    cosines = []
    for i, (text, _) in enumerate(SAMPLES):
        c = _cos(vllm_vecs[i], ref_vecs[i])
        cosines.append(c)
        vnorm = float(np.linalg.norm(vllm_vecs[i]))
        print(f"  [{i}] cos={c:.7f}  vllm_norm={vnorm:.5f}  «{text[:32]}»")
    cosines = np.array(cosines)
    mn, mean = cosines.min(), cosines.mean()
    print(f"\n--- min cos={mn:.7f}  mean={mean:.7f} ---")
    if mn > 0.9999:
        print("✅ G1 GO:min cosine > 0.9999 —— vLLM 出的查询向量与现有库(官方建)数学等价,可查现有库。")
    elif mn > 0.999:
        print("⚠ G1 边缘:0.999 < min < 0.9999 —— 方向高度一致但有微漂(疑 transformers 版本/bf16)。"
              "须跑 G2 混建混查看 top-k 是否真翻转;必要时 vllm env 对齐 transformers 版本重测。")
    else:
        print("❌ G1 NO-GO:min cosine ≤ 0.999 —— 向量漂移显著。现有库不能被 vLLM 直接查。"
              "先查 input 格式/tokenize 是否逐 token 一致(假阴常因此),否则须全库用 vLLM 重建(见 VLLM_PLAN §8)。")


def main() -> None:
    ap = argparse.ArgumentParser(description="vLLM↔官方 文本向量等价性探针(G0/G1)")
    ap.add_argument("--step", required=True, choices=["official", "vllm"])
    ap.add_argument("--out", default="/tmp/vllm_probe", help="向量存档目录(两步共用)")
    args = ap.parse_args()
    if args.step == "official":
        step_official(args.out)
    else:
        step_vllm(args.out)


if __name__ == "__main__":
    main()
