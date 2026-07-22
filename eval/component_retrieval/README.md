# embedder/examples/eval — 检索质量评估(BM25 vs BGE-M3 + RRF 权重)

embedder 的"质量封板":用真实语料构造检索评估基准,数据定 sparse 选型 + RRF 权重。
手动跑(GPU + Qwen3-VL + BGE-M3 模型 + `parsed/` 真实数据,非 CI)。脚本 `REPO`=pharos 仓根(`__file__` 相对);引擎包已随 pharos 安装。

## 流程

```bash
python build_corpus_chunks.py   # 选 14 文档 -> chunk -> chunks.jsonl + 程序化精确词查询 queries_exact.jsonl
python select_semantic.py       # 选语义候选 chunk -> candidates.jsonl(再由 agent 生成 queries_semantic.jsonl)
python index_eval.py            # index 1067 chunk 到 Qdrant:dense(Qwen3-VL) + bm25 + bge-m3 三向量
python eval.py                  # #1 单路召回对比 + #2 RRF 权重扫描
```

- **精确词查询**(25):程序化挖语料里 df==1 的稀有精确串(`Section 6.1`/`CLEF-2021`/`$1196`/长编号),golden=含它的 chunk。测 sparse 精确命中。
- **语义查询**(31):agent 读 chunk 生成自然问题,**刻意避开原词**(同义/概括表达),golden=对应 chunk。测语义召回。
- `chunks.jsonl` / `qdrant/` 由脚本生成,gitignore 不入库(可重生成);`queries_*.jsonl` 入库(语义集 agent 生成不可复现)。

## 结论

**#1 单路召回(MRR / Recall@10)**

| route | 精确词(25) | 语义(31) | 整体(56) |
|---|---|---|---|
| bm25  | **0.738** / 0.96 | 0.210 / 0.45 | 0.446 |
| bgem3 | 0.584 / 0.88 | 0.347 / 0.61 | 0.453 |
| dense | 0.149 / 0.24 | **0.794** / 0.94 | 0.506 |

→ **选 BM25 确认**:sparse 该负责的精确词 BM25 明显赢 BGE-M3(0.738 vs 0.584);语义是 dense 的活(0.794 碾压两 sparse,BGE-M3 的语义优势在 hybrid 里冗余);整体两 sparse 打平,而 BM25 零模型/零 GPU/零维护 vs BGE-M3 2.3G + GPU。

**#2 RRF 权重(dense + bm25,扫描 w_dense)**

| w_dense | 0.0 纯sparse | 0.4 | 0.5 等权 | 1.0 纯dense |
|---|---|---|---|---|
| 整体 MRR | 0.449 | **0.541** | 0.499 | 0.510 |

→ **hybrid 优于任何单路**(峰值 0.541 > 纯 dense 0.510 > 纯 sparse 0.449)。最优 w_dense≈0.4 但强依赖查询 exact:semantic 比例(本集≈1:1)。
**决定**:store 保持 Qdrant 标准 RRF(等权,0.499 已接近峰值),不改客户端加权;原则"sparse 权重别给太低"。

**caveat**:合成评估集、56 query 规模小、语义 golden 单一(对三路公平)→ 趋势可信,绝对值仅供参考。

## 坑

BGE-M3 在多 GPU 机器自动起 multiprocessing pool,spawn 子进程 re-import 脚本 → 无 `if __name__` 保护会重入卡死。
本目录脚本已用 `main()` + `if __name__=='__main__'` 保护、`devices='cuda:0'` 锁单卡。详见仓库记忆 reference。
