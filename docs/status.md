<div align="right">

**English** | [中文](status.zh.md)

</div>

# Status & Honest Maturity

> **Honest summary:** the code is public and runs. This is a research /
> example reference implementation in **early alpha** — not a turnkey
> product, and not "released" in the semantic-versioning sense. This page is
> the honest account of what works, what doesn't, and where the real risks are.

---

## What works today

- The design has been iterated through six internal review stages, each with
  per-PR design specs and code reviews.
- The implementation passes its own test suite (800+ tests) covering
  retrieval, auth, audit, soft-delete, the maintenance flag, backup, GC, and
  the MCP tool surface.
- Core invariants — cross-process maintenance gate, two-phase audit,
  evidence/hint partition, navigation-first retrieval — are in place and tested.
- It runs end-to-end on the maintainer's machine with the reference model
  stack (Qwen3-VL embeddings + reranker, MILCO learned sparse, Qwen3.6-VL page
  descriptions, MinerU layout parsing).

## What doesn't work / isn't verified yet

- **Reproducibility is only lightly verified.** A from-source install
  (`pip install -e ".[server,dev]"`) plus `import kb` succeeds on a clean
  Python 3.12 virtualenv. It has **not** been reproduced by a second person or
  on a second OS — that, plus the heavier optional `[mineru]` extra and the
  live model services, is the open gap.
- **The evaluation story is still maturing.** The golden set is mostly
  synthetic; broad real-document evaluation is the single largest open quality
  risk.
- **No released package.** There is no PyPI release or container image yet;
  installation is from source.
- **The model / deployment matrix is a reference, not a tested grid.** Only the
  one reference stack above has been run end-to-end. Other model endpoints
  *should* work — everything sits behind `*_SERVER_URL` / `*_MODEL_ID` — but
  have not been validated.

## Why publish now, as "research / example"

Two reasons.

First, **the design is the contribution at this stage.** Several decisions in
NaviKB — navigation-first as the primary access pattern, the evidence/hint
partition, two-phase audit on a small-team stack, the cross-process cache
epoch — are worth reading and arguing about on their own merits, independent of
how polished the packaging is. Publishing the code makes that discussion
concrete instead of hypothetical.

Second, **vaporware is cheap; an honest alpha is useful.** Rather than sit on
the code until everything is perfect, we'd rather ship a working reference
implementation and be honest about the gaps. If you set it up and something
breaks, that is exactly the feedback that closes the reproducibility gap.

## How you can help

- **🐛 Open an issue** if the install breaks, a doc is unclear, or a comparison
  is wrong. A second-person install report is the single most useful thing
  right now.
- **📋 Read the design docs** —
  [architecture](design/architecture.md),
  [navigation-first](design/navigation-first.md),
  [security-model](design/security-model.md),
  [comparison](design/comparison.md).
- **⭐ Star / watch** to follow along as the rough edges get sanded down.

## Roadmap (honest, no dates)

Roughly in priority order, the things most likely to change:

1. A reproducible clean-machine install — a dependency pass plus a
   second-person install report.
2. A real-document evaluation report (Recall@k, NDCG, MRR) on at least one
   non-synthetic corpus, with the golden set published.
3. A documented threat model, and an auth/audit review by someone other than
   the author.
4. A validated "bring your own model" matrix beyond the single reference stack.

We do not give calendar dates — every internal date we have set has slipped.
The list above is the honest backlog, not a release schedule.

---

*Research / example · early alpha. Last updated: 2026-06-05.*
