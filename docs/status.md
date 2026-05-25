<div align="right">

**English** | [中文](status.zh.md)

</div>

# Status & Release Plan

> **Honest summary:** the design is done and the implementation works
> on the maintainer's machine. It is **not yet shipped**. This page
> tells you what's true today, what blocks public release, and what
> the timeline looks like.

---

## What's true today

- The design has been iterated through six stages of internal review
  (the small-team stage T1a → T5) with explicit per-PR design specs
  and code reviews
- The implementation passes 828 of its own tests in private development,
  covering retrieval, auth, audit, soft-delete, maintenance flag, backup,
  GC, and the MCP tool surface
- Core invariants (cross-process maintenance gate, two-phase audit,
  evidence/hint partition, navigation-first retrieval) are in place
  and tested

## What's not true today

- The code is **not public**. Reasons: see [Why the code isn't public yet](#why-the-code-isnt-public-yet) below
- There is **no released package** on PyPI or any container registry
- The reproducibility story is unverified: it works for the maintainer
  but has not been independently set up by a second person
- The eval golden set is still mostly synthetic; real-document
  evaluation is the largest open quality risk

## Why the code isn't public yet

The bar for public release is not "tests pass." It is:

1. **A second person can reproduce the install + first ingest** on a
   clean machine in under 30 minutes, following only public docs
2. **The eval golden set has at least one real-document benchmark** —
   not just synthetic queries — covering the four query types we
   actually expect (fact lookup, process / how-to, table lookup,
   visual layout)
3. **The threat model is documented** and the auth surface has been
   reviewed by someone who isn't the author
4. **The model upgrade path is concrete** — currently the codebase
   targets BGE-M3 + ColQwen2 + bge-reranker-v2-m3. The next-gen stack
   (Qwen3-VL + MILCO + BGE-M4 etc.) is designed but not validated,
   and shipping with two viable embedding versions is part of the
   release plan

Until all four are real, calling the project "released" would
misrepresent its readiness. We'd rather under-promise.

## Release plan

We use a three-stage public release.

### Stage 0 — Design preview (current)

- **What's public:** this repository's documentation
- **What you can do:** read the design, open issues with design feedback,
  watch the repo for the Stage 1 trigger
- **What you can't do:** install, ingest, search — there's no code here yet
- **Duration:** until Stage 1 acceptance criteria are met

### Stage 1 — Closed alpha

- **Trigger:** a small group (5–10 trusted users) has independently
  set up NaviKB and confirmed the docs cover the install path
- **What changes:** alpha users get a private repo invite + a setup
  document optimized for their feedback loop
- **What's still private:** the main code repo
- **Expected duration:** 4–8 weeks of iteration

### Stage 2 — Public open source

- **Trigger:** acceptance criteria (above) all met
- **What changes:** code becomes a public repository under this same
  GitHub organization; install instructions land in this repo's docs
- **Versioning:** semantic versioning starts at v0.1.0 (still expect
  breaking changes through v0.x)

We do not give a calendar date for any of these stages, because every
calendar date we've given internally has slipped. The triggers above
are the actual gates.

## Tracking progress

The fastest way to follow real progress:

- **⭐ Star or watch this repo** — Stage 1 invitations and Stage 2
  release will both be announced here
- **🐛 Open an issue** with the `design-feedback` label — every issue
  is read, even though we won't accept code PRs until Stage 2
- **📋 Read the design docs** —
  [architecture](design/architecture.md),
  [navigation-first](design/navigation-first.md),
  [security-model](design/security-model.md), and
  [comparison](design/comparison.md) are all live

## What we will publish before Stage 2

To make the release credible, we plan to publish these artifacts
*before* the code drops:

- A reproducible evaluation report (Recall@k, NDCG, MRR) on at least
  one real-document corpus, with the golden set published openly
- A second-person install report from a Stage 1 user (anonymized if
  they prefer)
- A documented threat model and the auth/audit surface review notes
- The model-upgrade-design spec, validated end-to-end on the new
  embedding stack

If any of these don't land, the release date doesn't either. We will
not ship a public v0.1.0 with a "TODO: evaluation" in the README.

## Why this transparency

Two reasons.

First, **vaporware is cheap and unhelpful**. Anyone can register a
repo name and put a logo on it. We'd rather show enough design to be
credible and enough honesty about the gap to be trustworthy.

Second, **the design itself is the contribution at this stage.**
Several decisions in NaviKB (navigation-first as the primary access
pattern, evidence/hint partition, two-phase audit on a small-team
stack, cross-process cache epoch) are worth discussing on their own
merits, independent of when the code ships. That discussion is the
purpose of the design preview.

---

*Last updated: 2026-05-25.*
