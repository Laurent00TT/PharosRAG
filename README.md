<div align="right">

**English** | [中文](README.zh.md)

</div>

# NaviKB

**Navigable Knowledge Base** — a navigation-first, local-first knowledge base
for individuals and small teams.

> ⚠ **Status: design-stage, public preview.** This repository currently
> contains design documentation only. Implementation is in private
> development. See [docs/status.md](docs/status.md) for the release plan
> and acceptance criteria.

---

## What it is

NaviKB is a knowledge base built around two ideas the mainstream RAG stack
gets wrong:

1. **Navigation, not retrieval, is the primary access pattern.** Most RAG
   systems treat documents as a flat bag of chunks and rely on dense
   similarity to find the right text. NaviKB indexes the *structure* of
   each document (headings, tables, figures, page ranges) into a
   pointer-only **NavIndex**, and lets the agent walk that structure the
   same way a human reader would.
2. **Generated content is not evidence.** An LLM's description of a page
   is a recall hint. Citing it as if it were source text is a hallucination
   amplifier. NaviKB partitions every response into `evidence_fields`
   (parsed text, image URLs, captions) and `hint_fields` (LLM-generated
   descriptions), and tells the agent which it is allowed to cite.

NaviKB is built for the realistic shape of personal and small-team work:
a few hundred to a few thousand PDFs, a handful of operators, and a
strong preference for running locally without SaaS dependencies.

## What it is not

- **Not a hosted service.** Everything runs on your machine or LAN.
- **Not a vector-search wrapper.** Retrieval is one of four channels
  (text dense, text sparse, vision, description), fused with RRF
  and reranked — but always anchored to navigable structure.
- **Not a knowledge graph.** No entity extraction, no ontology induction,
  no LLM-built relationship edges. The graph is the document's own
  table of contents.
- **Not an enterprise platform.** No multi-tenant orchestration, no
  Kubernetes operators, no Celery / Redis / Kafka. SQLite + Qdrant + a
  worker process is the entire stack.

## Why this shape

The design is a response to two trends in 2026 RAG that we think are
diverging in the wrong direction:

- **Toward heavier infrastructure** — managed vector DBs, graph
  databases, agent platforms, observability stacks. Great for enterprise
  with a dedicated ops team. Hostile to individuals and 2-5 person teams.
- **Away from document structure** — chunk everything, embed everything,
  hope similarity finds it. Loses the affordances that make a real
  document navigable (sections, figures, tables, "see chapter 3").

NaviKB picks the opposite stance on both: small footprint, deep respect
for structure, agent-native via MCP.

See [docs/design/architecture.md](docs/design/architecture.md) for the
shape in detail, and [docs/design/navigation-first.md](docs/design/navigation-first.md)
(coming next) for the design rationale.

## Differentiation

| Concern | NaviKB | Typical RAG (LangChain / LlamaIndex) | GraphRAG | NaviRAG (research) |
|---|---|---|---|---|
| Primary access pattern | Pointer navigation + 4-channel retrieval | Dense retrieval over flat chunks | Graph walk over LLM-extracted entities | Hierarchical LLM-driven exploration |
| Local-first | ✅ default | ⚠ possible but uncommon | ❌ usually needs Neo4j-class infra | ❌ research-stage |
| MCP-native | ✅ first-class tool surface | ⚠ via adapters | ❌ | ❌ |
| Citation discipline | ✅ evidence/hint partition enforced | ❌ caller decides | ❌ | ❌ |
| Operator surface (auth, audit, backup, GC) | ✅ end-to-end | ❌ "build it yourself" | ⚠ enterprise stacks only | ❌ |
| Status | design-public, code-private | mature | mature | research prototype |

A fuller comparison with reasoning will land in `docs/design/comparison.md`.

## Documentation map

| Document | Status | Purpose |
|---|---|---|
| [README](README.md) (you are here) | ✅ public | 30-second overview |
| [docs/status.md](docs/status.md) | ✅ public | Where the code is, when it ships, what blocks release |
| [docs/design/architecture.md](docs/design/architecture.md) | ✅ public | Four-layer architecture: ingest → storage → retrieve → serve |
| `docs/design/navigation-first.md` | 🔄 in progress | Why NavIndex; pointer-only design; trade-offs vs chunking |
| `docs/design/security-model.md` | 🔄 in progress | Hard-cutover auth, two-phase audit, soft delete, maintenance flag, GC retention |
| `docs/design/comparison.md` | 🔄 in progress | Deep comparison with LangChain / LlamaIndex / GraphRAG / NaviRAG |
| [CONTRIBUTING.md](CONTRIBUTING.md) | ✅ public | What feedback is welcome right now (and what isn't) |

## Quick links

- 📋 [Status & release plan](docs/status.md)
- 🏛 [Architecture](docs/design/architecture.md)
- 💬 [Discussion: open an issue](../../issues) — design feedback welcome
- 🤝 [Contributing](CONTRIBUTING.md)
- 📜 [License: Apache-2.0](LICENSE)

## Naming

**NaviKB** = **Navi**gable + **K**nowledge **B**ase.

Pronounced "nah-vee-K-B" (three syllables) or "navy-K-B" (the lazy
two-syllable form). Positions the project in the 2026 *navigable
knowledge base* research direction (see
[NaviRAG](https://arxiv.org/abs/2604.12766),
["Don't Retrieve, Navigate"](https://arxiv.org/abs/2604.14572)) while
focusing on a different layer: those are retrieval-algorithm research
prototypes; NaviKB is the production-operations stack underneath that
class of design.
