<div align="right">

**English** | [中文](README.zh.md)

</div>

# NaviKB

**Navigable Knowledge Base** — a navigation-first, local-first knowledge base
for individuals and small teams.

> 🔬 **Status: research / example — early alpha.** The code is here and runs,
> but this is a *reference implementation*, not a turnkey product. Expect rough
> edges: a clean-machine install has not yet been independently reproduced, and
> the evaluation story is still maturing. See [docs/status.md](docs/status.md)
> for an honest account of what works, what doesn't, and where the risks are.

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
for the design rationale behind making navigation the primary access pattern.

## Models and deployment are a reference, not a requirement

NaviKB decouples every model behind a configurable endpoint
(`*_SERVER_URL` / `*_MODEL_ID` in `.env`). The stack we happened to
validate — Qwen3-VL for text + vision embeddings and reranking, MILCO for
learned sparse retrieval, Qwen3.6-VL for page descriptions, MinerU for
layout parsing — is **one reference configuration**, not a hard dependency.

Point the endpoints at whatever you have: a local llama.cpp / LM Studio, a
remote GPU host reached over an SSH tunnel, or a hosted OpenAI-compatible
API. The engine, the four-channel retrieval, the navigation layer, and the
operator surface (auth, audit, soft-delete, backup, GC) are the contribution
here — the specific models are swappable, and the deployment topology is an
example, not a mandate.

## Quickstart

> This is a research/example stack. Treat the commands below as a starting
> point, not a hardened install path — see the
> [status notes](docs/status.md) on reproducibility.

```powershell
# 1. Install (Python 3.11+)
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[server,dev]"
.\.venv\Scripts\python.exe -m pip install -e ".[mineru]"   # PDF layout parsing (optional)

# 2. Configure — copy the template, then point the *_SERVER_URL / *_MODEL_ID
#    entries at your own model endpoints (see "Models ... a reference" above).
Copy-Item .env.example .env

# 3. Run the local control plane
python scripts/serve.py --port 8000     # FastAPI tool server (REST + MCP)
python scripts/worker.py                 # background ingestion worker

# Or bring up main + Qdrant + worker together via docker compose:
# docker compose -f docker/docker-compose.local.yml --env-file .env up -d --build
```

The server binds `127.0.0.1` by default. Create an admin account, then
ingest a PDF — NaviKB parses PDF layout, so it ingests `.pdf` files. No
third-party PDFs ship with this repo; bring your own, or build sample PDFs
from the bundled **synthetic** sources
(`datasets/pdf_corpus_v1/synthetic_sources/`, see `scripts/pdf_corpus_*.py`):

```powershell
python scripts/manage_users.py create <username> --role admin
python scripts/ingest.py --path .\path\to\document.pdf
```

All API requests carry an `Authorization: Bearer <user_token>` header.
Search, or ask a question the agent answers from retrieved evidence:

```powershell
curl -X POST http://127.0.0.1:8000/search_docs `
  -H "Content-Type: application/json" `
  -H "Authorization: Bearer $env:KB_USER_TOKEN" `
  -d "{\"query\":\"approval threshold\",\"top_k\":5}"

curl -X POST http://127.0.0.1:8000/ask `
  -H "Content-Type: application/json" `
  -H "Authorization: Bearer $env:KB_USER_TOKEN" `
  -d "{\"question\":\"What is the approval threshold?\"}"
```

Health check the local stack with `python scripts/doctor.py`. The MCP tool
surface is mounted at `/mcp`.

## The optional workbench

A web workbench (React + Vite) lives in a separate repository,
**[navikb-workbench](https://github.com/Laurent00TT/navikb-workbench)**. It is
a thin HTTP client of the core's `/ui/api` surface — there is no shared code.
Run this core first, then run the workbench against it (its dev server proxies
`/ui/api` back to `127.0.0.1:8000`). The core runs fully headless without it.

## Differentiation

| Concern | NaviKB | Typical RAG (LangChain / LlamaIndex) | GraphRAG | NaviRAG (research) |
|---|---|---|---|---|
| Primary access pattern | Pointer navigation + 4-channel retrieval | Dense retrieval over flat chunks | Graph walk over LLM-extracted entities | Hierarchical LLM-driven exploration |
| Local-first | ✅ default | ⚠ possible but uncommon | ❌ usually needs Neo4j-class infra | ❌ research-stage |
| MCP-native | ✅ first-class tool surface | ⚠ via adapters | ❌ | ❌ |
| Citation discipline | ✅ evidence/hint partition enforced | ❌ caller decides | ❌ | ❌ |
| Operator surface (auth, audit, backup, GC) | ✅ end-to-end | ❌ "build it yourself" | ⚠ enterprise stacks only | ❌ |
| Status | research / example, early alpha | mature | mature | research prototype |

Detailed reasoning per cell — and an honest "where NaviKB is weaker"
section — lives in [docs/design/comparison.md](docs/design/comparison.md).

## Documentation map

| Document | Purpose |
|---|---|
| [README](README.md) (you are here) | 30-second overview + quickstart |
| [docs/status.md](docs/status.md) | Honest maturity: what works, what doesn't, the open risks |
| [docs/design/architecture.md](docs/design/architecture.md) | Four-layer architecture: ingest → storage → retrieve → serve |
| [docs/design/navigation-first.md](docs/design/navigation-first.md) | Why NavIndex; pointer-only design; trade-offs vs chunking |
| [docs/design/security-model.md](docs/design/security-model.md) | Hard-cutover auth, two-phase audit, soft delete, maintenance flag, GC retention |
| [docs/design/comparison.md](docs/design/comparison.md) | Deep comparison with LangChain / LlamaIndex / GraphRAG / NaviRAG |
| [CONTRIBUTING.md](CONTRIBUTING.md) | What feedback and contributions are welcome |

## Quick links

- 📋 [Status & honest maturity](docs/status.md)
- 🏛 [Architecture](docs/design/architecture.md)
- 🖥 [Workbench (separate repo)](https://github.com/Laurent00TT/navikb-workbench)
- 💬 [Discussion: open an issue](../../issues/new/choose) — design feedback, comparison corrections, docs clarity
- 🤝 [Contributing](CONTRIBUTING.md)
- 🔒 [Security policy](SECURITY.md)
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
