<div align="right">

**English** | [中文](comparison.zh.md)

</div>

# Comparison

> If you're searching for "yet another RAG library," this page is the
> short answer for whether NaviKB is what you want.

## TL;DR

| Use case | Best fit |
|---|---|
| "I want a Python library to build a RAG pipeline" | **LlamaIndex** or **LangChain** |
| "I want to chat with my docs in 5 minutes via a UI" | **Verba**, **Open WebUI + RAG plugin**, or hosted (NotebookLM, Perplexity Spaces) |
| "I want LLM-extracted entities + relations indexed as a graph" | **GraphRAG** (Microsoft) or **LightRAG** |
| "I'm researching navigable RAG algorithms for a paper" | **NaviRAG**, **CORPUS2SKILL**, **Don't Retrieve, Navigate** |
| "I want to operate a small-team KB with auth, audit, soft-delete, backup, MCP, all local" | **NaviKB** ← you're here |

The framing decision: NaviKB is a *system you operate*, not a library
you compose. If you want primitives to build with, the library
alternatives below will give you more flexibility and a faster start.
If you want something to *run* with operational defaults a small team
can trust, NaviKB is aimed at that gap.

## Detailed comparison

### vs LlamaIndex

[LlamaIndex](https://www.llamaindex.ai/) is the maturest open-source
RAG / agent framework. It is a *library* — you compose nodes,
retrievers, query engines, and agents in Python code.

| Dimension | LlamaIndex | NaviKB |
|---|---|---|
| Shape | Python library | End-to-end system (server + worker + scripts) |
| Hierarchical retrieval | `SummaryIndex`, `DocumentSummaryIndex`, `TreeIndex` available but optional | NavIndex is the primary access path |
| Pointer-only navigation | Not first-class; retrievers return text by default | First-class: `kb_search_nav` returns pointers, no text |
| MCP-native | Adapter available via third-party packages | Built-in `/mcp` sub-app, first-class tool surface |
| Citation discipline | Caller's responsibility | Enforced at API surface (evidence_fields vs hint_fields) |
| Auth / audit / backup / GC | "Build it yourself" | Designed in, end-to-end |
| Maturity | Mature, large community, many integrations | Pre-1.0 (Stage 0 design preview) |

**Pick LlamaIndex when** you want maximum flexibility, you're building
a custom RAG pipeline that doesn't fit a system's mold, you need
integrations with many specific vector DBs / LLM providers, or you're
prototyping and want to swap components.

**Pick NaviKB when** you want a system you can stand up, configure,
and trust to behave under operational stress (delete, restore,
backup, multi-process maintenance) without writing that layer yourself.

### vs LangChain

[LangChain](https://www.langchain.com/) is the other dominant
framework. Same library-vs-system distinction applies to NaviKB.

The specific thing LangChain optimizes for is *chain composition* —
ordering tools, retrievers, LLMs in a DAG. NaviKB takes the opposite
position: the access pattern (navigation-first + 4-channel retrieval
fused with RRF + cross-encoder rerank + evidence/hint partition) is
**baked in**. Operators get the pattern; they don't get to rebuild it.

This is good if you want the opinionated pattern. It is bad if your
problem doesn't fit it (e.g. you're building a multi-modal agent that
needs a custom plan-execute-observe loop). LangChain is more
flexible. NaviKB is more opinionated.

### vs Haystack

[Haystack](https://haystack.deepset.ai/) sits between LlamaIndex and a
fully-system project. It has pipeline abstractions and reasonable
defaults but expects you to deploy and operate it yourself.

The biggest distinction from NaviKB is the **default access pattern**:
Haystack is centered on retrieval pipelines (Retriever → Reader →
Generator). NaviKB is centered on the navigation graph; retrieval is
one component among several, not the spine.

For projects that already have Haystack and a working retriever, the
right thing to do is probably stick with it. NaviKB is not a drop-in
replacement.

### vs GraphRAG

[Microsoft GraphRAG](https://github.com/microsoft/graphrag) takes the
"knowledge graph from LLM extraction" approach: feed each document to
an LLM, extract entities and relationships, build a graph index,
serve queries by traversing the graph.

This is a powerful and very different design philosophy from NaviKB:

| Dimension | GraphRAG | NaviKB |
|---|---|---|
| Graph source | LLM-extracted entities + relations | Document's own structural index (headings, tables, figures) |
| Trust in the graph | Operators must trust the LLM's extraction quality | Graph is parser-deterministic; aliases are explicitly human-curated |
| Ingest cost | High (one or more LLM calls per chunk) | Low (no LLM calls in ingest's structural pass; description channel is optional) |
| Best for | Synthesizing across documents around discovered concepts | Helping the reader navigate the documents they actually have |
| Deployment | Designed for enterprise stacks (Neo4j-class infra) | Designed for one machine + SQLite |

GraphRAG and NaviKB are not really competing — they're answering
different questions. If your question is "what does my corpus say
about Topic X synthesizing across documents," GraphRAG is the better
shape. If your question is "where in this manual is the procedure
for Y," NaviKB is.

### vs NaviRAG and "Don't Retrieve, Navigate"

These are 2026 research papers exploring the same direction NaviKB
chose (navigation as the primary access pattern):

- [NaviRAG (arXiv:2604.12766)](https://arxiv.org/abs/2604.12766):
  "locate first, then forage." LLM agent navigates a distilled
  hierarchical representation, iteratively identifying information
  gaps.
- ["Don't Retrieve, Navigate" (arXiv:2604.14572)](https://arxiv.org/abs/2604.14572):
  distills enterprise knowledge into "navigable agent skills."

| Dimension | Research papers | NaviKB |
|---|---|---|
| Goal | Algorithm contribution | Production system |
| Hierarchy source | LLM-distilled from corpus | Parser-extracted from document structure |
| Implementation status | Reference code (NaviRAG: 5 stars, "under active organization"; DRN: paper only) | 800+ tests, pre-release |
| Operational surface | None (research demo) | Auth, audit, backup, GC, MCP server |
| Citation discipline | Not explicitly addressed | Enforced |

NaviKB is *not* an implementation of NaviRAG. It predates that paper
and uses a different mechanism (parser headings, not LLM distillation).
The naming overlap is intentional — NaviKB positions itself in the
"navigable KB" research direction — but the projects solve different
problems at different layers.

If you're a researcher comparing retrieval algorithms, read the papers
and ignore NaviKB. If you're an operator who wants to run a navigable
KB on real documents this quarter, NaviKB is the production-stack
answer underneath that class of design.

### vs Verba

[Verba](https://github.com/weaviate/verba) is Weaviate's open-source
"RAG out of the box" project — a web UI on top of Weaviate with
ingest, chat, and basic configuration.

| Dimension | Verba | NaviKB |
|---|---|---|
| Interface | Web chat UI (primary) | MCP for agents + REST for scripts (no built-in UI) |
| Vector store | Weaviate | Qdrant (local file mode default) |
| Target user | End user wanting to chat with their docs | Operator wanting to expose a KB to multiple agents |
| Citation discipline | Standard RAG (chunk + cite) | Evidence vs hint partition |
| Auth | Single-user (default deployment) | Multi-user from day 1 (manage_users.py) |

Pick Verba when an end-user wants a chat interface in a browser. Pick
NaviKB when the target consumer is an agent (Claude Code, Continue,
Cline, etc.) and you care about who-can-do-what beyond a single user.

### vs hosted services (NotebookLM, Perplexity Spaces, etc.)

Hosted services are a different category. They're easier to start with
and harder to control. They typically:

- Don't expose an agent tool surface (closed UI)
- Don't let you self-host on sensitive corpora
- Don't give you the audit trail you'd want for regulated documents
- Are good at the chat experience NaviKB explicitly does not provide

If your corpus is something you'd be OK sending to Google or
Perplexity, a hosted service is the path of least resistance. If
it's something that needs to stay on your machine or LAN, NaviKB
sits where they can't.

## Honest weaknesses

This section lists where the comparisons above would tilt against
NaviKB if we were being asked by someone evaluating us.

### Where NaviKB is weaker than the alternatives

- **Code is not public yet.** LlamaIndex, LangChain, Haystack, Verba,
  GraphRAG all have working installable packages. NaviKB has docs.
  The gap is real and only closes when we hit Stage 2 (see
  [docs/status.md](../status.md)).
- **Smaller ecosystem.** No connectors for 50 document formats, no
  100 vector DB integrations, no example notebooks for every LLM
  provider. NaviKB picks a small set (MinerU for parse, BGE-M3 for
  text embed, ColQwen2 for vision, Qdrant for vectors) and supports
  them well; adding a new combination requires writing code, not
  swapping a config.
- **No UI.** This is deliberate — the UI is the agent — but if you
  want a "chat with docs" experience for a non-agent user, NaviKB
  is the wrong tool.
- **No managed-service option.** NaviKB is local-first and intends to
  stay that way. Operators who want SaaS economics should look at
  LlamaCloud, Weaviate Cloud, etc.
- **Real-document evaluation not yet published.** The 800+ test count
  measures correctness on synthetic / unit-scope inputs. The real-
  document Recall/NDCG/MRR benchmark is part of the public-release
  blocker list. Until it lands, anyone comparing on retrieval quality
  is comparing the design's claims, not measured numbers.

### Where the alternatives are weaker than NaviKB

- **Operational depth.** Most RAG frameworks treat auth, audit, soft-
  delete, maintenance flag, backup, GC as "build it yourself." NaviKB
  designs them in.
- **Citation discipline.** Most stacks give the LLM whatever the
  retriever returns and trust it to cite responsibly. NaviKB
  partitions evidence vs hint at the API surface.
- **Cross-process correctness.** Most projects don't separate
  ingestion worker from serving process; the ones that do (e.g. Celery
  + a web app) rely on message brokers. NaviKB uses SQLite as the
  cross-process coordinator and pays the small-deployment cost (one
  machine) to skip the broker complexity.
- **Navigation-first as a design commitment.** Many frameworks have
  hierarchical retrieval as an option. NaviKB makes it the default
  and makes the pointer-only contract explicit in the API.

## How we'll keep this honest

This document will get specific things wrong. Comparison docs always
do — projects change, our understanding of them changes, and we have
our own bias as authors of NaviKB.

The mechanism for fixing that: if you maintain or use one of the
projects above and you think this page mischaracterizes it,
[open an issue](../../../issues) with the `comparison-correction`
label. We will read every one. We may not agree with every one, but
we'll engage on the technical merits.

## Where to read next

- [`architecture.md`](architecture.md) — what NaviKB looks like in
  detail
- [`navigation-first.md`](navigation-first.md) — the design commitment
  that drives most of the differentiation above
- [`security-model.md`](security-model.md) — the operational layer
  where the comparison most strongly favors NaviKB
- [`../status.md`](../status.md) — the honest current state, including
  the public-release blockers
