<div align="right">

**English** | [中文](architecture.zh.md)

</div>

# Architecture

NaviKB is structured as four layers with explicit, well-defined contracts
between them. Each layer is independently testable and can be swapped
without redesigning the whole stack. This document describes the shape;
it does not document the API surface or operational procedures.

## Audience

This document is for engineers evaluating whether NaviKB's design fits
their use case, and contributors thinking about where a new feature
belongs. It assumes familiarity with vector retrieval, large-language-
model APIs, and the Model Context Protocol (MCP).

## Four layers

```text
┌─────────────────────────────────────────────────────────────────┐
│  4. Serve     FastAPI tool server  +  MCP sub-app  +  REST API  │
│               Auth · Audit · Maintenance · Backup · GC          │
└────────────┬────────────────────────────────────────────────────┘
             │
┌────────────▼────────────────────────────────────────────────────┐
│  3. Retrieve  4-channel retrieval (dense, sparse, vision, desc) │
│               RRF fusion → cross-encoder rerank → NavIndex      │
│               pointer enrichment → evidence/hint partition       │
└────────────┬────────────────────────────────────────────────────┘
             │
┌────────────▼────────────────────────────────────────────────────┐
│  2. Storage   Qdrant (3 versioned collections, multi-modal)     │
│               SQLite (metadata, audit, users, maintenance)      │
│               Local image store (page renders)                  │
│               SQLite NavIndex (pointer-only structural index)   │
└────────────┬────────────────────────────────────────────────────┘
             │
┌────────────▼────────────────────────────────────────────────────┐
│  1. Ingest    MinerU parse  →  chunk  →  4 channels embed       │
│               → metadata + nav auto-build  →  background worker │
└─────────────────────────────────────────────────────────────────┘
```

The arrow direction is the data flow during ingestion. Read flow inverts:
serve → retrieve → storage. Layer 1 (ingest) never reads from layer 4
(serve); the only communication between them is via SQLite job state and
the cross-process cache-epoch counter.

### Layer 1: Ingest

Responsibility: take a path to a PDF (or other supported format) and
produce all the artifacts the retrieval layer needs.

The pipeline is deterministic in the sense that re-running on the same
input produces the same output (modulo timestamps); it is *not* a stream
processor. Each document is processed in isolation, in a background worker
process. The worker is a separate OS process from the FastAPI server,
not a thread or asyncio task — this is the single most important
operational decision in the system. It enables three things at once:

- Graceful crash recovery (a worker crash does not bring down the
  serving layer)
- Independent resource budgeting (worker can grab GPU memory for vision
  encoding without affecting query latency)
- Maintenance pause without restart (server flips a SQLite flag; worker
  observes it before claiming the next item)

The ingest pipeline produces, per document:

1. **Text chunks** with parent-chunk references (chunk size and overlap
   are tuned by the chunker; see `docs/design/navigation-first.md` for
   the rationale behind chunk shape)
2. **Page-level dense + learned-sparse vectors** — dense from
   Qwen3-VL-Embedding (dim 4096), learned sparse from MILCO
   (doc-side pruning; a separate model from the dense embedder)
3. **Page-level visual embeddings** from Qwen3-VL — a single dense
   vector per page, in the same 4096-dim space as the text channel
   (not a multi-vector MAX-SIM model)
4. **Page-level LLM descriptions** from a vision LLM (only when the
   page contains visual content; pure-text pages skip this channel)
5. **Metadata rows** in SQLite: doc_id, owner, status, version,
   effective_date, expiry_date, supersedes, deleted_at
6. **NavIndex entries**: pointer-only structural index built from the
   parser's headings and page labels

### Layer 2: Storage

Three independent stores, each chosen for the access pattern of its
data:

| Store | Backing | Holds | Why this choice |
|---|---|---|---|
| **Vector index** | Qdrant (local file mode by default, server mode optional) | text chunks, vision pages, descriptions — three versioned collections | Best small-team vector DB: zero-config local mode, drop-in scale-up to server, named vectors per channel, good Python ergonomics |
| **Relational state** | SQLite (one file, multiple tables) | documents, users, audit_log, maintenance_state, cache_epoch, usage_feedback, ingestion_jobs | Two processes (server + worker) need shared state; SQLite is the only zero-deploy option that gives true cross-process consistency. WAL mode + careful BEGIN IMMEDIATE on the worker's claim path |
| **Image store** | Local filesystem under `image_storage_path/<doc_id>/page_N.png` | rendered page images for vision retrieval + agent fetch | Simple, debuggable, cheap; soft-delete leaves files in place until GC purges them |

The NavIndex deserves its own bullet: it is a separate SQLite database
(by default `nav_index.db`) holding pointer-only NavEntry rows. Each
NavEntry knows its document, page range, label, parent/child entries,
and `resource_uris` — but **never** holds source text or generated
descriptions. This separation is what makes "navigation, not retrieval"
the primary access pattern: the index can be served, walked, and
expanded without ever loading the heavy content channels.

### Layer 3: Retrieve

Retrieval is intentionally a small composition of orthogonal stages:

```text
query
  │
  ├─► (optional) HyDE expansion + Multi-Query (query rewrite adapter)
  │
  ├─► 4-channel parallel retrieval
  │     • text dense (Qwen3-VL-Embedding)
  │     • text sparse (MILCO learned sparse)
  │     • vision (Qwen3-VL single-vector dense)
  │     • description (Qwen3-VL-Embedding on LLM descriptions)
  │
  ├─► RRF fusion (rank-based, not score-based — different channels'
  │   score scales are not commensurable)
  │
  ├─► cross-encoder rerank (text-only candidates; vision-only
  │   hits pass through at RRF score)
  │
  ├─► NavIndex enrichment (each evidence hit gets pointers to the
  │   surrounding NavEntry — section, parent, siblings)
  │
  └─► evidence/hint partition (parsed text → evidence_fields;
                                LLM descriptions → hint_fields)
```

Two non-obvious choices worth flagging:

- **RRF over score-fusion.** A dense cosine of 0.78, a sparse BM25-ish
  score of 12.4, a multi-vector MAX-SIM of 3.1, and a description-dense
  of 0.65 have no common scale. Rank-based RRF is the cheapest correct
  answer; trained late-fusion would be better but adds a training
  dependency the small-team stage explicitly avoids.

- **Cross-encoder reranks text only.** Earlier versions of NaviKB fed
  LLM-generated descriptions to the cross-encoder along with parsed
  text. That double-counted an already-uncertain signal. The current
  rule: text_chunk candidates rerank; vision-only hits pass through
  at their RRF score; pure description hits are surfaced but flagged
  as `result_type: hint`.

Lazy initialization is the rule for any heavy model. The reranker and
the LLM adapter (HyDE, Multi-Query, agent-final) are constructed on
first use, not at startup; this keeps the server's cold start fast
(~5 seconds) and avoids loading GPU memory for paths the operator
doesn't use.

### Layer 4: Serve

The serve layer wraps the retrieval engine in three faces:

1. **FastAPI REST endpoints** — programmatic access for scripts and
   ingestion submission (`POST /ingestion/jobs`, `DELETE /documents/{id}`,
   `POST /documents/{id}/restore`, `POST /admin/maintenance_mode`)
2. **MCP sub-app** — first-class agent surface, mounted under `/mcp`,
   exposing tools (`kb_search`, `kb_hybrid_search`, `kb_search_nav`,
   `kb_expand_nav_entry`, etc.) and resources (`kb://documents/...`,
   `kb://nav/...`). The MCP layer is where evidence/hint partition and
   navigation prompts are most visible
3. **Background worker** — separate process consuming the ingestion
   queue with fairness-aware claim semantics (one user's burst doesn't
   starve another user's first item)

Cross-cutting concerns live across the serve layer:

- **Auth**: hard cutover to user tokens. Every request — REST or MCP —
  must present a valid API key issued by `manage_users.py create`.
  The server refuses to start if the users table is empty
  (bootstrap-refuse-start invariant)
- **Audit**: three semantic classes — *single-phase mandatory* for
  user-management ops, *two-phase* for delete and reassign, *best-effort*
  for everything else. Two-phase guarantees that the action and its
  `.completed` audit row both succeed, or the action's effect
  is reversible (e.g. soft delete with a `.attempted` row already on disk)
- **Maintenance**: SQLite-backed flag both processes read. Writers
  observe it before mutating; the admin endpoint waits for active
  worker claims to drain before reporting `drained:true`
- **Soft delete + GC**: DELETE never touches Qdrant or images; it
  flips status + writes `deleted_at`. Restore is a flag flip back.
  GC runs out-of-band and purges expired soft-deleted docs
- **Cache invalidation**: SQLite cache-epoch counter solves the
  cross-process staleness problem. Worker bumps after ingest /
  deprecate; server folds the epoch into its search cache key

The detailed treatment of auth, audit, maintenance, soft delete, and
GC will live in `docs/design/security-model.md`.

## Process topology

```text
┌──────────────────────────────┐         ┌──────────────────────────────┐
│  FastAPI tool_server          │         │  Background worker            │
│  - REST + MCP sub-app         │         │  - claims from ingestion_jobs │
│  - lazy-loads reranker, LLMs  │         │  - runs ingest pipeline       │
│  - reads cache_epoch          │         │  - bumps cache_epoch on done  │
└─────────┬─────────────────────┘         └─────────────┬────────────────┘
          │                                              │
          │            shared SQLite (kb_metadata.db)   │
          │   ┌───────────────────────────────────┐     │
          ├──►│ documents · users · audit_log     │◄────┤
          │   │ maintenance_state · cache_epoch   │     │
          │   │ ingestion_jobs · ingestion_items  │     │
          │   └───────────────────────────────────┘     │
          │                                              │
          │            shared Qdrant                    │
          │   ┌───────────────────────────────────┐     │
          ├──►│ text_chunks · vision_pages · desc │◄────┤
          │   └───────────────────────────────────┘     │
          │                                              │
          │            shared filesystem                │
          │   ┌───────────────────────────────────┐     │
          └──►│ image_storage_path/<doc_id>/...   │◄────┘
              └───────────────────────────────────┘
```

The two processes never communicate directly. All coordination flows
through SQLite. This is unfashionable in 2026 — the default reach is
for a message broker — but it is the right answer at small-team
scale because it eliminates a class of failure modes (broker is down,
broker is slow, broker mis-routes) at the price of "you cannot run
the worker on a different machine from the server." For a single-
host or LAN deployment, that price is zero.

## Failure modes the design anticipates

This list is honest about what can go wrong, not aspirational about
what won't.

| Failure | What happens | Mitigation |
|---|---|---|
| Worker crashes mid-ingest | item lease expires, another claim attempt picks it up | `claim_next_item` reclaim window; per-item attempt counter; backoff |
| Two workers race for the same item | one wins via SQLite BEGIN IMMEDIATE write-lock; the other sees the row as claimed and skips | explicit BEGIN IMMEDIATE on the worker's claim transaction |
| Search cache returns stale results after ingest | up to `search_cache_ttl_s` (default 300s) of staleness | cache-epoch counter: worker bumps on commit, server folds into key |
| Vector store and metadata diverge | doctor `reconcile` detects orphan vectors, orphan images, missing-vector active docs | `doctor.py` deep reconcile + `gc_deleted_docs.py` for residue |
| Audit DB unavailable during a delete | two-phase: `.attempted` row already on disk; main op proceeds; `.completed` fails into JSONL fallback or HTTP 503 to caller | JSONL fallback file; admin reconstructs trail |
| Server in maintenance, user submits a write | every write path (REST + MCP write tools + worker claim) checks the flag and refuses | `require_no_maintenance` dependency + worker `maintenance_check` callback |
| Backup taken mid-write | maintenance drain + SQLite online backup API (captures uncheckpointed WAL) | drain loop with 5-min cap; backup script uses `sqlite3.Connection.backup()`, not `shutil.copy` |

## What's deliberately out of scope

- **Multi-host deployment.** SQLite as cross-process coordinator
  caps us at one machine. Going multi-host would require swapping the
  coordination layer; we don't plan to.
- **LLM-driven schema/ontology induction.** The "graph" in NaviKB is
  the document's own headings, not a model's guess at entity edges.
- **Hosted SaaS.** We intend to stay local-first. Operators who want
  a managed service should look elsewhere.
- **Streaming ingest.** Each document is processed end-to-end before
  the next one starts. For thousands of small documents this would
  be wasteful; we accept that cost in exchange for failure-mode simplicity.

## Where to read next

- [Status & release plan](../status.md) — what's done, what's in
  testing, what's blocked
- [`navigation-first.md`](navigation-first.md) — why NavIndex is the
  way it is
- [`security-model.md`](security-model.md) — auth, audit, soft-delete
  contract
- [`comparison.md`](comparison.md) — explicit comparison with related
  projects
