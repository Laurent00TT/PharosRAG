<div align="right">

**English** | [中文](navigation-first.zh.md)

</div>

# Navigation-First

> The single most important design decision in NaviKB. Everything else
> in the architecture is downstream of this.

## The position in one paragraph

A document is not a flat sequence of chunks; it is a navigable
structure. The reader (human or agent) cares about *getting to the
right part* before they care about *retrieving the right text from
that part*. Most RAG stacks invert this — they retrieve text first and
recover structure later, if at all. NaviKB starts from structure:
indexes it as pointers, exposes it to agents as a first-class tool
surface, and treats retrieval as the cheap follow-up once navigation
has narrowed the candidate space.

This document explains why we made that choice, what the data
structures look like, where the design pays off, and where it has
known limitations.

## The problem with chunk-first retrieval

Take a 200-page operations manual. Ask "what's the escalation procedure
for a Sev-1 incident?" In a typical RAG pipeline:

1. The query is embedded with a sentence-transformer
2. Cosine similarity against a vector store of ~2000 chunks (each
   ~300 tokens) returns the top-5
3. The top-5 chunks are concatenated, prefixed with a "use these as
   context" instruction, and sent to an LLM

This works often enough that it became the default. But the failure
modes it hides matter:

- **Loss of locality.** Chunk N and Chunk N+1 may be physically
  adjacent in the manual but score very differently against the query.
  The reader sees a Frankenstein context that mixes chapter 7 with
  chapter 12.
- **Loss of section-level intent.** A heading like "5.3 Escalation"
  is a strong signal — stronger than any individual sentence under it.
  Heading text often doesn't survive chunking, or survives only as a
  prefix that gets diluted.
- **Loss of relational structure.** The reader's question may actually
  be best answered by *the next section* — "5.4 Post-incident review"
  — which a flat retriever has no way to know about.
- **No affordance for the agent to ask "what else is in this chapter?"**
  Because the index doesn't expose chapters as a concept.

These aren't bugs in any specific implementation. They're consequences
of the data model.

## The position-first alternative

NaviKB indexes the structure of each document separately from its
content. The structural index — `NavIndex` — stores one row per
**NavEntry**, where a NavEntry is "a structurally meaningful unit
the reader can land on": a section, a sub-section, a table, a figure,
a page range.

```python
@dataclass
class NavEntry:
    entry_id: str                    # stable id
    label: str                       # "5.3 Escalation procedure"
    normalized_label: str            # "5.3 escalation procedure"
    entry_type: str                  # "section" | "table" | "figure" | "page_range"
    source: str                      # "parser_heading" | "manual_alias" | ...
    doc_id: str
    doc_name: str
    page_start: int
    page_end: int
    order_index: int                 # serialization order within the doc
    parent_entry_id: str | None      # for the section tree
    resource_uris: list[str]         # ["kb://documents/D-42/pages/87", ...]
    source_ref_ids: list[str]        # for traceability into the parser output
    aliases: list[str]               # human-curated alt labels
```

**Critical observation:** there is no `text` field. There is no
`embedding`. There is no `description`. A NavEntry is *pure pointer*.
It tells you where to go; it does not tell you what's there.

This is the architectural commitment. Once you accept that NavIndex
holds only pointers, several things follow:

- The index is small (hundreds of KB per typical document) and can be
  loaded into memory entirely
- An agent can walk it without exhausting any token budget
- Updating the index after a doc change is fast (rebuild just the
  affected entries, no re-embedding)
- Search latency is bounded by the count of entries, not the size of
  the corpus

## What the agent sees

NaviKB's MCP surface exposes navigation as first-class tools, distinct
from retrieval:

| Tool | Returns | When to use |
|---|---|---|
| `kb_search_nav(query)` | NavEntry list (pointers only) | When the question maps to a section / chapter / table — "where do they talk about X?" |
| `kb_expand_nav_entry(entry_id)` | NavEntry + its parent + children + siblings | When you've located the right entry and want the surrounding structure — "what else is in this chapter?" |
| `kb_get_toc(doc_id)` | Full NavEntry tree for one document | When you want to browse a doc's structure before deciding what to fetch |
| `kb_search(query)` | Evidence hits + their text + cite IDs | Direct evidence retrieval — when you already know roughly what text you want |
| `kb_hybrid_search(query)` | Evidence hits + suggested NavEntry fetches | The common path — retrieve evidence AND surface navigation hints in the same call |

The MCP `kb_answer_from_originals` prompt explicitly trains the agent
to use `kb_hybrid_search` first, then *fetch the original resource via
URI* before quoting. The pointer-first design makes this workflow
natural: the agent always has both "where" and "what" available.

## Retrieval is still important — but it's the second step

Navigation-first does not mean "no retrieval." The 4-channel retrieval
stack (text dense, text sparse, vision, description) is still the
backbone for the cases navigation can't reach by structure alone:

- Questions phrased as paraphrases of body text, where no heading
  matches
- Cross-document queries where the answer is in section 5.3 of doc A
  *and* the figure on page 87 of doc B
- Questions that surface relevant content the user didn't know existed

The choice is about *priority* and *first response*. When both
channels have something to say, NaviKB returns:

```text
evidence (cite-able):           N text/vision chunks with cross-encoder rerank
suggested_fetches (navigate):   M NavEntry pointers covering surrounding structure
```

The agent decides whether to cite directly or to navigate first.

## Why this is hard to do well

Three things have to be true for navigation-first to actually beat
chunk-first:

### 1. The parser has to produce real structure

If the parser only emits "page N text" without headings, table
boundaries, or figure references, NavIndex degenerates to one entry
per page — which carries less signal than chunking would. NaviKB
relies on MinerU (cloud API or local) for parse quality; the
implementation supports degraded fallback (page-as-entry) for parsers
that don't expose structure, but the design assumes a real parser is
available.

### 2. The label has to be meaningful enough to match queries

"5.3 Escalation procedure" matches the query "escalation" trivially.
"Section 5.3" does not. Real-world docs have a mix; NaviKB allows
**aliases** (manual labels added via `kb_add_nav_alias`) to repair
specific high-traffic entries that the parser labeled badly. Aliases
are first-class NavEntry fields, persisted, audit-logged, and
searchable through `kb_search_nav`.

### 3. The agent has to actually use navigation

A naive agent that ignores `kb_search_nav` and only calls `kb_search`
gets the chunk-first behavior with extra steps. NaviKB's MCP prompt
(`kb_answer_from_originals`) and tool descriptions are tuned to push
the agent toward navigation first. We treat the prompt engineering as
part of the system, not as something downstream callers should
re-invent.

## What NavIndex deliberately doesn't do

To keep the design honest, here's what NavIndex is *not* trying to be:

- **Not a knowledge graph.** No entity extraction. No relation
  inference. The "edges" are parent/child/next/prev in the document's
  own structure, plus manual aliases. We don't believe LLM-extracted
  edges are reliable enough to replace this at small-team scale.
- **Not a search engine.** Lexical substring + alias match on
  `normalized_label`. That's it. If you want semantic similarity,
  you've left navigation and entered the retrieval layer — that's
  fine, it's the same query call, but the contract is different.
- **Not stored alongside content.** NavIndex is a separate SQLite
  database (`nav_index.db`). This is deliberate: a heavy ingest can
  rebuild content channels without touching the NavIndex, and a quick
  manual edit (alias, hide) can update NavIndex without touching
  Qdrant.
- **Not generated by an LLM.** Aliases are explicitly human-curated
  (or extracted by the parser). We do not let an LLM hallucinate a
  hierarchy "for the user's convenience."

## Trade-offs we accept

| What we give up | What we get instead |
|---|---|
| Cannot answer well when documents have no real structure (one giant blob of text) | Excellent behavior on well-structured docs, which is most operational manuals, technical specs, legal filings, papers, and books |
| Cannot infer relationships the document doesn't have (no "topic" clusters across docs) | No risk of LLM-hallucinated relationships poisoning the navigation surface |
| Higher cost per document during ingest (must run a structural parser) | Lower cost per query during serving (small index, no LLM in the navigate path) |
| Operator must occasionally curate aliases for high-traffic queries | Curation IS the long-term quality lever — eval results improve over time without retraining |

## Comparison with adjacent ideas

This is summarized here for context; full comparison lives in
[`comparison.md`](comparison.md).

- **vs hierarchical retrievers in LlamaIndex / Haystack:** those
  retrieve through the hierarchy but still embed each chunk. NaviKB
  keeps the hierarchy pointer-only and lets retrieval be a separate
  composable.
- **vs GraphRAG:** GraphRAG builds a graph of LLM-extracted entities
  and relations. NaviKB uses the document's own structural graph.
  Different problems, different trust models.
- **vs NaviRAG / "Don't Retrieve, Navigate":** these are 2026 research
  papers exploring the same direction with different mechanisms (LLM-
  distilled hierarchies, agent skills). NaviKB's contribution is on
  the operational side: a navigation-first system you can run
  yourself today, with a real auth/audit/backup story.

## Where this design might be wrong

Two scenarios that would falsify the navigation-first thesis:

1. **The corpus has no structure to navigate.** If you're indexing a
   dump of unstructured emails, raw text scrapes, or conversation
   transcripts without speakers or timestamps, NavIndex degenerates
   and you lose to a well-tuned chunk-first system.
2. **The agent can't reason about navigation.** If your downstream LLM
   is small / unreliable enough that giving it "here's a list of
   pointers" produces worse outcomes than "here's a wall of text,"
   navigation-first hurts. We have not measured the lower bound on
   model capability rigorously yet — this is one of the items in
   [docs/status.md](../status.md) blocking public release.

If either of these is your situation, NaviKB is the wrong tool. We
will say so honestly in the documentation rather than try to retrofit
the design to cover them.

## Where to read next

- [`architecture.md`](architecture.md) — how the navigation-first
  decision shapes the rest of the stack
- [`security-model.md`](security-model.md) — how alias curation,
  hiding, and rebuilding interact with auth and audit
- [`comparison.md`](comparison.md) — explicit positioning against
  named alternatives
