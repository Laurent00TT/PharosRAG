<div align="right">

**English** | [中文](CONTRIBUTING.zh.md)

</div>

# Contributing to NaviKB

NaviKB is in [Stage 0: design preview](docs/status.md). This means
contributions look different from a typical open-source project right
now. This document describes what's welcome, what isn't, and why.

## What we welcome right now

### ✅ Design feedback

The design documents in `docs/design/` are the actual deliverable at
this stage. Reading them carefully and pushing back on assumptions is
the most valuable contribution you can make.

**How:** open an issue with the `design-feedback` label. Reference the
specific document and section (`docs/design/architecture.md §4 Serve`),
state the assumption you're questioning, and what you think the
alternative looks like.

**Examples of good feedback:**

- "The four-channel retrieval description says you use RRF over four
  ranks. Have you considered Sparse-Dense Fusion ([reference paper]),
  which would let you train a single late-fusion weight per channel?"
- "The cross-process cache-epoch counter assumes both processes can
  write to the same SQLite file. In a Docker Compose deployment with
  the worker on a separate container but a shared volume, this works.
  In a Kubernetes deployment with persistent volumes, the lock semantics
  are untested. Worth a note in the architecture doc."
- "The status doc claims 'navigation-first' is differentiating. But
  haystack and llamaindex both have document-tree retrievers. Worth
  comparing against those directly in `comparison.md`."

### ✅ Comparison + positioning input

If you maintain or use a related project (LlamaIndex, LangChain RAG,
GraphRAG, NaviRAG, Verba, etc.) and you think our `comparison.md`
mischaracterizes it, open an issue. We want the comparison to be
accurate, not flattering to NaviKB.

### ✅ Documentation typos / clarity issues

Open a PR. These will be merged. Suggest matching changes for both
language versions if you can; if you only write one, mention it in the
PR and we'll mirror.

## What we don't accept yet

### ❌ Code PRs against (non-existent) implementation

The code isn't in this repository (see [docs/status.md](docs/status.md)
for why). PRs adding `src/`, `tests/`, `scripts/` etc. will be closed
with a pointer to this document. This isn't a judgment on the PR; it's
that merging code now would let us announce things we haven't actually
tested second-person, which is exactly the bar we set.

### ❌ Feature requests that imply changing the project's scope

We have explicit out-of-scope items (multi-host deployment, LLM-driven
ontology induction, hosted SaaS — see [architecture.md §What's
deliberately out of scope](docs/design/architecture.md#whats-deliberately-out-of-scope)).
Issues asking for those will be closed, politely, with a pointer to
that list.

If you think one of the out-of-scope items should actually be in scope,
that's a design discussion — open a `design-feedback` issue, not a
feature request.

### ❌ "When will this ship?"

The release plan is in [docs/status.md](docs/status.md). It is
trigger-based, not date-based. Asking for a calendar date won't get
one, because every calendar date we've internally promised has slipped.

## How to open a good design-feedback issue

```
Title: [design-feedback] §<section>: <one-line summary>

Document: docs/design/architecture.md (or whichever)
Section: §3 Retrieve
Specific paragraph: "RRF over score-fusion..."

The assumption I'm questioning:
  <your understanding of what the doc claims>

Why I think it's worth revisiting:
  <evidence, reference, alternative approach>

What would convince me either way:
  <what data / argument would resolve this>
```

We commit to reading every issue. We don't commit to changing the
design just because feedback was given — the design has to actually
improve, not just compromise.

## Bilingual maintenance

Every doc has an English (`<name>.md`) and Chinese (`<name>.zh.md`)
version. They are expected to be content-equivalent, not
word-for-word translations.

When you change one, change the other in the same PR. If you can only
write in one language, write only the English version and note in the
PR description that the Chinese mirror needs updating — a maintainer
will do the Chinese pass before merge.

A PR that updates only one language without acknowledging the other
will be asked to either complete the mirror or downgrade the PR's
scope.

## Code of Conduct

Participation in any NaviKB space (issues, discussions, future PRs)
is governed by the [Contributor Covenant](CODE_OF_CONDUCT.md). The
short version: assume good faith, give technical specifics rather than
personal characterizations, and don't drag fights from other forums
into here.

## Maintainer response time

This is currently a one-maintainer project. Issue response is best-
effort, typically within a week. If something is genuinely urgent
(security concern about the design as documented — yes, even though
the code isn't out yet) email is faster than an issue. Email address
will appear on the maintainer's GitHub profile.
