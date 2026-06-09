<div align="right">

**English** | [中文](CONTRIBUTING.zh.md)

</div>

# Contributing to NaviKB

NaviKB is an early-alpha **research / example** project (see
[status](docs/status.md)). The code is public and runs, but it is not a turnkey
product — so the most useful contributions reflect that stage.

## What's most useful right now

### 🐛 Reproducibility reports and bug fixes

The single most useful thing you can do is **set it up on a clean machine and
tell us what broke.** A clean-machine install has not been widely reproduced,
so an install report (what worked, what didn't, your OS / Python version) — or a
PR that fixes a break you hit — is hugely valuable. Open an issue, or send a
focused PR with a note on how you reproduced the problem.

### 🧪 Tests and documentation

PRs that add test coverage, fix a flaky or environment-dependent test, or
clarify the docs are welcome. For docs, suggest matching changes in both
language versions if you can (see Bilingual maintenance below).

### 💬 Design, comparison, and positioning feedback

The design is a real part of the contribution here. If you think a design
decision is wrong, or that `comparison.md` mischaracterizes a related project
(LlamaIndex, LangChain, GraphRAG, NaviRAG, Verba, …), open an issue with the
`design-feedback` label. We want the comparison accurate, not flattering to
NaviKB.

**A good design-feedback issue:**

```text
Title: [design-feedback] <section>: <one-line summary>
Document: docs/design/architecture.md (or whichever) + section
The assumption I'm questioning: <what the doc claims>
Why it's worth revisiting: <evidence, reference, alternative>
What would convince me either way: <data / argument that resolves it>
```

## Before a larger code change, open an issue first

For anything beyond a focused fix — a new feature, a refactor, a dependency
change — please open an issue to discuss it first. Two reasons:

- This is early-alpha; internals still move, and a big PR against a moving
  target is painful for everyone.
- Some things are **deliberately out of scope** (multi-host deployment,
  LLM-driven ontology induction, hosted SaaS — see the
  [architecture doc](docs/design/architecture.md)). Issues or PRs for those
  will be closed politely with a pointer. If you think an out-of-scope item
  should be in scope, that's a `design-feedback` discussion, not a feature
  request.

## "When will it be stable?"

The honest maturity and roadmap are in [docs/status.md](docs/status.md). It's
trigger-based, not date-based — every internal date we've promised has slipped.

## Bilingual maintenance

Every doc has an English (`<name>.md`) and Chinese (`<name>.zh.md`) version.
They are content-equivalent, not word-for-word translations. When you change
one, change the other in the same PR — or note that the mirror needs updating
and a maintainer will do the pass before merge.

## Code of Conduct

Participation in any NaviKB space (issues, discussions, PRs) is governed by the
[Contributor Covenant](CODE_OF_CONDUCT.md). The short version: assume good
faith, give technical specifics rather than personal characterizations, and
don't drag fights from other forums in here.

## Maintainer response time

This is currently a one-maintainer project. Issue response is best-effort,
typically within a week. For a security concern you don't want public, email
(address on the maintainer's GitHub profile) is faster than an issue — see
[SECURITY.md](SECURITY.md).
