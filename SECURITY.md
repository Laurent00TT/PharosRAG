# Security Policy

NaviKB is an **early-alpha research / example** project (see
[docs/status.md](docs/status.md)). The code is public, but it has not been
independently audited and is not yet recommended for production use. Please
read this before reporting anything security-related.

## What's in scope

Both the design and the code are public, and review of either is welcome:

- **The threat model** in [docs/design/security-model.md](docs/design/security-model.md)
  — what it covers, what it explicitly excludes, and whether the trade-offs
  are sound.
- **The auth contract** — token issuance, hashing (Argon2id), brute-force
  aggregation, the bootstrap-refuse-start invariant.
- **The audit model** — best-effort / single-phase / two-phase semantics and
  the JSONL fallback for two-phase completed writes.
- **The soft-delete + GC contract** — separation of the metadata flip from
  physical purge, retention semantics, restore expectations.
- **The maintenance / backup protocol** — drain semantics, SQLite online
  backup, the flag-reset-in-copy detail.
- **The evidence-vs-hint partition** in MCP responses.
- **Code-level issues** — auth bypass, injection, path traversal, unsafe
  deserialization, secrets handling, dependency vulnerabilities, and the like.

## How to report

For a **design-level** concern, open an issue using the
[security-design template](https://github.com/Laurent00TT/navikb/issues/new?template=security_design.yml)
(it auto-applies the `security-design` + `design-feedback` labels — non-maintainers
usually can't add labels themselves on a public repo, so the template does it).

For a **code-level vulnerability** you don't want to be world-readable, please do
**not** open a public issue. Use GitHub Private Vulnerability Reporting on this
repo if it is enabled, or email the maintainer (address on the GitHub profile).

## Coordinated disclosure

This is a small-team, early-alpha project — there is no formal SLA, but the
intent is:

- Acknowledge a confirmed report within a reasonable window.
- Target ~90 days from confirmed report to public disclosure, negotiable for
  actively-exploited issues.
- Credit researchers who responsibly disclose — opt-in; no names are added
  without permission.

As the project matures we plan to enable GitHub Private Vulnerability Reporting
and maintain a security-advisory feed so downstream operators can subscribe.

## Why no bug bounty

No money — and at small-team scale, bounties tend to produce a flood of
low-quality reports that drown out signal. The bar is quality, not volume.

## Anything else

For reports that aren't strictly security but you don't want to be
world-readable (e.g. Code of Conduct violations, or a sensitive
mischaracterization in `comparison.md`): the maintainer's email is on their
GitHub profile. GitHub public repos have no "private issue" feature, so email
(or Private Vulnerability Reporting) is the confidential channel.
