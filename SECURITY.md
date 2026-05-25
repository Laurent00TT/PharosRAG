# Security Policy

NaviKB is currently in [Stage 0: design preview](docs/status.md).
This means our security posture differs from a typical released
open-source project. Please read this document before reporting
anything security-related.

## What's in scope right now

At Stage 0, the only attack surface NaviKB exposes publicly is its
**design**. We welcome security review at that layer:

- The threat model in [docs/design/security-model.md](docs/design/security-model.md)
  — what it covers, what it explicitly excludes, and whether you
  think the trade-offs are sound
- The auth contract — token issuance, hashing (Argon2id), brute-force
  aggregation, bootstrap-refuse-start invariant
- The audit model — best-effort / single-phase / two-phase semantics
  and the JSONL fallback for two-phase completed writes
- The soft-delete + GC contract — separation of metadata flip from
  physical purge, retention semantics, restore expectations
- The maintenance / backup protocol — drain semantics, SQLite online
  backup, the flag-reset-in-copy detail
- The evidence-vs-hint partition in MCP responses

If you find a hole in any of the above as described, please open an
issue using the [security-design template](https://github.com/Laurent00TT/navikb/issues/new?template=security_design.yml).
The template auto-applies both `security-design` and `design-feedback`
labels — non-maintainers usually can't add labels themselves on a
public repo, so the template handles it for you.

For reports you don't want to be world-readable, email the maintainer
instead (address on the GitHub profile); see the bottom of this
document for the full reasoning.

## What's NOT in scope until Stage 2

Because the code is not public yet, **no code-level vulnerability
report is meaningful at this stage**. We cannot accept reports of the
form "I found CVE-X in your implementation" because there is no
implementation here to find a CVE in.

When NaviKB reaches Stage 2 (public open source), this document will
be updated with:

- A formal vulnerability disclosure address (`security@` or GitHub
  Private Vulnerability Reporting once enabled)
- Supported version policy
- Coordinated disclosure timeline
- Public PGP key

Until then, the answer to "where do I report a code vulnerability" is:
there is no code; please come back at Stage 2.

## What we will do at Stage 2

The plan, documented now so it's not invented at release time:

1. **Enable GitHub Private Vulnerability Reporting** on the repo,
   which gives researchers a private channel that doesn't require
   us to maintain a separate inbox
2. **Publish a SECURITY.md update** with the exact coordinated-
   disclosure window (currently planning 90 days from confirmed
   report to public disclosure, negotiable for actively-exploited
   issues)
3. **Maintain a security advisory feed** via GitHub Security
   Advisories so downstream operators can subscribe

## Why we're not running a bug bounty

No money. Also: at small-team scale, bounties tend to produce a flood
of low-quality reports that drown signal. We're keeping the bar
quality-not-volume — design feedback now, and at Stage 2, real
vulnerability reports through Private Vulnerability Reporting.

## Reporting channel for *anything else*

Reports that aren't strictly security but you don't want to be
world-readable (e.g. Code of Conduct violations, sensitive
mischaracterization in `comparison.md`): the maintainer's email
address is listed on their GitHub profile. GitHub public repos do
not have a "private issue" feature; email is the only confidential
channel until Private Vulnerability Reporting is enabled.

## Acknowledgements

When Stage 2 lands, we'll add an acknowledgements section here
listing researchers who responsibly disclosed issues. The list will
be opt-in; we won't add names without permission.
