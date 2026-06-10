<div align="right">

**English** | [中文](security-model.zh.md)

</div>

# Security & Operations Model

> The layer where most small-team RAG projects stop. NaviKB starts here.

## Why this document exists

A knowledge base is not just a search engine; it holds documents that
real people care about. Who can read what, who deleted what when, how
to take a consistent backup, how to recover from a botched ingest —
these are *operational* concerns, not "nice-to-haves." This document
describes how NaviKB handles them.

The audience is operators evaluating whether to trust NaviKB with
their documents, and contributors thinking about adding features that
touch read/write/delete paths.

## Threat model (current)

NaviKB is designed for a deliberately constrained threat model. Being
explicit about what it protects against — and what it doesn't —
matters more than claiming broad security.

### In scope

- **Untrusted MCP / REST clients on a trusted network.** Any client
  reaching the server must present a valid user token. There is no
  anonymous read path. There is no legacy fallback for unauthenticated
  requests.
- **Member-vs-admin privilege boundary.** Members can only modify
  documents they own. Admin-only operations (maintenance toggle,
  reassign, force-purge, include_deprecated search) are gated and
  refuse cleanly with HTTP 403.
- **Cross-process consistency.** Worker process and server process can
  both write to shared SQLite + Qdrant; coordination prevents silent
  stale reads (cache-epoch counter) and silent half-completed writes
  (two-phase audit, BEGIN IMMEDIATE on claim).
- **Audit non-repudiation for destructive actions.** Delete and
  reassign produce two-phase audit rows (.attempted + .completed) so
  even a crash mid-operation leaves a recoverable trail.
- **Brute-force-resistant auth.** Failed-auth attempts from one source
  IP are aggregated into `auth.brute_force` audit rows after a
  threshold; doctor surfaces these.
- **Soft-delete with retention.** A doc DELETE is a flag flip, not a
  physical purge. Restore is possible within retention. GC purges only
  after explicit retention expiry.

### Out of scope (deliberate)

- **Network-layer attackers.** NaviKB does not terminate TLS itself,
  does not implement reverse-proxy-style request shaping, and does
  not have its own WAF. Operators exposing it beyond `127.0.0.1`
  should put it behind a real reverse proxy (Caddy / nginx / Cloudflare
  Tunnel) with TLS + (optionally) mTLS or a higher-layer auth gate.
- **Compromised maintainer machine.** If the host running NaviKB is
  compromised, all bets are off: the attacker has the SQLite file with
  every user's token hash, every audit row, and every document's
  vector representation. We do not pretend to defend against this.
- **Side-channel attacks on the LLM.** Prompt injection from
  user-supplied document content is a real risk for any RAG system.
  NaviKB partitions evidence vs hint (see below) but does not claim
  to neutralize prompt injection — the safety header on resource
  payloads is a mitigation, not a solution.
- **Multi-tenant strong isolation.** NaviKB has owner_id-based
  permissions but does NOT implement per-tenant SQLite namespacing,
  per-tenant Qdrant collections, or per-tenant audit. The small-team
  shape is "trusted operators, one organizational unit." A SaaS
  multi-tenant deployment would need a different design.

## Auth: hard cutover to user tokens

There are two paths into the system: REST endpoints and the MCP sub-app
(mounted under `/mcp`). Both go through the same `authenticate_token`
function. There is no third path; there is no anonymous fallback.

A token is created by `manage_users.py create <username> --role
{admin|member}` and printed exactly once to stdout. The plaintext token
is hashed (Argon2-class) and stored in the `users` table. Subsequent
authentication is constant-time comparison of the request token's hash
against the stored hash.

### The bootstrap refuse-start invariant

The server fails to start (with a FATAL log line) if the users table
is empty. The reasoning: without at least one admin user, no one can
authenticate, but every endpoint would still 401, which looks like a
generic auth bug. Failing loud at startup makes the misconfiguration
obvious.

The first-time setup flow is therefore:

```text
1. Initialize the database (run any script that calls UsersStore.init)
2. python scripts/manage_users.py create <name> --role admin
3. Now the server can start
```

This is documented in the quickstart (not yet public) and is the most
common new-install failure mode once code ships.

### Token hash storage

Tokens are stored as Argon2id hashes, not plaintext. Argon2id parameters
default to OWASP-recommended values; the parameters are versioned in the
hash itself (Argon2's standard `$argon2id$v=19$m=...$t=...$p=...` prefix),
so a future parameter change can re-hash on next successful login without
breaking older tokens.

### Brute-force protection

`auth.failed` events are aggregated by source IP in an in-process
counter (the `BruteForceTracker`). When N failures within a window arrive
from the same IP, one `auth.brute_force` row is written to the audit log
(with the IP, count, and window) and the counter resets. This produces
one signal-rich audit row per attack burst instead of N low-signal
`auth.failed` rows.

Tracker state is in-memory only. A process restart loses the window,
which means an attacker gets at most "threshold − 1" additional attempts
before the next aggregation fires. We consider that an acceptable trade
for not introducing a Redis dependency.

## Audit: three semantic classes

The audit log is the most security-critical piece of state in NaviKB.
Every operation that touches documents or users flows through one of
three contracts.

### Class 1: best-effort

Action types: `auth.failed`, `auth.brute_force`, `authz.denied`,
`doc.ingest`, `job.cancel`, `doc.restore`, etc.

These call `audit_log.write(action, ...)` directly. The write goes to
the SQLite audit_log table. If the DB insert fails (transient lock,
disk full, etc.), an error trace is emitted and the failure is
swallowed — the main operation continues. The intent is to never let
audit failure break a non-destructive operation.

### Class 2: single-phase mandatory

Action types: `user.create`, `user.disable`, `user.role_change`,
`user.key_reset`.

These are user-management operations where the audit row is part of
the operation's correctness, not an incidental side-effect. They use the
`make_record()` factory + the caller's own transactional session, so
the user table change and the audit row commit (or roll back) together.
If audit cannot be written, the user operation also fails — there is
no "user created but no audit row" possibility.

### Class 3: two-phase

Action types: `doc.delete`, `doc.reassign`.

These are destructive document operations where the audit trail must
survive even if the operation fails mid-way. The contract:

```text
1. audit_log.write_attempted("doc.delete", target_id=..., ...)
     → writes a doc.delete.attempted row, best-effort
2. main op runs (e.g. mark_deleted + cache invalidate)
3. audit_log.write_completed("doc.delete", ...)
     → writes a doc.delete.completed row, MANDATORY
        - tries SQLite insert first
        - on failure, appends to a JSONL fallback file with fsync
        - if BOTH fail, raises AuditWriteFailure
          → caller's HTTP handler maps to 503 with a documented
            "operation completed but audit lost" detail message
            for the admin to manually reconcile
```

This means:

- If `.attempted` is on disk but `.completed` is not, the operation
  may or may not have happened — admin investigates
- If both `.attempted` and `.completed` are on disk, the operation
  succeeded with a trail
- If neither is on disk, the operation never started (or failed
  before even the best-effort attempt)

The JSONL fallback survives a SQLite DB crash, and the fsync after
each line means even a power loss preserves at least one terminal
state on disk.

### What two-phase explicitly does NOT do

Two-phase does NOT make the operation atomic. There is a window
between `.attempted` and `.completed` where a crash leaves an
"attempted but unknown" state. The contract is honest: the audit log
will tell you "we tried to delete D-42 at time T" — the admin then
reconciles by looking at the actual state of D-42.

This is the right trade for a small-team stack. A truly atomic
implementation would require either two-phase commit across SQLite +
Qdrant (no SQLite-supported way to do this cleanly) or a write-ahead
log of intended ops (which is what the JSONL fallback approximates).

## Soft delete: the core of restore + GC

NaviKB's DELETE is a pure flag flip:

```python
# What DELETE /documents/{id} does:
1. require_owner_or_admin(doc, user)            # 403 if wrong caller
2. write_attempted("doc.delete", ...)           # best-effort phase 1
3. meta_db.mark_deleted(doc_id)                 # status=deleted, deleted_at=now
4. invalidate_all_search_caches()               # local + cross-process bump
5. emit("document.soft_deleted", ...)           # trace event
6. write_completed("doc.delete", ...)           # mandatory phase 2
```

**Critically, DELETE does not touch Qdrant or image storage.** Vectors
stay. Images stay. The active-gate (status != active → not retrieved,
not exposed via GET) makes the doc invisible to the read path; it does
not erase the underlying bytes.

### Why this matters

This is the only design that makes restore meaningful. If DELETE
purged vectors immediately, "restore" would be a metadata-only flip
that left no content for the agent to retrieve. By keeping content
intact during retention, restore is a real recovery operation:

```python
# What POST /documents/{id}/restore does:
1. require_owner_or_admin(doc, user)
2. meta_db.restore_document(doc_id)             # status=active, deleted_at=NULL
3. invalidate_all_search_caches()
4. audit_log.write("doc.restore", ...)          # best-effort, single row
```

The doc is searchable again immediately. The vectors, images, and nav
entries were never deleted.

### GC: the eventual hard delete

`scripts/gc_deleted_docs.py` is the script that actually purges. It:

1. Refuses to run if the server reports maintenance ON (avoids racing
   the backup script)
2. Scans `documents` where `status='deleted'`
3. Skips rows where `deleted_at IS NULL` (pre-T5 data — no truth source
   to compute retention) unless `--force-purge` is explicitly passed
4. Skips rows where `(now - deleted_at).days < retention_days` (default 30)
5. For each candidate, in order:
   1. Delete Qdrant vectors by `doc_id`
   2. Delete image files under `image_storage_path/<doc_id>/`
   3. Delete NavIndex entries for `doc_id`
   4. DELETE the metadata row

The order is "soft to hard" — Qdrant first, metadata last — so a
partial failure leaves a state the next GC run can recover from rather
than orphaning data.

GC is operator-scheduled, not automatic. The reasoning is in
[architecture.md](architecture.md#failure-modes-the-design-anticipates):
silent automatic purging is one of the most dangerous defaults a
knowledge base can have. Operators explicitly schedule GC via cron /
systemd timer / Task Scheduler.

## Maintenance: the cross-process pause

The maintenance flag is a single-row SQLite table (`maintenance_state`).
Both server and worker read it. When `on=1`:

- The server's write endpoints return HTTP 503 (`require_no_maintenance`
  FastAPI dependency)
- The worker's `claim_next_item` returns None (worker idles between
  poll intervals)
- The MCP write tools refuse to execute (return
  `reason: write_tools_disabled`)
- The admin endpoint itself **stays available** (otherwise you couldn't
  turn maintenance off)
- The read paths (GET endpoints, MCP read tools) stay available

Flipping ON triggers a drain loop in the admin endpoint: it waits up to
5 minutes for active worker claims (non-expired lease) to finish, then
returns `{on: true, drained: true|false}`. The 5-minute cap exists so
the HTTP request doesn't hang forever; if `drained: false`, the backup
script decides whether to proceed (default: refuse; pass `--best-effort`
to override).

### Why a SQLite flag and not an asyncio Event

In-process signals don't cross the worker-server boundary. The worker
is a separate OS process; an asyncio.Event in the server can never
reach it. Every cross-process state in NaviKB lives in SQLite for this
reason.

The performance cost is one indexed PK SELECT per write request — sub-
millisecond at small-team write rates. We measured this carefully and
it is invisible in latency profiles.

## Cache-epoch: the cross-process invalidation primitive

The search cache is a per-process TTLCache (default 5-minute TTL). The
worker can invalidate its own cache; the server can invalidate its
own. Neither can directly invalidate the other.

The fix: a single-row SQLite `cache_epoch` table that both processes
write to. The server folds the current epoch value into its cache key.
The worker bumps the epoch after every ingest / deprecate. When the
epoch changes, every old cache key becomes stale and the next request
misses and recomputes.

Concretely:

```python
# Server search path
epoch = await cache_epoch_store.get_epoch()    # ~0.1ms PK lookup
key = cache_key(query, ..., epoch=epoch)
cached = local_cache.get(key)
if cached is not None: return cached
# ... recompute ...

# Worker ingest path (after successful commit)
await cache_epoch_store.bump()                 # UPDATE epoch = epoch + 1
```

The cost on the read path is one extra SQLite SELECT per search.
Cheap and worth it.

## Backup: the consistency primitive

Operator-driven backup uses `scripts/backup_kb.py`. The protocol:

```text
1. POST /admin/maintenance_mode {on: true}     # wait up to 5min for drain
2. For each KB SQLite DB: sqlite3.Connection.backup()
                                              # online backup API,
                                              # captures uncheckpointed WAL
3. shutil.copytree(qdrant_path, dest/qdrant,
                   ignore=*.db, *.db-wal, *.db-shm, *.lock, *.jsonl)
                                              # Qdrant's own files,
                                              # excluding our SQLite + locks
4. shutil.copytree(image_storage_path, dest/images)
5. Open the COPIED kb_metadata.db and reset maintenance_state.on=0
                                              # so restoring the backup
                                              # doesn't come up in maintenance
6. Write manifest.json (ts, schema_version, sha256 per file)
7. POST /admin/maintenance_mode {on: false}   # always (finally block)
```

**Critical details that took iterations to get right:**

- `shutil.copy` on a WAL-mode SQLite file misses uncheckpointed pages
  in the `-wal` sidecar. The SQLite `Connection.backup()` API is the
  only way to get a consistent online snapshot without checkpointing.
- The qdrant_path directory contains both our SQLite databases AND
  Qdrant's own state. `shutil.copytree(ignore=...)` excludes the
  former so they are not inadvertently copied twice and corrupted.
- The backup contains a SQLite snapshot taken WHILE maintenance was
  on. If we didn't reset the flag in the copy, the restored backup
  would come up locked. Step 5 is non-obvious but necessary.
- `--strict` is the default, not opt-in. A backup that silently
  proceeded with `drained: false` is a mid-write snapshot. Operators
  must explicitly pass `--best-effort` to accept that risk.

## Evidence vs Hint: the citation contract

This is a security model concern, not just a data-model one. An LLM-
generated description of a page is *not* evidence — citing it as if it
were source text is a hallucination amplifier.

NaviKB partitions every response that the agent can see:

- `evidence_fields`: parsed text, image URLs, figure captions, figure
  indices — content that the parser extracted from the source document
- `hint_fields`: `generated_description` — content the LLM wrote
  about the page

The MCP `evidence_response_to_mcp_payload` formatter labels each result
in the text channel:

- Has `text_chunk` → `result_type: evidence`, labeled "Text preview
  (evidence)"
- Only `vision_description` → `result_type: hint`, labeled
  "Generated description (RECALL HINT only — not evidence; fetch the
  page resource to cite)"

The agent prompt is tuned to cite only from `evidence_fields`. If an
agent ignores the partition and cites a hint, that's a downstream
failure mode — but the contract is enforced at the API surface so the
failure is visible.

## Doctor: the operational ground truth

`scripts/doctor.py` is how operators ask "is my system actually
consistent?" It runs without writes (it explicitly opens the job store
in `readonly=True` mode to avoid taking the BEGIN IMMEDIATE write lock):

- **Single-host check** (default): config validity, storage paths,
  Qdrant collection presence, HTTP service reachability
- **Deep reconcile** (`--no-reconcile` to skip):
  - Every active document has at least one text vector in Qdrant
  - Every Qdrant `doc_id` has a corresponding metadata row
  - Every active document has an image directory
  - No soft-deleted document has lingering vectors (pending GC)
  - No image directory exists without a metadata row
- **Team view** (`team` subcommand): per-user 24h activity, active
  workers, queue depth by owner, recent destructive actions,
  soft-deleted inventory with retention countdown, orphan owners,
  maintenance status

Findings are categorized OK / WARN / ERROR with actionable messages.
ERROR findings include specific doc_ids and suggested remediation
commands (e.g. "run python scripts/gc_deleted_docs.py").

## What this layer does NOT solve

- **Network exposure.** Use a reverse proxy and TLS. NaviKB is
  designed to be locally trusted, not internet-facing.
- **Document-level secrets.** Every document is owned by exactly one
  user (or NULL = admin-only); we do not have row-level redaction or
  field-level encryption. Don't put national-security documents in
  NaviKB.
- **Insider-threat audit tamper detection.** The audit log is in
  SQLite; an admin with shell access can `UPDATE audit_log SET ...`
  and we won't detect it. Append-only blockchain-style audit is out
  of scope for the small-team stack.
- **Compliance certifications.** No SOC 2, no HIPAA, no ISO 27001.
  The audit trail is designed to support those if you wanted to build
  toward them, but we make no claims.

## Where to read next

- [`architecture.md`](architecture.md) — the four-layer architecture
  this security model lives in
- [`navigation-first.md`](navigation-first.md) — the primary access
  pattern that auth, audit, and the evidence/hint partition support
- [`comparison.md`](comparison.md) — how this operational depth
  compares to typical RAG frameworks
