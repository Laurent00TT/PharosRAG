"""Admin CLI for KB user management.

Reads DB URL from env KB_USERS_DB_URL (test-friendly). In production
the server lifespan will create the schema; this CLI assumes it's
already initialized.

Subcommands:
  create <username> --role {member|admin}     # generate key, print ONCE
  list [--all]                                 # show users (--all includes disabled)
  disable <username>                           # soft-disable
  set-role <username> --role {member|admin}   # promote / demote
  reset-key <username>                         # rotate key, print ONCE
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from kb.audit.log import AuditLog
from kb.auth.users import UsersStore
from kb.config import Settings


def _db_url() -> str:
    url = os.environ.get("KB_USERS_DB_URL")
    if not url:
        # Default: same SQLite file as MetadataDB (per spec P1-2: users
        # lives in kb_metadata.db). Production server lifespan will pass
        # this through; for ad-hoc admin use, point KB_USERS_DB_URL at
        # the prod metadata DB.
        s = Settings()
        url = f"sqlite+aiosqlite:///{s.qdrant_path}/kb_metadata.db"
    return url


async def cmd_create(args: argparse.Namespace) -> int:
    db_url = _db_url()
    store = UsersStore(db_url=db_url)
    audit = AuditLog(db_url=db_url, fallback_path=Settings().audit_fallback_path)
    try:
        await store.init()
        await audit.init()
        try:
            plaintext, user = await store.create_user(
                username=args.username, role=args.role,
                audit_log=audit, acting_user_id="_system",
            )
        except ValueError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
    finally:
        await audit.aclose()
        await store.aclose()
    print(f"Created user {user.username!r} (user_id={user.user_id[:8]}…, role={user.role})")
    print()
    print("API key (save this — shown once, only sha256 hash is stored):")
    print(f"  {plaintext}")
    print()
    print("Hand this key to the user via a secure channel (1Password / encrypted IM).")
    print("If lost, run:  manage_users.py reset-key " + user.username)
    return 0


async def cmd_list(args: argparse.Namespace) -> int:
    store = UsersStore(db_url=_db_url())
    await store.init()
    try:
        users = await store.list_users(include_disabled=args.all)
    finally:
        await store.aclose()
    if not users:
        print("(no users)")
        return 0
    # Plain ASCII table — small team, no need for rich
    print(f"{'username':<16} {'role':<7} {'key_prefix':<28} {'created':<20} {'disabled':<20}")
    print("-" * 95)
    for u in users:
        disabled = u.disabled_at.isoformat(timespec="seconds") if u.disabled_at else "-"
        print(
            f"{u.username:<16} {u.role:<7} {u.key_prefix:<28} "
            f"{u.created_at.isoformat(timespec='seconds'):<20} {disabled:<20}"
        )
    return 0


async def cmd_reassign(args: argparse.Namespace) -> int:
    """admin: reassign all documents owned by --from to --to.

    Writes two-phase audit per document. Historical audit rows are NOT
    modified — only new reassign rows added (内部设计说明（未公开）: 原 audit 不变)."""
    from kb.storage.metadata_db import MetadataDB, DocumentRecord
    from sqlalchemy import select, update as sa_update

    db_url = _db_url()
    users = UsersStore(db_url=db_url)
    meta = MetadataDB(db_url=db_url)
    audit = AuditLog(db_url=db_url, fallback_path=Settings().audit_fallback_path)
    try:
        await users.init()
        await meta.init()
        await audit.init()

        src_user = await users.get_by_username(args.from_username)
        if src_user is None:
            print(f"ERROR: --from user {args.from_username!r} not found", file=sys.stderr)
            return 1
        dst_user = await users.get_by_username(args.to_username)
        if dst_user is None:
            print(f"ERROR: --to user {args.to_username!r} not found", file=sys.stderr)
            return 1

        # Find all docs to reassign
        async with meta._session_factory() as session:
            result = await session.execute(
                select(DocumentRecord.doc_id).where(DocumentRecord.owner_id == src_user.user_id)
            )
            doc_ids = [row[0] for row in result.all()]

        if not doc_ids:
            print(f"No documents owned by {args.from_username!r}; nothing to do.")
            return 0

        print(f"Reassigning {len(doc_ids)} document(s) from {args.from_username!r} to {args.to_username!r}...")

        for did in doc_ids:
            # Phase 1: attempted
            await audit.write_attempted(
                "doc.reassign",
                user_id="_system",  # T5 follow-up: this should be the admin user_id
                target_kind="document", target_id=did,
                payload={"from_user_id": src_user.user_id, "to_user_id": dst_user.user_id},
            )
            # Main op
            async with meta._session_factory() as session:
                await session.execute(
                    sa_update(DocumentRecord)
                    .where(DocumentRecord.doc_id == did)
                    .values(owner_id=dst_user.user_id)
                )
                await session.commit()
            # Phase 2: completed
            await audit.write_completed(
                "doc.reassign",
                user_id="_system",
                target_kind="document", target_id=did,
                payload={"from_user_id": src_user.user_id, "to_user_id": dst_user.user_id},
            )
            print(f"  - {did}: {args.from_username} -> {args.to_username}")
    finally:
        await audit.aclose()
        await meta.aclose()
        await users.aclose()
    return 0


async def cmd_disable(args: argparse.Namespace) -> int:
    # T5: --reassign-to <username> chains a reassign before disabling.
    # Reassign FIRST so alice still owns the docs at the moment of
    # transfer (audit attributes correctly), then disable her after.
    reassign_to = getattr(args, "reassign_to", None)
    if reassign_to:
        sub_args = type("X", (), dict(
            from_username=args.username, to_username=reassign_to,
        ))()
        rc = await cmd_reassign(sub_args)
        if rc != 0:
            return rc

    db_url = _db_url()
    store = UsersStore(db_url=db_url)
    audit = AuditLog(db_url=db_url, fallback_path=Settings().audit_fallback_path)
    try:
        await store.init()
        await audit.init()
        user = await store.get_by_username(args.username)
        if user is None:
            print(f"ERROR: user {args.username!r} not found", file=sys.stderr)
            return 1
        await store.disable_user(user.user_id, audit_log=audit, acting_user_id="_system")
    finally:
        await audit.aclose()
        await store.aclose()
    print(f"User {args.username!r} disabled. Files owned by this user are retained "
          "(spec invariant I-3). Use 'reassign' (T5) to transfer ownership.")
    return 0


async def cmd_set_role(args: argparse.Namespace) -> int:
    db_url = _db_url()
    store = UsersStore(db_url=db_url)
    audit = AuditLog(db_url=db_url, fallback_path=Settings().audit_fallback_path)
    try:
        await store.init()
        await audit.init()
        user = await store.get_by_username(args.username)
        if user is None:
            print(f"ERROR: user {args.username!r} not found", file=sys.stderr)
            return 1
        old_role = user.role
        await store.set_role(user.user_id, args.role, audit_log=audit, acting_user_id="_system")
    finally:
        await audit.aclose()
        await store.aclose()
    print(f"role changed: {old_role} -> {args.role}")
    return 0


async def cmd_reset_key(args: argparse.Namespace) -> int:
    db_url = _db_url()
    store = UsersStore(db_url=db_url)
    audit = AuditLog(db_url=db_url, fallback_path=Settings().audit_fallback_path)
    try:
        await store.init()
        await audit.init()
        user = await store.get_by_username(args.username)
        if user is None:
            print(f"ERROR: user {args.username!r} not found", file=sys.stderr)
            return 1
        plaintext = await store.reset_key(user.user_id, audit_log=audit, acting_user_id="_system")
    finally:
        await audit.aclose()
        await store.aclose()
    print(f"Rotated key for {args.username!r}. Old key is now invalid.")
    print()
    print("New API key (shown once):")
    print(f"  {plaintext}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="KB user admin CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_create = sub.add_parser("create", help="Create a new user + key")
    p_create.add_argument("username")
    p_create.add_argument("--role", choices=["member", "admin"], required=True)
    p_create.set_defaults(handler=cmd_create)

    p_list = sub.add_parser("list", help="List users")
    p_list.add_argument("--all", action="store_true", help="Include disabled users")
    p_list.set_defaults(handler=cmd_list)

    p_disable = sub.add_parser("disable", help="Soft-disable a user")
    p_disable.add_argument("username")
    p_disable.add_argument(
        "--reassign-to", default=None,
        help="Username to inherit docs before disabling (T5).",
    )
    p_disable.set_defaults(handler=cmd_disable)

    p_set_role = sub.add_parser("set-role", help="Change a user's role")
    p_set_role.add_argument("username")
    p_set_role.add_argument("--role", choices=["member", "admin"], required=True)
    p_set_role.set_defaults(handler=cmd_set_role)

    p_reset = sub.add_parser("reset-key", help="Rotate a user's API key")
    p_reset.add_argument("username")
    p_reset.set_defaults(handler=cmd_reset_key)

    p_reassign = sub.add_parser("reassign", help="Reassign all docs from one user to another.")
    p_reassign.add_argument("--from", dest="from_username", required=True)
    p_reassign.add_argument("--to",   dest="to_username",   required=True)
    p_reassign.set_defaults(handler=cmd_reassign)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return asyncio.run(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
