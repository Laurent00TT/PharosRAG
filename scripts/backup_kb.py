"""Take a consistent backup of the KB (T5-8 v2 review fixes inside).

Flow:
  1. POST /admin/maintenance_mode {on:true}     — wait drain (5min cap)
  2. SQLite .backup() API for the 3 KB DBs      — captures uncheckpointed WAL
  3. shutil.copytree for qdrant (ignore *.db/.lock/...) and images
  4. Open the COPIED kb_metadata.db and reset maintenance_state.on=0
  5. Write manifest.json (ts, schema_version, sha256 checksums)
  6. POST /admin/maintenance_mode {on:false}    — always (finally)

strict=True (default): drained:false from pause → abort + resume + raise.
strict=False (--best-effort): drained:false is tolerated; snapshot may
be inconsistent but operator opted in.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import shutil
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import httpx


_QDRANT_TREE_IGNORE_PATTERNS = (
    "*.db", "*.db-wal", "*.db-shm",
    "*.lock",
    "*.jsonl",
)


async def _set_maintenance_mode(server_url: str, admin_key: str, *, on: bool) -> dict:
    async with httpx.AsyncClient(timeout=320.0) as client:
        resp = await client.post(
            f"{server_url}/admin/maintenance_mode",
            json={"on": on},
            headers={"X-API-Key": admin_key},
        )
        resp.raise_for_status()
        return resp.json()


def _sqlite_online_backup(src: Path, dst: Path) -> None:
    """SQLite Python connection.backup() API for atomic, WAL-aware online backup."""
    src_conn = sqlite3.connect(str(src))
    dst_conn = sqlite3.connect(str(dst))
    try:
        with dst_conn:
            src_conn.backup(dst_conn)
    finally:
        src_conn.close()
        dst_conn.close()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _reset_maintenance_in_copy(backup_meta_db: Path) -> None:
    """Open the copied kb_metadata.db and clear maintenance_state.on.
    The live DB was put in maintenance BEFORE the copy; the snapshot
    inherits that state. Restoring the snapshot must come up clean.

    Use quoted "on" identifier — `on` is a SQLite reserved word."""
    conn = sqlite3.connect(str(backup_meta_db))
    try:
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='maintenance_state'"
        )
        if cur.fetchone() is None:
            return
        conn.execute(
            'UPDATE maintenance_state SET "on"=0, set_at=NULL, set_by_user_id=NULL WHERE id=1'
        )
        conn.commit()
    finally:
        conn.close()


async def backup(
    *,
    qdrant_path: str,
    image_storage_path: str,
    backup_dir: str,
    server_url: str,
    admin_key: str,
    strict: bool = True,
    nav_db_path: str | None = None,
) -> dict:
    src = Path(qdrant_path)
    img = Path(image_storage_path)
    dest = Path(backup_dir)
    dest.mkdir(parents=True, exist_ok=True)

    # Default nav DB lives inside qdrant_path; respect a customized path.
    nav_db = Path(nav_db_path) if nav_db_path else (src / "nav_index.db")

    pause_resp = await _set_maintenance_mode(server_url, admin_key, on=True)
    if strict and not pause_resp.get("drained"):
        await _set_maintenance_mode(server_url, admin_key, on=False)
        raise RuntimeError(
            f"server did not drain (active={pause_resp.get('still_active')}); "
            f"pass --best-effort to proceed anyway"
        )

    try:
        # (P0 #3a) SQLite online backup for the 3 KB DBs.
        # Two DBs live inside qdrant_path; nav_index.db may be configured
        # outside qdrant_path via NAV_DB_PATH — use the resolved nav_db path.
        checksums: dict[str, str] = {}
        kb_dbs = [
            ("kb_metadata.db", src / "kb_metadata.db"),
            ("ingestion_jobs.db", src / "ingestion_jobs.db"),
            ("nav_index.db", nav_db),
        ]
        for name, srcp in kb_dbs:
            if not srcp.exists():
                continue
            dstp = dest / f"{name}.bak"
            _sqlite_online_backup(srcp, dstp)
            checksums[name] = _sha256_file(dstp)

        # (P1.2) reset the maintenance flag in the copied metadata DB
        meta_backup = dest / "kb_metadata.db.bak"
        if meta_backup.exists():
            _reset_maintenance_in_copy(meta_backup)
            checksums["kb_metadata.db"] = _sha256_file(meta_backup)

        # (P0 #3b) Qdrant tree with ignore patterns
        # src must exist; raise immediately rather than silently skipping
        shutil.copytree(
            src, dest / "qdrant",
            ignore=shutil.ignore_patterns(*_QDRANT_TREE_IGNORE_PATTERNS),
            dirs_exist_ok=True,
        )

        # images — full copytree (no exclusions); skip only if path absent
        if img.exists():
            shutil.copytree(img, dest / "images", dirs_exist_ok=True)

        manifest = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "schema_version": 5,
            "checksums": checksums,
            "qdrant_path": str(src),
            "image_storage_path": str(img),
            "drained": pause_resp.get("drained"),
            "strict": strict,
        }
        (dest / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8",
        )
        return manifest
    finally:
        try:
            await _set_maintenance_mode(server_url, admin_key, on=False)
        except Exception as exc:  # noqa: BLE001
            print(f"WARN: failed to resume server: {exc}", file=sys.stderr)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Take a consistent KB backup.")
    parser.add_argument("backup_dir")
    parser.add_argument("--server-url", default="http://127.0.0.1:8765")
    parser.add_argument("--admin-key", default=None)
    parser.add_argument(
        "--best-effort", action="store_true",
        help="Allow backup even if workers did not drain (default: refuse).",
    )
    return parser


async def amain(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    import os
    from kb.config import Settings
    settings = Settings()
    admin_key = args.admin_key or os.environ.get("KB_ADMIN_KEY")
    if not admin_key:
        print("ERROR: provide --admin-key or set KB_ADMIN_KEY", file=sys.stderr)
        return 2
    try:
        manifest = await backup(
            qdrant_path=settings.qdrant_path,
            image_storage_path=settings.image_storage_path,
            backup_dir=args.backup_dir,
            server_url=args.server_url,
            admin_key=admin_key,
            strict=not args.best_effort,
            nav_db_path=settings.nav_db_path,    # NEW: honor custom NAV_DB_PATH
        )
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 3
    print(f"Backup written to {args.backup_dir}")
    print(f"  ts: {manifest['ts']}")
    print(f"  drained: {manifest['drained']}")
    print(f"  strict: {manifest['strict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(amain()))
