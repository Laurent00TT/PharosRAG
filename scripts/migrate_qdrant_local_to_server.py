"""Migrate Qdrant local-file collections to a running Qdrant server.

Local-file mode (`QdrantClient(path=...)`) uses a simplified SQLite-backed
store. Once you switch to server mode (`QdrantClient(url=...)`) the existing
files are no longer recognized. This script moves every collection over.

Usage:
    python scripts/migrate_qdrant_local_to_server.py \
        --local-path ./qdrant_data \
        --server-url http://localhost:6333 \
        --batch-size 64
"""

import argparse
import time

from qdrant_client import QdrantClient
from qdrant_client.http import models as rest


def _pick_batch_size(name: str, default: int) -> int:
    """Multi-vector (vision) points are big — shrink the batch to avoid 400/disconnect."""
    return 2 if "vision" in name else default


def migrate(local_path: str, server_url: str, batch_size: int) -> None:
    local = QdrantClient(path=local_path)
    server = QdrantClient(url=server_url, timeout=120)

    cols = local.get_collections().collections
    print(f"Found {len(cols)} local collection(s): {[c.name for c in cols]}")

    for col in cols:
        name = col.name
        info = local.get_collection(name)
        bs = _pick_batch_size(name, batch_size)
        print(f"\n=== {name} ===")
        print(f"  points: {info.points_count}, batch_size: {bs}")

        if server.collection_exists(name):
            srv_count = server.count(name, exact=True).count
            if srv_count >= info.points_count:
                print(f"  skip — server already has {srv_count} points (>= local {info.points_count})")
                continue
            print(f"  resume — server has {srv_count}, local has {info.points_count}")
        else:
            print(f"  creating on server...")
            server.create_collection(
                collection_name=name,
                vectors_config=info.config.params.vectors,
                sparse_vectors_config=info.config.params.sparse_vectors,
            )

        offset = None
        moved = 0
        t0 = time.time()
        while True:
            points, offset = local.scroll(
                collection_name=name,
                limit=bs,
                offset=offset,
                with_payload=True,
                with_vectors=True,
            )
            if not points:
                break

            for attempt in range(3):
                try:
                    server.upsert(
                        collection_name=name,
                        points=[
                            rest.PointStruct(id=p.id, vector=p.vector, payload=p.payload)
                            for p in points
                        ],
                        wait=True,
                    )
                    break
                except Exception as exc:
                    if attempt == 2:
                        raise
                    print(f"  retry {attempt+1}/3 after {type(exc).__name__}: {str(exc)[:100]}")
                    time.sleep(1.5)

            moved += len(points)
            if moved % (bs * 10) == 0 or moved == info.points_count:
                pct = moved * 100 // max(info.points_count, 1)
                print(f"  moved {moved}/{info.points_count} ({pct}%)")
            if offset is None:
                break

        print(f"  done {moved} points in {time.time()-t0:.1f}s")

    local.close()
    print("\nMigration complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-path", default="./qdrant_data")
    parser.add_argument("--server-url", default="http://localhost:6333")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    migrate(args.local_path, args.server_url, args.batch_size)
