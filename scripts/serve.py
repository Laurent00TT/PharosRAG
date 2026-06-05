# scripts/serve.py
"""
Start the Knowledge Base Tool Server.

Usage:
  python scripts/serve.py
  python scripts/serve.py --host 0.0.0.0 --port 8000

Architecture: embedding (TextChannel / VisionChannel) and rerank are HTTP clients
to remote model servers (see .env MULTIMODAL_EMBEDDING_SERVER_URL /
RERANKER_SERVER_URL), not in-process models. The tool_server process loads no GPU
model itself, so its own cold start is fast; first-search latency instead depends
on those upstream services (and the SSH tunnel) being reachable.
See the project README for the full stack.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="Knowledge Base Tool Server")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    args = parser.parse_args()

    uvicorn.run(
        "kb.tool_server.app:app",
        host=args.host,
        port=args.port,
        log_level="info",
    )
