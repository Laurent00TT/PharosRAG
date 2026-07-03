"""Pharos CLI:serve(HTTP 守护进程)/ mcp(薄适配器)/ index(建库)/ ask(一次性问答)/ health。

serve/index 需要 WSL navikb 环境(GPU + 引擎依赖);mcp/ask/health 只需 httpx(+ mcp 包)。
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

from . import __version__, config


def _client(url: str | None):
    import httpx
    config.load_env()                      # 评审修:带 --url 时也要加载 .env(PHAROS_API_KEY 在里面)
    base = url or config.adapter_base_url()
    headers = {}
    if os.environ.get("PHAROS_API_KEY"):
        headers["X-API-Key"] = os.environ["PHAROS_API_KEY"]
    return httpx.Client(base_url=base, timeout=httpx.Timeout(600.0, connect=5.0), headers=headers)


def cmd_serve(args) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    import uvicorn
    from .service import create_app
    cfg = config.from_env()
    host, port = args.host or cfg.host, args.port or cfg.port
    print(f"Pharos v{__version__}  http://{host}:{port}  index={cfg.qdrant_path} "
          f"collection={cfg.collection} tenant={cfg.tenant or '(未设,fail-closed)'}", flush=True)
    uvicorn.run(create_app(cfg), host=host, port=port, log_level="info")


def cmd_mcp(_args) -> None:
    from .mcp_adapter import main as adapter_main
    adapter_main()


def cmd_index(args) -> None:
    from .indexer import run_index
    cfg = config.from_env()
    run_index(cfg, corpus=args.corpus, dest=args.dest, collection=args.collection,
              tenant=args.tenant, visibility=args.visibility, allow=args.allow,
              only=args.only, limit=args.limit)


def cmd_ask(args) -> None:
    with _client(args.url) as c:
        try:
            r = c.post("/v1/ask", json={"query": args.query, "top_k": args.top_k,
                                        "rerank": args.rerank, "include_contexts": args.contexts,
                                        "doc_ids": args.doc_id or None, "doc_type": args.doc_type,
                                        "kind": args.kind, "strategy": args.strategy})
        except Exception as e:
            raise SystemExit(f"连不上 Pharos 守护进程({c.base_url}):{type(e).__name__}。先跑 pharos serve。")
        try:
            data = r.json()
        except ValueError:                 # 评审修:5xx/非 JSON 不裸抛 traceback
            raise SystemExit(f"守护进程返回异常(HTTP {r.status_code},非 JSON),详见服务端日志。")
    if args.json:
        print(json.dumps(data, ensure_ascii=False, indent=2))
        return
    if data.get("status") != "ok":
        raise SystemExit(f"[{data.get('status')}] {data.get('hint', '')}")
    print(data["answer"])
    if data.get("citations"):
        print("\n来源:")
        for c_ in data["citations"]:
            sec = f" §{c_['section']}" if c_.get("section") else ""
            print(f"  [{c_['marker']}] {c_['title']} (p.{c_['page']}{sec})  {c_['chunk_id']}")
    if data.get("hints"):
        print("\n💡 提示:")
        for h in data["hints"]:
            print(f"  · {h}")
    if data.get("finish_reason") == "length":
        print("\n⚠ 答案被 max_tokens 截断(finish_reason=length)。", file=sys.stderr)


def cmd_health(args) -> None:
    with _client(args.url) as c:
        try:
            r = c.get("/healthz")
        except Exception as e:
            raise SystemExit(f"连不上 Pharos 守护进程({c.base_url}):{type(e).__name__}。先跑 pharos serve。")
    print(json.dumps(r.json(), ensure_ascii=False, indent=2))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="pharos", description="Pharos:个人多格式 agentic RAG 系统")
    p.add_argument("--version", action="version", version=f"pharos {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("serve", help="启动 HTTP 守护进程(独占索引 + GPU 模型)")
    sp.add_argument("--host", default=None)
    sp.add_argument("--port", type=int, default=None)
    sp.set_defaults(fn=cmd_serve)

    sp = sub.add_parser("mcp", help="启动 MCP 薄适配器(stdio→HTTP,接 Claude Code)")
    sp.set_defaults(fn=cmd_mcp)

    sp = sub.add_parser("index", help="从 MinerU 解析目录建索引(需 GPU;先停 serve)")
    sp.add_argument("--corpus", default=None, help="语料目录(默认引擎仓 parsed/)")
    sp.add_argument("--dest", default=None, help="索引输出目录(默认 PHAROS_INDEX_DIR=~/rag_real)")
    sp.add_argument("--collection", default=None)
    sp.add_argument("--tenant", default=None, help="ACL tenant(默认 PHAROS_TENANT 或 demo)")
    sp.add_argument("--visibility", default="public", choices=["public", "restricted"])
    sp.add_argument("--allow", default="", help="逗号分隔 principals(restricted 时用)")
    sp.add_argument("--only", default=None, help="只建目录名以此前缀开头的文档")
    sp.add_argument("--limit", type=int, default=None, help="只建前 N 篇(冒烟)")
    sp.set_defaults(fn=cmd_index)

    sp = sub.add_parser("ask", help="一次性问答(经守护进程 /v1/ask,闭管道+引用)")
    sp.add_argument("query")
    sp.add_argument("--top-k", type=int, default=None, dest="top_k")
    sp.add_argument("--rerank", action="store_true")
    sp.add_argument("--kind", default=None, choices=["text", "table", "image", "chart"],
                    help="只在此类块里检索(数字/表格题用 table)")
    sp.add_argument("--doc-type", default=None, dest="doc_type", help="限定文档类型(如 financial_report_en)")
    sp.add_argument("--doc-id", action="append", default=None, dest="doc_id", help="限定文档(可多次)")
    sp.add_argument("--strategy", default=None, choices=["hybrid", "dense", "sparse"],
                    help="检索选路(默认 hybrid)")
    sp.add_argument("--contexts", action="store_true", help="citations 带被引原文")
    sp.add_argument("--json", action="store_true", help="输出原始 JSON")
    sp.add_argument("--url", default=None, help="守护进程地址(默认 PHAROS_URL)")
    sp.set_defaults(fn=cmd_ask)

    sp = sub.add_parser("health", help="守护进程健康检查")
    sp.add_argument("--url", default=None)
    sp.set_defaults(fn=cmd_health)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
