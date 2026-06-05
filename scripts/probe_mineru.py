"""MinerU API entry-point probe — does not modify code, just prints which
import paths are actually available.

Usage:
  python scripts/probe_mineru.py
  python scripts/probe_mineru.py --pdf path/to/sample.pdf

Output:
  - For each candidate import path, the result (OK / ImportError / etc.)
  - The exposed public names of the entry function or class
  - (optional) Run a real PDF through hybrid / vlm / pipeline and inspect the
    middle_json structure

Use the result to decide how scripts/mineru_server.py:_run_mineru should be wired.
"""
from __future__ import annotations

import argparse
import importlib
import inspect
import json
import sys
import traceback
from pathlib import Path


# Candidate import paths — covers MinerU 2.x / 3.0 / 3.1 known naming conventions.
_CANDIDATES = [
    # MinerU 3.x recommended client (mineru_vl_utils package)
    ("mineru_vl_utils", "MinerUClient"),
    ("mineru_vl_utils", "MinerU"),
    # MinerU 3.1+ backend module paths (what mineru_server.py used to assume)
    ("mineru.backend.hybrid.hybrid_analyze", "hybrid_analyze"),
    ("mineru.backend.vlm.vlm_analyze", "vlm_analyze"),
    ("mineru.backend.pipeline.pipeline_analyze", "doc_analyze_streaming"),
    ("mineru.backend.pipeline.pipeline_analyze", "pipeline_analyze"),
    # MinerU 2.x / magic_pdf legacy names
    ("magic_pdf.pipe.UNIPipe", "UNIPipe"),
    ("magic_pdf.pipe.OCRPipe", "OCRPipe"),
    ("magic_pdf.pipe.TXTPipe", "TXTPipe"),
    # Top-level shortcut entry-points
    ("mineru", "parse"),
    ("mineru", "MinerU"),
    ("mineru", "Client"),
]


def probe_one(module_path: str, name: str) -> dict:
    """Try import + getattr; return a structured result."""
    try:
        mod = importlib.import_module(module_path)
    except ImportError as e:
        return {
            "module": module_path,
            "name": name,
            "status": "module_missing",
            "error": str(e),
        }
    except Exception as e:
        return {
            "module": module_path,
            "name": name,
            "status": "module_error",
            "error": f"{type(e).__name__}: {e}",
        }
    obj = getattr(mod, name, None)
    if obj is None:
        return {
            "module": module_path,
            "name": name,
            "status": "name_missing",
            "available_names": [n for n in dir(mod) if not n.startswith("_")][:20],
        }
    return {
        "module": module_path,
        "name": name,
        "status": "OK",
        "kind": type(obj).__name__,
        "callable": callable(obj),
        "signature": _safe_signature(obj),
    }


def _safe_signature(obj) -> str:
    try:
        return str(inspect.signature(obj))
    except (TypeError, ValueError):
        return "(signature unavailable)"


def list_top_level() -> dict:
    """If the mineru top-level package can be imported, list its public names."""
    try:
        mod = importlib.import_module("mineru")
    except Exception as e:
        return {"status": "mineru top-level import failed", "error": str(e)}
    return {
        "status": "OK",
        "module_file": getattr(mod, "__file__", "?"),
        "version": getattr(mod, "__version__", "?"),
        "public_names": sorted(n for n in dir(mod) if not n.startswith("_")),
    }


def try_parse_pdf(pdf_path: Path) -> dict:
    """If a real PDF is provided, run hybrid_analyze or MinerUClient against it
    and inspect the output shape."""
    if not pdf_path.exists():
        return {"status": "skipped", "reason": "PDF not found"}

    pdf_bytes = pdf_path.read_bytes()
    attempts = []

    # Attempt 1: hybrid_analyze
    try:
        from mineru.backend.hybrid.hybrid_analyze import hybrid_analyze
        result = hybrid_analyze(pdf_bytes, lang="ch")
        attempts.append({
            "approach": "hybrid_analyze(pdf_bytes, lang='ch')",
            "status": "OK",
            "result_type": type(result).__name__,
            "result_keys": list(result.keys())[:20] if isinstance(result, dict) else None,
        })
        return {"first_success": "hybrid_analyze", "attempts": attempts}
    except ImportError as e:
        attempts.append({"approach": "hybrid_analyze", "status": "ImportError", "error": str(e)})
    except Exception as e:
        attempts.append({
            "approach": "hybrid_analyze",
            "status": "runtime_error",
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc().splitlines()[-5:],
        })

    # Attempt 2: MinerUClient
    try:
        from mineru_vl_utils import MinerUClient  # type: ignore
        client = MinerUClient()
        # We don't call any methods — just verify the constructor succeeds
        attempts.append({
            "approach": "MinerUClient()",
            "status": "construct_OK",
            "methods": [m for m in dir(client) if not m.startswith("_")][:20],
        })
        return {"first_success": "MinerUClient", "attempts": attempts}
    except ImportError as e:
        attempts.append({"approach": "MinerUClient", "status": "ImportError", "error": str(e)})
    except Exception as e:
        attempts.append({
            "approach": "MinerUClient",
            "status": "runtime_error",
            "error": f"{type(e).__name__}: {e}",
        })

    return {"first_success": None, "attempts": attempts}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=Path, default=None,
                        help="optional: real PDF to actually parse")
    parser.add_argument("--json", action="store_true",
                        help="emit a structured JSON report (machine-readable)")
    args = parser.parse_args()

    top = list_top_level()
    probes = [probe_one(m, n) for m, n in _CANDIDATES]
    parse_result = None
    if args.pdf:
        parse_result = try_parse_pdf(args.pdf)

    report = {
        "top_level": top,
        "candidates": probes,
        "parse_attempt": parse_result,
        "ok_candidates": [p for p in probes if p["status"] == "OK"],
    }

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
        return

    # Human-readable output — uses ASCII flags to avoid console UTF-8 issues
    print("=== MinerU top-level ===")
    print(json.dumps(top, ensure_ascii=False, indent=2, default=str))

    print("\n=== Candidate import paths ===")
    for p in probes:
        flag = "[OK]" if p["status"] == "OK" else "[--]"
        print(f"  {flag} {p['module']}.{p['name']}  ->  {p['status']}")
        if p["status"] == "OK":
            print(f"       kind={p['kind']}  signature={p['signature']}")
        elif p["status"] == "name_missing":
            print(f"       module loads but no '{p['name']}' attr")
            print(f"       available (head): {p.get('available_names', [])}")

    print("\n=== Recommended next step ===")
    ok = report["ok_candidates"]
    if not ok:
        print("[FAIL] No candidate is available — MinerU is likely not installed or the package layout has changed.")
        print("  Confirm install: uv pip install 'magic-pdf[full]' or pip install mineru")
        print("  Then inspect site-packages/ for the actual layout.")
    else:
        print(f"[PASS] {len(ok)} entry-point(s) available:")
        for p in ok:
            print(f"  - {p['module']}.{p['name']}  signature={p['signature']}")
        print("\n  -> wire scripts/mineru_server.py:_load_mineru / _run_mineru to use the paths above.")

    if parse_result:
        print("\n=== Real PDF parse attempt ===")
        print(json.dumps(parse_result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
