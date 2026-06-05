"""Smoke test: drive mineru_gpu_worker on one PDF without touching Qdrant.

What we verify end-to-end:
  - mineru_server cuda subprocess spawns & becomes healthy
  - /parse returns a usable middle_json
  - /exit + process death releases GPU

(PoC v1.0 also verified a co-resident ColQwen /unload here; that vision model
was retired in v1.1, so mineru_gpu_worker now only manages the MinerU process.)
"""
import asyncio
import sys
import threading
import time
import subprocess
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from kb.config import Settings
from kb.ingestion.mineru_worker import mineru_gpu_worker
from kb.parser.mineru_local import MinerULocalParser


def _gpu_mem_mib() -> int:
    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=2,
    )
    return int(r.stdout.strip())


async def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default=str(REPO_ROOT / "test_doc"))
    ap.add_argument("--pattern", default="*266.pdf")
    args = ap.parse_args()

    pdf = next(Path(args.input_dir).glob(args.pattern))
    print(f"PDF: {pdf.name}  {pdf.stat().st_size/1024:.0f} KB")

    # ingest_use_gpu_mineru=False so MinerULocalParser doesn't spawn its
    # own worker — this script drives mineru_gpu_worker manually below.
    settings = Settings().model_copy(update={"ingest_use_gpu_mineru": False})
    parser = MinerULocalParser(settings=settings)

    samples: list[tuple[str, int]] = []
    stop = False

    def monitor():
        while not stop:
            try:
                samples.append((time.strftime("%H:%M:%S"), _gpu_mem_mib()))
            except Exception:
                pass
            time.sleep(2)

    th = threading.Thread(target=monitor, daemon=True)
    th.start()

    samples.append(("baseline", _gpu_mem_mib()))
    t_total = time.monotonic()

    async with mineru_gpu_worker(settings):
        t_parse = time.monotonic()
        pages = await parser.parse_async(pdf)
        parse_s = time.monotonic() - t_parse
        print(f"parse: {parse_s:.1f}s, pages={len(pages)}")

    # Give OS a sec to reclaim
    await asyncio.sleep(3)
    samples.append(("post-exit", _gpu_mem_mib()))

    stop = True
    elapsed = time.monotonic() - t_total
    print(f"\nTotal cycle (unload + spawn + parse + exit): {elapsed:.1f}s")
    print(f"GPU mem: baseline {samples[0][1]} MiB, peak {max(s[1] for s in samples)} MiB, "
          f"post-exit {samples[-1][1]} MiB")
    print("\nSamples (every 2s):")
    for ts, mib in samples:
        print(f"  {ts:>10}: {mib:>5} MiB")


if __name__ == "__main__":
    asyncio.run(main())
