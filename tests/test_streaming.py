# tests/test_streaming.py
import asyncio
import pytest
from kb.ingestion.streaming_pipeline import streaming_ingest


@pytest.mark.asyncio
async def test_streaming_processes_multiple_docs(tmp_path):
    """3 个文档共享 batch，每个完成后 callback 被触发一次。"""
    files = []
    for i in range(3):
        p = tmp_path / f"doc{i}.txt"
        p.write_text(f"content of doc {i}", encoding="utf-8")
        files.append(p)

    completed = []

    async def fake_ingest_one(path):
        await asyncio.sleep(0.01)
        return {"doc_id": path.stem, "pages_processed": 1}

    async def on_done(path, result):
        completed.append((path.name, result))

    await streaming_ingest(
        files,
        ingest_fn=fake_ingest_one,
        on_doc_done=on_done,
        max_concurrent=2,
    )

    assert len(completed) == 3
    names = {n for n, _ in completed}
    assert names == {"doc0.txt", "doc1.txt", "doc2.txt"}


@pytest.mark.asyncio
async def test_streaming_respects_concurrency_limit(tmp_path):
    """max_concurrent=2 时同时运行的不超过 2 个。"""
    files = [tmp_path / f"d{i}.txt" for i in range(5)]
    for f in files:
        f.write_text("x", encoding="utf-8")

    in_flight_max = 0
    in_flight = 0
    lock = asyncio.Lock()

    async def fake_ingest_one(path):
        nonlocal in_flight, in_flight_max
        async with lock:
            in_flight += 1
            in_flight_max = max(in_flight_max, in_flight)
        await asyncio.sleep(0.05)
        async with lock:
            in_flight -= 1
        return {"doc_id": path.stem}

    async def on_done(path, result): pass

    await streaming_ingest(
        files,
        ingest_fn=fake_ingest_one,
        on_doc_done=on_done,
        max_concurrent=2,
    )

    assert in_flight_max <= 2


@pytest.mark.asyncio
async def test_streaming_continues_on_individual_failure(tmp_path):
    """单文档失败不中断其他文档。"""
    files = [tmp_path / f"d{i}.txt" for i in range(3)]
    for f in files:
        f.write_text("x", encoding="utf-8")

    async def fake_ingest_one(path):
        if path.name == "d1.txt":
            raise ValueError("simulated failure")
        return {"doc_id": path.stem}

    completed = []
    async def on_done(path, result):
        completed.append((path.name, result))

    await streaming_ingest(
        files,
        ingest_fn=fake_ingest_one,
        on_doc_done=on_done,
        max_concurrent=2,
    )

    # d0 和 d2 应成功，d1 应作为 error 报告
    assert len(completed) == 3
    name_to_result = dict(completed)
    assert name_to_result["d0.txt"]["doc_id"] == "d0"
    assert name_to_result["d2.txt"]["doc_id"] == "d2"
    assert "error" in name_to_result["d1.txt"]
