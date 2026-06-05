# tests/test_eval.py
from kb.eval.metrics import recall_at_k, ndcg_at_k, mean_reciprocal_rank


def test_recall_at_k_full_match():
    retrieved = [("d", 0), ("d", 1), ("d", 2)]
    relevant = [("d", 0), ("d", 1)]
    assert recall_at_k(retrieved, relevant, k=5) == 1.0


def test_recall_at_k_partial():
    retrieved = [("d", 0), ("d", 9)]
    relevant = [("d", 0), ("d", 1)]
    assert recall_at_k(retrieved, relevant, k=5) == 0.5


def test_recall_at_k_truncates():
    retrieved = [("d", 9), ("d", 0), ("d", 1)]
    relevant = [("d", 0), ("d", 1)]
    # k=1: only first hit considered → no relevant in top-1 → 0.0
    assert recall_at_k(retrieved, relevant, k=1) == 0.0


def test_recall_at_k_empty_relevant():
    assert recall_at_k([("d", 0)], [], k=5) == 0.0


def test_ndcg_at_k_perfect():
    retrieved = [("d", 0), ("d", 1)]
    relevant = [("d", 0), ("d", 1)]
    assert abs(ndcg_at_k(retrieved, relevant, k=2) - 1.0) < 1e-6


def test_ndcg_at_k_decreasing():
    # Relevant doc at position 0 vs position 2
    retrieved_good = [("d", 0), ("d", 9), ("d", 8)]
    retrieved_bad = [("d", 9), ("d", 8), ("d", 0)]
    relevant = [("d", 0)]
    assert ndcg_at_k(retrieved_good, relevant, k=3) > ndcg_at_k(retrieved_bad, relevant, k=3)


def test_ndcg_at_k_no_match():
    assert ndcg_at_k([("d", 9)], [("d", 0)], k=5) == 0.0


def test_mrr_first_position():
    assert mean_reciprocal_rank([("d", 0), ("d", 1)], [("d", 0)]) == 1.0


def test_mrr_third_position():
    retrieved = [("d", 9), ("d", 8), ("d", 0)]
    assert abs(mean_reciprocal_rank(retrieved, [("d", 0)]) - (1.0 / 3)) < 1e-6


def test_mrr_no_match():
    assert mean_reciprocal_rank([("d", 9)], [("d", 0)]) == 0.0


def test_classify_eval_failure_no_recall():
    from kb.eval.runner import classify_failure

    assert classify_failure([], [("d", 1)]) == "no_recall"


def test_classify_eval_failure_wrong_page():
    from kb.eval.runner import classify_failure

    assert classify_failure([("d", 2)], [("d", 1)]) == "wrong_page"


from pathlib import Path
import json
from kb.eval.testset import TestQuery, load_testset


def test_load_testset_parses_jsonl(tmp_path):
    p = tmp_path / "ts.jsonl"
    p.write_text(
        '{"query":"q1","relevant":[["d1",0],["d1",1]]}\n'
        '{"query":"q2","relevant":[["d2",5]]}\n',
        encoding="utf-8",
    )
    ts = load_testset(p)
    assert len(ts) == 2
    assert ts[0].query == "q1"
    assert ts[0].relevant == [("d1", 0), ("d1", 1)]


def test_load_testset_handles_metadata(tmp_path):
    p = tmp_path / "ts.jsonl"
    p.write_text(
        '{"query":"q","relevant":[["d",0]],"metadata":{"category":"flowchart"}}\n',
        encoding="utf-8",
    )
    ts = load_testset(p)
    assert ts[0].metadata == {"category": "flowchart"}


def test_load_testset_skips_blank_lines(tmp_path):
    p = tmp_path / "ts.jsonl"
    p.write_text(
        '{"query":"q1","relevant":[["d1",0]]}\n\n   \n{"query":"q2","relevant":[["d2",1]]}\n',
        encoding="utf-8",
    )
    ts = load_testset(p)
    assert len(ts) == 2


import pytest
from unittest.mock import AsyncMock, MagicMock
from kb.eval.runner import EvalRunner, EvalResult
from kb.eval.testset import TestQuery
from kb.models import SearchResponse, SearchResult


def _r(doc_id: str, page_num: int) -> SearchResult:
    return SearchResult(
        doc_id=doc_id, doc_name="x.pdf", page_num=page_num,
        heading_path=[], text_chunk="", vision_description=None,
        figure_index=None, figure_caption=None, image_url="",
        page_type="text", doc_version=None, effective_date=None,
        score=0.5, match_channels=["dense"],
    )


@pytest.mark.asyncio
async def test_eval_runner_computes_metrics():
    engine = MagicMock()
    # First query: perfect retrieval
    # Second query: relevant doc at rank 2
    engine.search = AsyncMock(side_effect=[
        (SearchResponse(results=[_r("d1", 0), _r("d1", 1)], total=2), False),
        (SearchResponse(results=[_r("dX", 9), _r("d2", 5), _r("d2", 6)], total=3), False),
    ])
    testset = [
        TestQuery(query="q1", relevant=[("d1", 0), ("d1", 1)]),
        TestQuery(query="q2", relevant=[("d2", 5)]),
    ]
    runner = EvalRunner(engine=engine)
    result = await runner.run(testset, top_k=5)

    assert isinstance(result, EvalResult)
    assert result.n_queries == 2
    assert result.k == 5
    # q1: full recall (1.0), q2: 1.0 too (doc at rank 1 within top 5)
    assert result.avg_recall_at_k == 1.0
    # MRR: q1=1.0 (first hit at rank 0), q2=1/2=0.5 → avg = 0.75
    assert abs(result.avg_mrr - 0.75) < 1e-6


@pytest.mark.asyncio
async def test_eval_runner_uses_top_k_for_metrics():
    """Regression: ensure `top_k=10` actually computes Recall@10, not Recall@5."""
    engine = MagicMock()
    # 6 retrieved hits; relevant at rank 5 (0-indexed). Recall@5=0, Recall@10=1.0.
    hits = [_r("dX", i) for i in range(5)] + [_r("d_target", 0)]
    engine.search = AsyncMock(return_value=(
        SearchResponse(results=hits, total=6), False,
    ))
    testset = [TestQuery(query="q", relevant=[("d_target", 0)])]

    runner = EvalRunner(engine=engine)
    r5 = await runner.run(testset, top_k=5)
    r10 = await runner.run(testset, top_k=10)

    assert r5.k == 5 and r10.k == 10
    assert r5.avg_recall_at_k == 0.0   # relevant at rank 5 not in top-5
    assert r10.avg_recall_at_k == 1.0  # but is in top-10


@pytest.mark.asyncio
async def test_eval_runner_per_query_details():
    engine = MagicMock()
    engine.search = AsyncMock(return_value=(
        SearchResponse(results=[_r("d1", 0)], total=1), False,
    ))
    testset = [TestQuery(query="x", relevant=[("d1", 0)])]
    runner = EvalRunner(engine=engine)
    result = await runner.run(testset, top_k=3)

    assert len(result.per_query) == 1
    pq = result.per_query[0]
    assert pq["query"] == "x"
    assert pq["retrieved"] == [("d1", 0)]
    assert pq["recall"] == 1.0   # generic key, not "recall@5"


@pytest.mark.asyncio
async def test_eval_runner_groups_metrics_by_query_type():
    engine = MagicMock()
    engine.search = AsyncMock(return_value=(SearchResponse(results=[], total=0), False))
    testset = [
        TestQuery(query="q1", relevant=[("d", 1)], metadata={"query_type": "table_lookup"}),
        TestQuery(query="q2", relevant=[("d", 2)], metadata={"query_type": "visual_layout"}),
    ]

    result = await EvalRunner(engine).run(testset, top_k=5)

    assert "table_lookup" in result.group_metrics
    assert "visual_layout" in result.group_metrics
    assert result.group_metrics["table_lookup"]["n_queries"] == 1


@pytest.mark.asyncio
async def test_eval_runner_empty_testset_has_empty_group_metrics():
    engine = MagicMock()
    engine.search = AsyncMock()

    result = await EvalRunner(engine).run([], top_k=5)

    assert result.group_metrics == {}


@pytest.mark.asyncio
async def test_eval_runner_adds_failure_bucket_for_misses():
    engine = MagicMock()
    engine.search = AsyncMock(return_value=(
        SearchResponse(results=[_r("d", 2)], total=1),
        False,
    ))
    testset = [TestQuery(query="q", relevant=[("d", 1)])]

    result = await EvalRunner(engine).run(testset, top_k=5)

    assert result.per_query[0]["failure_bucket"] == "wrong_page"


from unittest.mock import patch as _patch
from kb.eval.ragas_eval import run_ragas_eval


@pytest.mark.asyncio
async def test_ragas_returns_none_if_not_installed():
    # Simulate ImportError on the ragas import
    import builtins
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "ragas" or name.startswith("ragas."):
            raise ImportError("ragas not installed")
        return original_import(name, *args, **kwargs)

    with _patch.object(builtins, "__import__", side_effect=fake_import):
        result = await run_ragas_eval(
            queries=["q"], contexts=[["c"]], answers=["a"]
        )
    assert result is None


@pytest.mark.asyncio
async def test_ragas_returns_dict_when_installed():
    fake_eval_result = MagicMock()
    fake_eval_result.__iter__ = MagicMock(return_value=iter([]))
    # Mock the ragas internals so we don't need the real package
    with _patch.dict("sys.modules", {
        "ragas": MagicMock(evaluate=MagicMock(return_value={"faithfulness": 0.8})),
        "ragas.metrics": MagicMock(faithfulness="x", answer_relevancy="y",
                                   context_precision="z"),
        "datasets": MagicMock(Dataset=MagicMock(from_dict=MagicMock(return_value="ds"))),
    }):
        result = await run_ragas_eval(
            queries=["q"], contexts=[["c"]], answers=["a"]
        )
    assert isinstance(result, dict)
    assert result.get("faithfulness") == 0.8
