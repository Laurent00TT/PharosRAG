# src/kb/eval/ragas_eval.py
"""Optional RAGAS integration for end-to-end RAG evaluation.

Requires: pip install ragas datasets
The function returns None if RAGAS is not installed, allowing eval scripts to
gracefully skip end-to-end metrics when only retrieval-side eval is wanted.
"""
import asyncio
import logging

logger = logging.getLogger(__name__)


async def run_ragas_eval(
    queries: list[str],
    contexts: list[list[str]],
    answers: list[str],
    ground_truths: list[str] | None = None,
) -> dict | None:
    """Run RAGAS metrics: faithfulness, answer_relevancy, context_precision.
    Returns the metric dict or None if RAGAS isn't installed.
    """
    try:
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, context_precision, faithfulness
        from datasets import Dataset
    except ImportError:
        logger.warning(
            "RAGAS not installed; skipping end-to-end eval. "
            "Install with: pip install ragas datasets"
        )
        return None

    data: dict = {"question": queries, "answer": answers, "contexts": contexts}
    if ground_truths:
        data["ground_truth"] = ground_truths

    dataset = Dataset.from_dict(data)
    metrics = [faithfulness, answer_relevancy, context_precision]
    # ragas.evaluate is sync; run in thread to avoid blocking the loop
    result = await asyncio.to_thread(evaluate, dataset, metrics=metrics)
    return dict(result)
