from kb.config import Settings
from kb.search.planner import QueryPlanner


def test_planner_detects_visual_query(tmp_path):
    settings = Settings(
        qdrant_path=str(tmp_path / "q"),
        image_storage_path=str(tmp_path / "i"),
    )

    plan = QueryPlanner(settings=settings).plan("这个页面图表长什么样", {})

    assert plan.needs_vision is True
    assert "vision" in plan.enabled_channels


def test_planner_detects_process_query(tmp_path):
    settings = Settings(
        qdrant_path=str(tmp_path / "q"),
        image_storage_path=str(tmp_path / "i"),
    )

    plan = QueryPlanner(settings=settings).plan("审批流程有哪些步骤", {})

    assert plan.query_type == "process"
    assert "dense" in plan.enabled_channels
    assert "desc" in plan.enabled_channels


def test_planner_uses_light_plan_for_exact_doc_filter(tmp_path):
    settings = Settings(
        qdrant_path=str(tmp_path / "q"),
        image_storage_path=str(tmp_path / "i"),
    )

    plan = QueryPlanner(settings=settings).plan(
        "CEO approval",
        {"doc_name": "manual.pdf"},
    )

    assert plan.multi_query_n <= 2


def test_planner_respects_fast_profile(tmp_path):
    settings = Settings(
        qdrant_path=str(tmp_path / "q"),
        image_storage_path=str(tmp_path / "i"),
        kb_profile="fast",
    )

    plan = QueryPlanner(settings=settings).plan("general question", {})

    assert plan.use_hyde is False
    assert plan.multi_query_n == 1
