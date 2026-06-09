import logging
from dataclasses import dataclass

from kb.config import Settings
from kb.search.constants import (
    CHANNEL_DESC,
    CHANNEL_TEXT_DENSE,
    CHANNEL_TEXT_SPARSE,
    CHANNEL_VISION,
)

logger = logging.getLogger(__name__)


# Per-query-class minimum multi_query_n. Empirically, single-query recall
# on these classes is weak — the planner upgrades them to at least N so a
# fast/cheap user config doesn't silently degrade these specific paths.
# Listed here so the override is auditable, not buried in a one-liner.
_MIN_MULTI_QUERY_N_BY_CLASS = {
    "table_lookup": 2,
    "process": 2,
}


def _resolve_multi_query_n(query_class: str, configured: int) -> int:
    """Return the effective multi_query_n. Logs an INFO when the
    planner's class-specific floor exceeds the user's configured value
    — keeps the override transparent rather than a silent surprise."""
    floor = _MIN_MULTI_QUERY_N_BY_CLASS.get(query_class, 0)
    if floor > configured:
        logger.info(
            "QueryPlanner upgraded multi_query_n from %d to %d for %s "
            "(empirical recall floor)", configured, floor, query_class,
        )
        return floor
    return configured


@dataclass
class SearchPlan:
    query_type: str
    use_hyde: bool
    multi_query_n: int
    enabled_channels: list[str]
    candidate_multiplier: int
    needs_vision: bool
    reason: str


class QueryPlanner:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def plan(self, query: str, doc_filter: dict | None) -> SearchPlan:
        q = query.lower()
        doc_filter = doc_filter or {}
        search_config = self._settings.effective_search_config()
        all_channels = [
            CHANNEL_TEXT_DENSE,
            CHANNEL_TEXT_SPARSE,
            CHANNEL_DESC,
            CHANNEL_VISION,
        ]

        if doc_filter.get("doc_name"):
            return SearchPlan(
                query_type="fact_lookup",
                use_hyde=False,
                multi_query_n=1,
                enabled_channels=[CHANNEL_TEXT_DENSE, CHANNEL_TEXT_SPARSE, CHANNEL_DESC],
                candidate_multiplier=2,
                needs_vision=False,
                reason="exact doc filter",
            )
        if any(w in q for w in ("图", "截图", "长什么样", "页面", "位置", "chart", "image")):
            return SearchPlan(
                query_type="visual_layout",
                use_hyde=False,
                multi_query_n=2,
                enabled_channels=all_channels,
                candidate_multiplier=3,
                needs_vision=True,
                reason="visual terms",
            )
        if any(w in q for w in ("表", "金额", "字段", "对照", "多少", "table")):
            return SearchPlan(
                query_type="table_lookup",
                use_hyde=False,
                multi_query_n=_resolve_multi_query_n("table_lookup", search_config["multi_query_n"]),
                enabled_channels=[CHANNEL_TEXT_DENSE, CHANNEL_TEXT_SPARSE, CHANNEL_DESC],
                candidate_multiplier=3,
                needs_vision=False,
                reason="table terms",
            )
        if any(w in q for w in ("流程", "步骤", "审批", "如何", "怎么办", "process", "steps")):
            return SearchPlan(
                query_type="process",
                use_hyde=search_config["hyde_enabled"],
                multi_query_n=_resolve_multi_query_n("process", search_config["multi_query_n"]),
                enabled_channels=[CHANNEL_TEXT_DENSE, CHANNEL_TEXT_SPARSE, CHANNEL_DESC],
                candidate_multiplier=3,
                needs_vision=False,
                reason="process terms",
            )
        return SearchPlan(
            query_type="general",
            use_hyde=search_config["hyde_enabled"],
            multi_query_n=search_config["multi_query_n"],
            enabled_channels=all_channels,
            candidate_multiplier=3,
            needs_vision=True,
            reason="default",
        )
