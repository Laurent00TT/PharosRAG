import time
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from kb.auth.context import get_current_user
from kb.tool_server.observability import log_search
from kb.tool_server.security import verify_api_key

router = APIRouter(dependencies=[Depends(verify_api_key)])


class SearchFilter(BaseModel):
    model_config = ConfigDict(extra="forbid")

    doc_name: str | None = Field(default=None, min_length=1, max_length=500)
    page_type: Literal["flowchart", "table", "screenshot", "mixed", "text"] | None = None

    def to_filter_dict(self) -> dict:
        data: dict = {}
        if self.doc_name is not None:
            data["doc_name"] = self.doc_name
        if self.page_type is not None:
            data["page_type"] = self.page_type
        return data


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    doc_filter: SearchFilter = Field(default_factory=SearchFilter)
    hyde: bool = False
    # ADMIN-ONLY debug flag (security review P0 #1). When True, retrieval
    # skips list_valid_doc_ids() and queries ALL Qdrant vectors — which
    # includes T5 soft-deleted vectors that sit in retention waiting for
    # GC. Member callers requesting include_deprecated=True now get 403
    # so a member can't search back content the system already marked
    # as deleted/deprecated. Admins still need it for forensic queries.
    include_deprecated: bool = False


@router.post("/search_docs")
async def search_docs(req: SearchRequest, request: Request):
    engine = request.app.state.engine
    # Security review P0 #1: include_deprecated bypasses the active gate
    # by setting doc_ids=None in the retriever — so any member could
    # otherwise pull soft-deleted vectors during the GC retention window.
    # Gate the flag to admins; deny BEFORE the engine call so we don't
    # leak that the doc-id index exists for the requested filter.
    if req.include_deprecated:
        user = get_current_user()
        if user is None or user.role != "admin":
            raise HTTPException(
                status_code=403,
                detail="include_deprecated is admin-only",
            )
    t0 = time.monotonic()
    response, cache_hit = await engine.search(
        query=req.query,
        top_k=req.top_k,
        doc_filter=req.doc_filter.to_filter_dict(),
        hyde=req.hyde,
        include_deprecated=req.include_deprecated,
    )
    latency = (time.monotonic() - t0) * 1000
    top_score = response.results[0].score if response.results else 0.0
    channels = list({c for r in response.results for c in r.match_channels})
    log_search(
        query=req.query,
        latency_ms=latency,
        channels_used=channels,
        result_count=response.total,
        top_score=top_score,
        hyde_triggered=req.hyde,
        cache_hit=cache_hit,
    )
    return response
