from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from kb.tool_server.security import verify_api_key

router = APIRouter(dependencies=[Depends(verify_api_key)])


class FeedbackRequest(BaseModel):
    query_id: str = Field(min_length=1, max_length=200)
    was_helpful: bool
    comment: str = Field(default="", max_length=2000)
    expected_doc: str = Field(default="", max_length=500)


@router.post("/feedback")
async def record_feedback(req: FeedbackRequest, request: Request):
    feedback_db = getattr(request.app.state, "feedback_db", None)
    if feedback_db is None:
        raise HTTPException(status_code=503, detail="Feedback DB is not initialized")

    updated = await feedback_db.update_feedback(
        query_id=req.query_id,
        was_helpful=req.was_helpful,
        comment=req.comment,
        expected_doc=req.expected_doc,
    )
    if not updated:
        raise HTTPException(status_code=404, detail="query_id not found")
    return {"updated": True}
