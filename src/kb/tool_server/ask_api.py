import inspect
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from kb.adapters.ollama import OllamaAdapter
from kb.agent.agent import KBAgent
from kb.tool_server.security import verify_api_key

router = APIRouter(dependencies=[Depends(verify_api_key)])


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    adapter: Literal["ollama", "openai_compat"] = "openai_compat"
    ollama_model: str = Field(default="gemma2", min_length=1, max_length=200)
    openai_compat_base_url: str | None = None
    openai_compat_api_key: str | None = None
    openai_compat_model: str | None = None
    openai_compat_supports_vision: bool = False
    max_iterations: int = Field(default=5, ge=1, le=10)


def _build_adapter(req: AskRequest, settings):
    from kb.adapters.factory import make_chat_adapter

    if req.adapter == "ollama":
        return OllamaAdapter(model=req.ollama_model, task_type="answer")

    if req.adapter == "openai_compat":
        return make_chat_adapter(
            adapter_type="openai_compat",
            url=req.openai_compat_base_url or settings.agent_url,
            api_key=req.openai_compat_api_key or settings.agent_api_key,
            model=req.openai_compat_model or settings.agent_model,
            supports_vision=(
                req.openai_compat_supports_vision
                or settings.agent_supports_vision
            ),
            disable_thinking=False,
            task_type="answer",  # Phase 6.4 — kb.llm.call attribution
        )

    raise HTTPException(
        status_code=400,
        detail=f"Unknown adapter: {req.adapter!r}. "
               f"Expected one of: ollama, openai_compat.",
    )


@router.post("/ask")
async def ask(req: AskRequest, request: Request):
    settings = request.app.state.settings
    adapter = _build_adapter(req, settings)
    try:
        agent = KBAgent(
            adapter=adapter,
            engine=request.app.state.engine,
            feedback_db=request.app.state.feedback_db,
            max_iterations=req.max_iterations,
            settings=settings,
        )
        result = await agent.answer(req.question)
        return {
            "answer": result["answer"],
            "citations": [
                {"doc_id": c.doc_id, "page_num": c.page_num}
                for c in result["citations"]
            ],
            "citations_hallucinated": [
                {"doc_id": c.doc_id, "page_num": c.page_num}
                for c in result.get("citations_hallucinated", [])
            ],
            "retrieved": [list(t) for t in result["retrieved"]],
            "query_id": result.get("query_id"),
            "confidence": result.get("confidence"),
        }
    finally:
        # Use a separate local so we don't shadow `result` from the try block —
        # that would pollute exception context if anything before this raised.
        maybe_close = getattr(adapter, "aclose", None)
        if maybe_close is not None:
            close_coro = maybe_close()
            if inspect.isawaitable(close_coro):
                await close_coro
