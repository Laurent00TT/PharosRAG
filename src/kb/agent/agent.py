# src/kb/agent/agent.py
import json
import logging
import time

from kb.adapters.base import AdapterMessage, BaseLLMAdapter
from kb.agent.citation import Citation, parse_citations, validate_citations
from kb.agent.prompts import AGENT_PROMPT_VERSION, SEARCH_DOCS_TOOL_DEF, SYSTEM_PROMPT
from kb.config import Settings
from kb.feedback.db import UsageFeedbackDB
from kb.models import SearchResponse
from kb.observability import (
    AgentFinalEvent,
    AgentIterationEvent,
    emit,
    preview,
)
from kb.search.engine import SearchEngine

logger = logging.getLogger(__name__)


class KBAgent:
    def __init__(
        self,
        adapter: BaseLLMAdapter,
        engine: SearchEngine,
        feedback_db: UsageFeedbackDB | None = None,
        max_iterations: int = 5,
        settings: Settings | None = None,
    ) -> None:
        self._adapter = adapter
        self._engine = engine
        self._feedback_db = feedback_db
        self._max_iter = max_iterations
        self._settings = settings or Settings()
        self._attached_image_keys: set[tuple[str, int]] = set()

    async def answer(self, question: str) -> dict:
        """Run the LLM tool-calling loop until a final answer is produced.
        Returns: {answer, citations, retrieved}."""
        emit(
            "agent.start",
            question=preview(question),
            max_iterations=self._max_iter,
            supports_vision=self._adapter.supports_vision(),
            prompt_version=AGENT_PROMPT_VERSION,
        )
        messages: list[AdapterMessage] = [
            AdapterMessage(role="system", content=SYSTEM_PROMPT),
            AdapterMessage(role="user", content=question),
        ]
        retrieved_pages: list[tuple[str, int]] = []
        rounds_completed = 0
        query_id: str | None = None
        confidence_reasons: set[str] = set()
        search_attempted = False

        for round_idx in range(self._max_iter):
            t_iter = time.monotonic()
            # ctx pre-flight estimate: rough char/4 ≈ tokens for text, +1500/img.
            # Goal is observability, not enforcement — we trust llama-server's 64K ctx.
            est = _estimate_history_tokens(messages)
            emit(
                "agent.ctx.snapshot",
                round=round_idx + 1,
                history_messages=len(messages),
                estimated_tokens=est["total"],
                text_tokens=est["text"],
                image_tokens=est["image"],
                images_in_history=est["image_count"],
            )
            response = await self._adapter.chat(
                messages, tools=[SEARCH_DOCS_TOOL_DEF]
            )
            rounds_completed = round_idx + 1
            emit(AgentIterationEvent(
                round=rounds_completed,
                duration_ms=round((time.monotonic() - t_iter) * 1000, 1),
                thinking_detected=response.thinking_detected,
                thinking_chars=response.thinking_chars,
                content_chars=response.content_chars,
                total_tokens=response.total_tokens,
                tool_calls_count=len(response.tool_calls),
                finish_reason="tool_call" if response.tool_calls else "final_answer",
            ))

            if not response.tool_calls:
                raw_citations = parse_citations(response.content)
                valid_citations, hallucinated = validate_citations(
                    raw_citations, retrieved_pages
                )
                if retrieved_pages and not valid_citations:
                    confidence_reasons.add("no_valid_citations")
                if hallucinated:
                    confidence_reasons.add("hallucinated_citations_removed")
                if search_attempted and not retrieved_pages:
                    confidence_reasons.add("no_retrieved_pages")
                confidence = _build_confidence_summary(confidence_reasons)
                final_answer = _apply_uncertainty_prefix(response.content, confidence)
                actually_used = [(c.doc_id, c.page_num) for c in valid_citations]
                if self._feedback_db is not None:
                    query_id = await self._feedback_db.record(
                        query=question,
                        retrieved=retrieved_pages,
                        actually_used=actually_used,
                    )
                if hallucinated:
                    emit(
                        "agent.hallucinated_citation",
                        count=len(hallucinated),
                        examples=[
                            f"{c.doc_id[:10]}:{c.page_num}" for c in hallucinated[:3]
                        ],
                    )
                emit(AgentFinalEvent(
                    iterations=rounds_completed,
                    citations_count=len(valid_citations),
                    citations_hallucinated_count=len(hallucinated),
                    answer_chars=len(final_answer or ""),
                    answer_preview=preview(final_answer or ""),
                    retrieved_count=len(retrieved_pages),
                    success=True,
                ))
                return {
                    "answer": final_answer,
                    "citations": valid_citations,
                    "citations_hallucinated": hallucinated,
                    "retrieved": retrieved_pages,
                    "query_id": query_id,
                    "confidence": confidence,
                }

            messages.append(AdapterMessage(
                role="assistant",
                content=response.content or "",
                tool_calls=response.tool_calls,
            ))

            for tc in response.tool_calls:
                search_resp = None
                # Unified tool result schema:
                #   {"status": "ok" | "error", "results": [...], "total": N, "error": str?}
                # Always populate `results` (empty on error) so the LLM sees a
                # consistent shape regardless of success/failure.
                if tc.name != "search_docs":
                    tool_payload = {
                        "status": "error",
                        "error": f"unknown tool: {tc.name}",
                        "results": [],
                        "total": 0,
                    }
                    emit(
                        "agent.tool_call",
                        tool_name=tc.name,
                        recognized=False,
                        round=rounds_completed,
                    )
                else:
                    args = tc.arguments
                    if "query" not in args:
                        # LLM forgot the required arg — return error to it instead of crashing
                        tool_payload = {
                            "status": "error",
                            "error": "missing required argument: query",
                            "results": [],
                            "total": 0,
                        }
                        emit(
                            "agent.tool_call",
                            tool_name="search_docs",
                            recognized=True,
                            args_ok=False,
                            args_error="missing query",
                            round=rounds_completed,
                        )
                    else:
                        emit(
                            "agent.tool_call",
                            tool_name="search_docs",
                            recognized=True,
                            args_ok=True,
                            query_preview=preview(str(args.get("query", ""))),
                            top_k=args.get("top_k", 5),
                            round=rounds_completed,
                        )
                        search_resp, _ = await self._engine.search(
                            query=args["query"],
                            top_k=args.get("top_k", 5),
                            doc_filter=args.get("doc_filter", {}),
                        )
                        search_attempted = True
                        if search_resp.low_confidence:
                            confidence_reasons.add(
                                search_resp.confidence_reason or "low_confidence"
                            )
                        for r in search_resp.results:
                            retrieved_pages.append((r.doc_id, r.page_num))
                        tool_payload = self._format_search_result(search_resp)

                messages.append(AdapterMessage(
                    role="tool",
                    content=json.dumps(tool_payload, ensure_ascii=False),
                    tool_call_id=tc.id,
                ))

                # True multimodal: vision-capable adapter receives the image bytes
                if (
                    self._adapter.supports_vision()
                    and tc.name == "search_docs"
                    and search_resp is not None
                    and search_resp.results
                ):
                    multimodal_msg = self._build_multimodal_message(search_resp.results)
                    if multimodal_msg is not None:
                        messages.append(multimodal_msg)

        logger.warning("Agent hit max_iterations=%d without final answer", self._max_iter)
        emit(AgentFinalEvent(
            iterations=rounds_completed,
            citations_count=0,
            answer_chars=0,
            retrieved_count=len(retrieved_pages),
            success=False,
            failure_reason="max_iterations_reached",
        ))
        return {
            "answer": "I was unable to find a complete answer within the iteration limit.",
            "citations": [],
            "retrieved": retrieved_pages,
            "query_id": None,
            "confidence": _build_confidence_summary({"max_iterations_reached"}),
        }

    def _build_multimodal_message(self, results: list) -> AdapterMessage | None:
        """Collect image bytes from search results and build one multimodal user message.
        Returns None when no images can be loaded."""
        from kb.adapters.base import ImagePart, build_multimodal_content

        images: list[ImagePart] = []
        descriptions: list[str] = []
        total_bytes = 0
        results_with_url = sum(1 for r in results if r.image_url)
        skipped_by_count = 0
        skipped_by_bytes = 0
        skipped_duplicate = 0
        skipped_load_failed = 0
        max_images = self._settings.agent_max_images_per_turn
        max_bytes = self._settings.agent_max_image_bytes_per_turn
        for r in results:
            if not r.image_url:
                continue
            key = (r.doc_id, r.page_num)
            if key in self._attached_image_keys:
                skipped_duplicate += 1
                continue
            if len(images) >= max_images:
                skipped_by_count += 1
                continue
            img_bytes = self._load_image_bytes(r.image_url)
            if not img_bytes:
                skipped_load_failed += 1
                continue
            if total_bytes + len(img_bytes) > max_bytes:
                skipped_by_bytes += 1
                continue
            images.append(ImagePart(data=img_bytes, mime_type="image/png"))
            descriptions.append(
                f"- doc {r.doc_name} page {r.page_num} [doc_id={r.doc_id}]"
            )
            total_bytes += len(img_bytes)
            self._attached_image_keys.add(key)
        emit(
            "agent.multimodal_budget",
            images_attached=len(images),
            results_with_url=results_with_url,
            total_image_bytes=total_bytes,
            max_images=max_images,
            max_image_bytes=max_bytes,
            skipped_by_count=skipped_by_count,
            skipped_by_bytes=skipped_by_bytes,
            skipped_duplicate=skipped_duplicate,
            skipped_load_failed=skipped_load_failed,
        )
        if not images:
            emit(
                "agent.multimodal_attached",
                images_count=0,
                results_with_url=results_with_url,
                attached=False,
                reason="no_loadable_images",
            )
            return None

        emit(
            "agent.multimodal_attached",
            images_count=len(images),
            results_with_url=results_with_url,
            total_image_bytes=total_bytes,
            attached=True,
        )
        text = (
            "Below are the original page images for the search_docs results above. "
            "Use them together with the text to answer:"
            + "\n" + "\n".join(descriptions)
        )
        return AdapterMessage(
            role="user",
            content=build_multimodal_content(text=text, images=images),
        )

    def _load_image_bytes(self, image_url: str) -> bytes | None:
        """Load image bytes from an image_url.

        Supports local:// (local path), file://, and bare local filesystem paths.
        http(s):// remote URLs are deliberately not supported (interface stub).
        """
        from pathlib import Path
        if image_url.startswith("local://"):
            path_str = image_url[len("local://"):]
            try:
                return Path(path_str).read_bytes()
            except Exception as e:
                logger.warning("Failed to read local image %s: %s", image_url, e)
                return None
        elif image_url.startswith("file://"):
            path_str = image_url[len("file://"):]
            try:
                return Path(path_str).read_bytes()
            except Exception as e:
                logger.warning("Failed to read file:// image %s: %s", image_url, e)
                return None
        elif image_url.startswith(("http://", "https://")):
            # Remote URLs — not supported in this version
            logger.debug("Unsupported image_url scheme: %s (skipping multimodal)", image_url)
            return None
        else:
            # Treat as a bare local filesystem path
            try:
                p = Path(image_url)
                if p.is_file():
                    return p.read_bytes()
                logger.debug("image_url is not a valid local path: %s", image_url)
                return None
            except Exception as e:
                logger.warning("Failed to read local path %s: %s", image_url, e)
                return None

    def _format_search_result(self, resp: SearchResponse) -> dict:
        results_for_llm = []
        for r in resp.results:
            entry = {
                # Pre-built citation marker — LLM should copy this verbatim into its answer.
                # Listed first so it's the most visible field.
                "citation_marker": f"[{r.doc_id}:{r.page_num}]",
                "doc_id": r.doc_id,
                "doc_name": r.doc_name,
                "page_num": r.page_num,
                "heading_path": r.heading_path,
                "text": r.text_chunk,
                "score": round(r.score, 4),
            }
            # Include image_url AND vision_description independently — both are useful
            # context for the LLM (image gives detail; description gives semantic summary).
            # The actual multimodal payload is appended as a separate role="user" multipart
            # message via _build_multimodal_message; the image_url in this tool payload is
            # only a citation hint for the LLM.
            if self._adapter.supports_vision() and r.image_url:
                entry["image_url"] = r.image_url
            if r.vision_description:
                entry["vision_description"] = r.vision_description
            results_for_llm.append(entry)
        # Unified schema with the error path: status + results + total
        payload: dict = {
            "status": "ok",
            "results": results_for_llm,
            "total": resp.total,
        }
        # Propagate cold-start UX fields so the LLM can tell the user the KB is empty
        if resp.message:
            payload["message"] = resp.message
        if resp.suggested_upload_topics:
            payload["suggested_upload_topics"] = resp.suggested_upload_topics
        if resp.low_confidence:
            payload["low_confidence"] = True
            payload["confidence_reason"] = resp.confidence_reason
        return payload


def _build_confidence_summary(reasons: set[str]) -> dict:
    ordered = sorted(reason for reason in reasons if reason)
    return {
        "status": "low" if ordered else "grounded",
        "reasons": ordered,
    }


def _apply_uncertainty_prefix(answer: str, confidence: dict) -> str:
    if confidence.get("status") != "low":
        return answer
    if "知识库证据不足" in answer:
        return answer
    return f"知识库证据不足，以下回答仅供参考：{answer}"


# Per-image token cost estimate. Qwen2.5-VL / Qwen3.6 vision encoders produce
# ~1500 patch tokens per image at typical PDF page resolution. This is a
# rough order-of-magnitude figure for ctx accounting only.
_IMG_TOKEN_COST = 1500


def _estimate_history_tokens(messages: list[AdapterMessage]) -> dict:
    """Rough char/4 + image-count tokenization estimate.

    Returns {"total", "text", "image", "image_count"}. Used for trace-level
    ctx-budget monitoring; not authoritative — server's actual tokenizer
    differs by ~10-30%.
    """
    text_tokens = 0
    image_tokens = 0
    image_count = 0
    for m in messages:
        content = m.content
        if isinstance(content, str):
            text_tokens += len(content) // 4
        elif isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") == "text":
                    text_tokens += len(part.get("text", "")) // 4
                elif part.get("type") == "image_url":
                    image_tokens += _IMG_TOKEN_COST
                    image_count += 1
    return {
        "total": text_tokens + image_tokens,
        "text": text_tokens,
        "image": image_tokens,
        "image_count": image_count,
    }
