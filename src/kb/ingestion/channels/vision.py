# src/kb/ingestion/channels/vision.py
"""VisionChannel — single-vector image embedding via multimodal_embedding_server.

PoC v1.1 refactor (spec §3.1):
  - Legacy (removed in v1.1): ColQwen2.5 multi-vector (128 dim/patch via the
    standalone colqwen_server).
  - New (this file): Qwen3-VL-Embedding-8B single-vector (4096 dim) via the
    same vllm-backed multimodal_embedding_server that handles text. Text and
    vision now share one vector space.

Protocol (VERIFIED 2026-05-30 against the live cloud deployment):
  ``POST /v1/embeddings`` with a chat-messages body (vllm's multimodal
  embedding superset of the OpenAI text-only embeddings API):
  ``{"model": ..., "messages": [{"role":"user","content":[
       {"type":"image_url","image_url":{"url":"data:image/png;base64,..."}}]}]}``

  SERVER REQUIREMENT: the embedding server MUST be launched with
  ``--chat-template <qwen3-vl chat template>`` (the template that ships
  with Qwen3-VL-Embedding-8B as ``chat_template.jinja``). Without it vllm
  cannot render the ``messages`` array and returns HTTP 400
  ("default chat template is no longer allowed ... you must provide a
  chat template"). The text channel is unaffected because it uses the
  ``input`` field, which needs no template. See the model startup script and
  the project README §3.6. The template injects a default system
  instruction ("Represent the user's input.") so an image-only content
  array is sufficient — no extra text part is needed on the document side
  (queries carry the instruction prefix, see search/vision_query.py).

Caller compatibility:
  Signature shrinks from ``list[list[float]] | None`` (multi-vec) to
  ``list[float] | None`` (single-vec). Aligns with qdrant_store
  upsert_vision_page after P-6 schema change. pipeline.py callers see
  the new return type directly.
"""
import asyncio
import base64
import logging

import httpx

from kb.config import Settings

logger = logging.getLogger(__name__)


class VisionChannel:
    """Image bytes -> 4096-dim dense vector via multimodal_embedding_server."""

    def __init__(
        self,
        settings: Settings | None = None,
        timeout: float = 60.0,
    ) -> None:
        if settings is None:
            settings = Settings()
        self._url = settings.multimodal_embedding_server_url.rstrip("/")
        self._model_id = settings.multimodal_embedding_model_id
        self._output_dim = settings.embedding_output_dim or None
        # Same lazy-client pattern as TextChannel — see comment there.
        self._timeout = timeout

    async def aclose(self) -> None:  # no-op kept for caller compatibility
        return

    async def embed_async(self, image_bytes: bytes) -> list[float] | None:
        """Embed a single image. Returns None on server failure (caller
        decides whether to degrade — same pattern as the other HTTP channels).

        Protocol: vllm /v1/embeddings is a SUPERSET of OpenAI's text-only
        embeddings API — it also accepts a chat-messages array for
        multimodal inputs (vllm docs: "the schema of messages is exactly
        the same as in Chat Completions API"). We use the chat-messages
        form with type=image_url (NOT type=image — that's not a vllm
        recognized type). Requires the server to be started with a
        --chat-template (see class docstring); otherwise vllm returns 400.
        """
        try:
            b64 = base64.b64encode(image_bytes).decode("ascii")
            # vllm multimodal embedding via chat-messages superset
            payload: dict = {
                "model": self._model_id,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{b64}"},
                            }
                        ],
                    }
                ],
            }
            if self._output_dim:
                payload["dimensions"] = self._output_dim

            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(
                    f"{self._url}/v1/embeddings", json=payload,
                )
                resp.raise_for_status()
                data = resp.json()
            return data["data"][0]["embedding"]
        except httpx.HTTPStatusError as e:
            # Capture the server's error BODY, not just the status line. vllm
            # returns the actionable reason here (e.g. a missing chat-template
            # 400) — logging only the httpx status string hid this root cause
            # for an entire ingest run. See class docstring.
            body = ""
            try:
                body = e.response.text[:500]
            except Exception:  # pragma: no cover - defensive
                pass
            logger.warning(
                "Vision embed failed (HTTP %s: %s); skipping vision channel for this page",
                e.response.status_code, body,
            )
            return None
        except Exception as e:
            logger.warning(
                "Vision embed failed (%s); skipping vision channel for this page",
                e,
            )
            return None

    def embed(self, image_bytes: bytes) -> list[float] | None:
        """Sync entry-point kept for legacy pipeline call sites."""
        return asyncio.run(self.embed_async(image_bytes))
