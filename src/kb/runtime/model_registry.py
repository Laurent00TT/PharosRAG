"""Model registry — multi-dimensional key cache, thread-safe, no failure pollution.

Ported from MinerU's AtomModelSingleton, with a few key changes:
1. On init failure, raises ModelInitError instead of exit(1) (a FastAPI process must not crash).
2. Failed keys are not cached, so the next call retries.
3. Generic — any (name, **kwargs) -> factory call works.

NOTE on integration status: TextChannel and Reranker now use the process-global
registry to avoid duplicate model loads in combined serve/ingest processes.
This becomes especially useful in scenarios like:
  - ingestion + serve in the same process (avoid duplicate copies)
  - multiple uvicorn workers sharing one model (per-worker load wastes VRAM)
  - same model with multiple configs (e.g. reranker on different devices)
At that point, switch TextChannel / Reranker to fetch via ModelRegistry.get(...).
"""
import logging
import threading
from typing import Any, Callable

logger = logging.getLogger(__name__)


class ModelInitError(RuntimeError):
    """Model initialization failure."""


class ModelRegistry:
    """In-process singleton that caches model instances by multi-dimensional key.

    Usage:
        reg = ModelRegistry()
        bge = reg.get("bge-m3", lambda name, **kw: BGEM3FlagModel("BAAI/bge-m3"))
        ocr = reg.get("ocr", build_ocr, lang="ch", det_threshold=0.3)
    """

    def __init__(self) -> None:
        self._models: dict[tuple, Any] = {}
        self._lock = threading.RLock()

    def get(
        self,
        name: str,
        factory: Callable[..., Any],
        **kwargs,
    ) -> Any:
        """Get or create a model keyed by (name, **kwargs).

        factory signature: factory(name: str, **kwargs) -> model_instance
        """
        key = self._make_key(name, kwargs)
        with self._lock:
            if key not in self._models:
                try:
                    self._models[key] = factory(name, **kwargs)
                    logger.info("Model %s initialized (key=%s)", name, key)
                except Exception as e:
                    logger.error("Model %s init failed: %s", name, e)
                    raise ModelInitError(f"Failed to initialize {name}: {e}") from e
            return self._models[key]

    def evict(self, name: str, **kwargs) -> None:
        """Remove a model from cache. Does not call unload — the factory's
        instance is expected to handle its own destruction."""
        key = self._make_key(name, kwargs)
        with self._lock:
            self._models.pop(key, None)

    def clear(self) -> None:
        """Clear the entire cache."""
        with self._lock:
            self._models.clear()

    @staticmethod
    def _make_key(name: str, kwargs: dict) -> tuple:
        # Sort kwargs so dict-iteration order does not affect the key
        return (name,) + tuple(sorted(kwargs.items()))


_GLOBAL_MODEL_REGISTRY = ModelRegistry()


def get_model_registry() -> ModelRegistry:
    return _GLOBAL_MODEL_REGISTRY
