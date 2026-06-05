import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator

RESOURCE_MINERU_GPU = "mineru_gpu"
RESOURCE_DESCRIPTION_LLM = "description_llm"
RESOURCE_TEXT_EMBEDDING = "text_embedding"

DEFAULT_RESOURCE_LIMITS = {
    RESOURCE_MINERU_GPU: 1,
    RESOURCE_DESCRIPTION_LLM: 1,
    RESOURCE_TEXT_EMBEDDING: 1,
}


class ResourceManager:
    def __init__(self, limits: dict[str, int] | None = None) -> None:
        merged = {**DEFAULT_RESOURCE_LIMITS, **(limits or {})}
        self._locks = {
            name: asyncio.Semaphore(max(1, limit))
            for name, limit in merged.items()
        }

    @property
    def default_resources(self) -> tuple[str, ...]:
        return tuple(DEFAULT_RESOURCE_LIMITS)

    @asynccontextmanager
    async def acquire_many(self, names: list[str]) -> AsyncIterator[None]:
        acquired: list[str] = []
        try:
            for name in sorted(set(names)):
                sem = self._locks.setdefault(name, asyncio.Semaphore(1))
                await sem.acquire()
                acquired.append(name)
            yield
        finally:
            for name in reversed(acquired):
                self._locks[name].release()
