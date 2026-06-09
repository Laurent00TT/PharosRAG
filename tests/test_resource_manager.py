import asyncio

import pytest


@pytest.mark.asyncio
async def test_resource_manager_serializes_same_resource():
    from kb.jobs.resource_manager import ResourceManager

    rm = ResourceManager({"mineru_gpu": 1})
    inside = 0
    max_inside = 0

    async def task():
        nonlocal inside, max_inside
        async with rm.acquire_many(["mineru_gpu"]):
            inside += 1
            max_inside = max(max_inside, inside)
            await asyncio.sleep(0)
            inside -= 1

    await asyncio.gather(task(), task())

    assert max_inside == 1


@pytest.mark.asyncio
async def test_resource_manager_allows_different_resources_in_parallel():
    from kb.jobs.resource_manager import ResourceManager

    rm = ResourceManager({"mineru_gpu": 1, "description_llm": 1})
    started = asyncio.Event()
    both_inside = asyncio.Event()

    async def first():
        async with rm.acquire_many(["mineru_gpu"]):
            started.set()
            await both_inside.wait()

    async def second():
        await started.wait()
        async with rm.acquire_many(["description_llm"]):
            both_inside.set()

    await asyncio.wait_for(asyncio.gather(first(), second()), timeout=1)


@pytest.mark.asyncio
async def test_resource_manager_acquires_many_in_stable_order():
    from kb.jobs.resource_manager import ResourceManager

    rm = ResourceManager({"a": 1, "b": 1})

    async def task(names):
        async with rm.acquire_many(names):
            await asyncio.sleep(0)

    await asyncio.wait_for(
        asyncio.gather(
            task(["a", "b"]),
            task(["b", "a"]),
        ),
        timeout=1,
    )
