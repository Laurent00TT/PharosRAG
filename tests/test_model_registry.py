import threading
import pytest
from kb.runtime.model_registry import ModelRegistry, ModelInitError


def test_registry_caches_by_key():
    registry = ModelRegistry()
    call_count = 0

    def factory(name: str, **kw):
        nonlocal call_count
        call_count += 1
        return f"model_{name}_{kw.get('version', 'v1')}"

    m1 = registry.get("text_emb", factory, version="v1")
    m2 = registry.get("text_emb", factory, version="v1")
    assert m1 is m2
    assert call_count == 1


def test_registry_different_keys_different_models():
    registry = ModelRegistry()
    factory = lambda name, **kw: f"{name}-{kw.get('lang', 'en')}"
    m1 = registry.get("ocr", factory, lang="en")
    m2 = registry.get("ocr", factory, lang="zh")
    assert m1 != m2


def test_registry_failure_does_not_pollute_cache():
    registry = ModelRegistry()
    attempts = [0]

    def flaky_factory(name: str, **kw):
        attempts[0] += 1
        if attempts[0] == 1:
            raise RuntimeError("init fail")
        return "ok"

    # 第一次失败应抛出 ModelInitError
    with pytest.raises(ModelInitError):
        registry.get("xx", flaky_factory)

    # 第二次能成功（因为失败的没被缓存）
    result = registry.get("xx", flaky_factory)
    assert result == "ok"


def test_registry_thread_safe():
    registry = ModelRegistry()
    call_count = [0]

    def slow_factory(name: str, **kw):
        import time
        time.sleep(0.05)
        call_count[0] += 1
        return f"model_{name}"

    results = []
    def worker():
        m = registry.get("shared", slow_factory)
        results.append(m)

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads: t.start()
    for t in threads: t.join()

    assert call_count[0] == 1, f"应仅初始化一次，实际 {call_count[0]} 次"
    assert all(r == results[0] for r in results)


def test_registry_evict_removes_from_cache():
    registry = ModelRegistry()
    call_count = 0

    def factory(name: str, **kw):
        nonlocal call_count
        call_count += 1
        return f"model_{call_count}"

    m1 = registry.get("xx", factory, lang="en")
    registry.evict("xx", lang="en")
    m2 = registry.get("xx", factory, lang="en")
    assert m1 != m2  # evict 后重新创建


def test_registry_clear():
    registry = ModelRegistry()
    factory = lambda name, **kw: object()
    registry.get("a", factory)
    registry.get("b", factory)
    registry.clear()
    # 重新 get 应该返回新对象
    a = registry.get("a", factory)
    assert a is not None


def test_global_model_registry_is_singleton():
    from kb.runtime.model_registry import get_model_registry

    assert get_model_registry() is get_model_registry()


def test_reranker_uses_global_model_registry(mocker):
    from kb.runtime.model_registry import get_model_registry
    from kb.search.reranker import Reranker

    registry = get_model_registry()
    registry.clear()
    fake_model = object()
    cls = mocker.patch("kb.search.reranker.FlagReranker", return_value=fake_model)

    a = Reranker(model_name="reranker-test", device="cpu")
    b = Reranker(model_name="reranker-test", device="cpu")

    assert a._model is fake_model
    assert b._model is fake_model
    cls.assert_called_once()
    registry.clear()


def test_text_channel_uses_global_model_registry(mocker):
    from kb.ingestion.channels.text import TextChannel
    from kb.runtime.model_registry import get_model_registry

    registry = get_model_registry()
    registry.clear()
    fake_model = object()
    cls = mocker.patch("kb.ingestion.channels.text.BGEM3FlagModel", return_value=fake_model)

    a = TextChannel(model_name="bge-test")
    b = TextChannel(model_name="bge-test")

    assert a._model is fake_model
    assert b._model is fake_model
    cls.assert_called_once()
    registry.clear()
