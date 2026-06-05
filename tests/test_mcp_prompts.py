from __future__ import annotations

from kb.tool_server.mcp_prompts import KB_ANSWER_FROM_ORIGINALS


def test_answer_prompt_requires_fetching_originals():
    assert "nav_hits as navigation pointers only" in KB_ANSWER_FROM_ORIGINALS
    assert "Read suggested_fetches resources before answering" in KB_ANSWER_FROM_ORIGINALS
    assert "[doc_id:page_num]" in KB_ANSWER_FROM_ORIGINALS


def test_register_prompts_attaches_prompt():
    from kb.tool_server.mcp_prompts import register_prompts

    class _FakeMCP:
        def __init__(self):
            self.prompts = {}

        def prompt(self):
            def decorator(fn):
                self.prompts[fn.__name__] = fn
                return fn
            return decorator

    fake = _FakeMCP()
    register_prompts(fake, state=None)
    assert "kb_answer_from_originals" in fake.prompts
    assert fake.prompts["kb_answer_from_originals"]() == KB_ANSWER_FROM_ORIGINALS
