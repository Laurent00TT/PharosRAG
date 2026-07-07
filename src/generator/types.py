"""generator 数据契约(纯 stdlib)。"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Message:
    role: str            # system | user | assistant(OpenAI-compatible)
    content: str


@dataclass
class Citation:
    marker: int          # 答案里的 [n]
    chunk_id: str
    doc_id: str
    title: str           # doc_meta.title(引用展示)
    section: str         # section_path / breadcrumb
    page: int
    text: str            # 被引用的 context 文本(溯源)


@dataclass
class Answer:
    text: str                                 # LLM 生成的答案(含 [n] 引用标记)
    citations: list[Citation]                 # 答案实际引用到的来源(去重、按 marker 升序)
    n_contexts: int                           # 喂进 prompt 的 context 数
    # 生成本答案那次 LLM 调用的 finish_reason 快照('length'=被 max_tokens 截断)。随答案走而非读 llm
    # 实例属性 —— 实例级 last_finish_reason 会被后续调用覆盖(smart-ask 重试被弃用时残留第二轮的值,
    # 与返回的第一轮答案错位)。零召回不调 LLM 时为 None。
    finish_reason: str | None = None
    raw_messages: list[Message] = field(default_factory=list)   # 调试:发给 LLM 的 messages
