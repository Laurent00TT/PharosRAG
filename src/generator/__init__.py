"""generator: RAG 的生成层(检索 → LLM 答案合成,带引用 + ACL 出口 + grounding)。

LLM 无关:`LLMClient` 协议可插拔(本地 Qwen / OpenAI-compatible API / Claude…),`MockLLM` 供脚手架验证。
依赖注入:`Generator(retriever, llm)` —— retriever 由 embedder 提供,核心代码零运行时依赖(纯 stdlib)。
"""
from .generate import Generator
from .llm import LLMClient, MockLLM, OpenAICompatibleLLM, messages_to_dicts
from .prompt import PromptBuilder
from .signals import DEFAULT_TABLE_LEG, is_refusal, looks_numeric
from .types import Answer, Citation, Message

__all__ = ["Generator", "LLMClient", "MockLLM", "OpenAICompatibleLLM", "messages_to_dicts",
           "PromptBuilder", "Answer", "Citation", "Message",
           "looks_numeric", "is_refusal", "DEFAULT_TABLE_LEG"]
