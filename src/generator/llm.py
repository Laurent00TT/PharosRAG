"""LLM 客户端:可插拔协议 + MockLLM(脚手架验证)+ OpenAICompatibleLLM(真后端)。

真实现只需 `complete(messages) -> str`。接口用 OpenAI-compatible 的 messages 形态,本地(vLLM)与各家 API 都兼容。
"""
from __future__ import annotations

import os
import re
from typing import Protocol

from .types import Message


class LLMClient(Protocol):
    """任何实现 complete(messages)->str 的对象都能插进 Generator。"""

    def complete(self, messages: list[Message]) -> str: ...


class MockLLM:
    """假 LLM:不调真模型,从 prompt 的 context 编号回显一个带引用的答案,用于验证 generator 脚手架。
    行为:引用前 min(2, 去重编号数) 个 context;无 context 时回"信息不足"(测 grounding 退路)。"""

    def complete(self, messages: list[Message]) -> str:
        user = next((m.content for m in messages if m.role == "user"), "")
        # 只数 PromptBuilder 加的 [cite:n](正文裸 [n] 不算)-> 判定基于真实 context 数,不被正文污染(review#2)
        markers = sorted({int(x) for x in re.findall(r"\[cite:(\d+)\]", user)})
        if not markers:
            return "I don't have enough information in the provided context to answer."
        refs = "".join(f"[cite:{n}]" for n in markers[:2])
        return f"Based on the retrieved context {refs}, here is a grounded mock answer citing the provided sources."


def messages_to_dicts(messages: list[Message]) -> list[dict]:
    """把 list[Message] 转成各 LLM API 期望的 [{'role','content'}]。真后端(OpenAI/vLLM/Claude)实现 complete 时复用,
    避免每个后端各写一份易错转换(review#5)。"""
    return [{"role": m.role, "content": m.content} for m in messages]


class OpenAICompatibleLLM:
    """任何 OpenAI 兼容 chat/completions 端点的 LLMClient —— DeepSeek、GLM 代理、本地 vLLM 通吃。
    **型号/端点/思考开关全走配置,不写死**:换后端 = 改 base_url+model,complete() 一行不动。需 `pip install openai`。

    思考模式(DeepSeek V4):`thinking=True` -> extra_body {"thinking":{"type":"enabled"}};关时**显式发 disabled**
    —— V4 Flash 思考可能默认开,不显式关会要求回传 reasoning_content 而 400。思考链走 `reasoning_content`,与 `content`
    分开 -> 不污染答案的 [cite:n] 格式;每次调用存到 self.last_reasoning(分开,默认不喂回、不展示)。

    评估三旋钮:model(flash/pro)、thinking(on/off)、reasoning_effort(high/max)—— eval 里一把扫,数据定档。
    grounded RAG 默认 temperature=0(忠实、可复现)。key 从 api_key= 或环境变量(默认 DEEPSEEK_API_KEY)读,放 .env 勿提交。"""

    def __init__(self, *, model: str, base_url: str = "https://api.deepseek.com",
                 api_key: str | None = None, api_key_env: str = "DEEPSEEK_API_KEY",
                 thinking: bool = False, reasoning_effort: str | None = None,
                 max_tokens: int = 2000, temperature: float = 0.0, timeout: float = 120,
                 send_thinking: bool | None = None):
        from openai import OpenAI       # lazy:不用此后端就不强求装 openai(对齐包的 import 卫生)
        key = api_key or os.environ.get(api_key_env)
        if not key:
            raise ValueError(f"缺 API key:传 api_key= 或设环境变量 {api_key_env}(建议放 .env,勿提交)。")
        self._client = OpenAI(base_url=base_url, api_key=key, timeout=timeout)
        self.model = model
        self.thinking = thinking
        self.reasoning_effort = reasoning_effort
        self.max_tokens = max_tokens
        self.temperature = temperature
        # R3.C:thinking 是 DeepSeek 专有 extra_body 字段;原生 OpenAI/多数 vLLM/其它兼容代理对未知 body 字段会 400。
        # 故只对 DeepSeek 系后端注入(默认按 base_url 自动判定,可显式 send_thinking= 覆盖),保住"OpenAI 兼容通吃"契约。
        self.send_thinking = send_thinking if send_thinking is not None else ("deepseek" in base_url.lower())
        self.last_reasoning: str | None = None      # 最近一次思考链(reasoning_content),分开存,不进答案
        self.last_finish_reason: str | None = None  # R3.B:最近一次结束原因;=='length' 表示答案被 max_tokens 截断(尾部 [cite:n] 可能被切)

    def complete(self, messages: list[Message]) -> str:
        extra: dict = {}
        if self.send_thinking:
            extra["thinking"] = {"type": "enabled" if self.thinking else "disabled"}
            if self.thinking and self.reasoning_effort:
                extra["reasoning_effort"] = self.reasoning_effort
        resp = self._client.chat.completions.create(
            model=self.model, messages=messages_to_dicts(messages),
            max_tokens=self.max_tokens, temperature=self.temperature, extra_body=extra or None)
        # R3.F:内容审查命中/上游异常可能被包成 choices=[]/None -> 明确报错,不 IndexError、不静默空答(与"模型正常答空"区分)
        choice = resp.choices[0] if getattr(resp, "choices", None) else None
        if choice is None:
            raise RuntimeError("LLM 返回空 choices(可能内容审查命中或上游异常);无法作答")
        self.last_finish_reason = getattr(choice, "finish_reason", None)   # R3.B:透出截断信号供调用方/评估过滤
        self.last_reasoning = getattr(choice.message, "reasoning_content", None)   # 思考链分离,不混进 content
        return choice.message.content or ""
