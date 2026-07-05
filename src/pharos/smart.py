"""smart-ask 产品层(/v1/ask 的"变聪明"逻辑,设计见 DESIGN.md D9):

第 1 层 hints:答案是拒答/部分拒答时,按本次请求参数生成可操作建议(教用户/教 agent 用旋钮);
第 2 层 失败驱动表格补检:数值型问题**第一轮拒答/部分拒答时**,带 kind=table 补检腿重问一轮
(与主检索并集,非替换)。⚠ 不做前置腿:88 题实测,前置腿把表格题 0.625→0.875 的同时
误伤 5 道原本答对的散文题(相近数值带偏/过度保守);失败驱动下答对的题永不触发,零误伤面。

设计红线:不做循环推理(重试硬上限 1 轮;agent 编排在本 workload 上实测净负,那是 MCP 出口的
职责);一切自动行为在响应 `auto` 字段留痕;PHAROS_SMART_ASK=off 一键回纯净模式。
数值判定/拒答判定/补检腿参数全部来自引擎 generator.signals(与 eval --smart-tables 同源,防漂移)。
"""
from __future__ import annotations

import re

from generator import is_refusal as _ir

_CJK = re.compile(r"[一-鿿]")


def is_refusal(answer_text: str) -> bool:
    return _ir(answer_text)


def build_hints(query: str, *, auto: list[str], req_kind, req_rerank: bool, numeric: bool) -> list[str]:
    """拒答时的下一步建议。不重复建议已自动做过/用户已用过的动作;最多 3 条,按性价比排序。"""
    retried = any(a.startswith("table_leg") for a in auto)
    hints: list[str] = []
    if numeric and req_kind is None and not retried:
        hints.append('数值类问题可加 "kind": "table" 只在表格里检索(CLI: --kind table)。')
    if _CJK.search(query or ""):
        hints.append("本库以英文文档为主:数值/术语类问题用文档语言的关键词(如 net income)通常显著提升命中。")
    if not req_rerank and not retried:
        hints.append('可加 "rerank": true 精排纠正排序(慢几秒,对表格/跨语言尤其有效)。')
    hints.append("也可用 /v1/retrieve(mode=concise, top_k 调大)自查正确内容排第几,判断该调哪个旋钮。")
    return hints[:3]
