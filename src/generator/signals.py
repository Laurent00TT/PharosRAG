"""查询信号:轻量规则判定(零 LLM 调用),供闭管道的产品层做便宜路由。

Pharos smart-ask 与 eval --smart-tables 共用本模块(单一来源,防两处词表漂移)。
判定故意偏宽松:误触发的代价只是多并集几段表格(便宜),漏触发的代价是数值题答不全(贵)。
"""
from __future__ import annotations

import re

_NUM_HINT = re.compile(
    r"[0-9０-９]"                                                    # 阿拉伯数字(含全角)
    r"|[一二三四五六七八九十百千万亿两]+\s*(?:年|月|倍|个|次|款|家|人|元)"   # 中文数量词
    r"|多少|几[个年月倍成]|占比|比例|比率|增长|涨跌|同比|环比"
    r"|营收|收入|利润|成本|费用|金额|价格|市值|销量|净额|余额"
    r"|how\s+(?:much|many)|percent|revenue|income|profit|price|cost|amount|total",
    re.I)

# 数值题的推荐表格补检腿(与主检索**并集**,非替换):kind=table 清散文竞争,
# 小池精排跨语言纠偏(实测:中文逐年净利润题,五年表从全库 20 名外 -> 第 4)。
# ⚠ 用法必须是**失败驱动**(第一轮拒答才带腿重试),不能前置:88 题实测,前置腿把表格题
# 0.625->0.875 的同时误伤 5 道原本答对的散文题(相近数值带偏/过度保守),散文 0.861->0.792;
# 失败驱动下答对的题永不触发,零误伤面。留档见 pharos TESTING §3。
# rerank_top_n=50(默认全深度):实测教训——30 时五年表(混合粗排 31-50 名)进不了精排池,
# 重试腿形同虚设;精排池深度必须 >= "对的块在粗排的最差名次",表格题跨语言场景实测要 50。
DEFAULT_TABLE_LEG = {"kind": "table", "top_k": 5, "rerank": True, "rerank_top_n": 50}

# 拒答/部分拒答模式(中英)。pharos smart-ask 与 eval --smart-tables 共用(单一来源)。
_REFUSAL = re.compile(
    r"信息不足|没有足够|无法回答|无法得出|无法计算|未提供|没有提供|没有相关|没有找到|查不到|找不到"
    r"|don'?t have enough|not contain|no (?:relevant )?information|unable to answer|cannot answer"
    r"|not explicitly stated|not (?:directly )?provided",
    re.I)


def looks_numeric(query: str) -> bool:
    """问题是否大概率要用数值回答(数字/数量词/金额财务词/how much…)。"""
    return bool(_NUM_HINT.search(query or ""))


def is_refusal(answer_text: str) -> bool:
    """答案是否为拒答/部分拒答(含"没有 X 年数据"这类部分缺失声明)。"""
    return bool(_REFUSAL.search(answer_text or ""))
