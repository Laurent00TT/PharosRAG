"""embedder 数据契约。"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class User:
    """检索主体 = ACL 硬过滤的输入。tenant 隔离 + principals(用户的 groups + 自身 id)。
    对应 chunker 的 acl 契约:filter 放行 acl_tenant==user.tenant 且 (acl_allow ∩ principals 非空 或 public)。"""
    tenant: str
    principals: list[str]


@dataclass
class Hit:
    """一条检索命中(hybrid + ACL filter 后)。payload 携带 small-to-big / 出口引用所需字段。"""
    chunk_id: str
    doc_id: str
    kind: str
    text: str
    score: float
    payload: dict = field(default_factory=dict)   # doc_meta / image_path / source_indices / section_anchor / ...
    # 分数口径:'rrf' = hybrid 融合分(嵌入式 local 用 RRF k=2,真 server k=60 量级骤降;随名次,不可跨查询比/不可归一);
    # 'rerank' = cross-encoder sigmoid 0~1。Batch1.B:rerank 后必须把 score+score_kind 一起改写,否则名次与分数脱节。
    score_kind: str = "rrf"
