"""ACL 客户端逻辑(集中一处便审计)。chunk.acl 是自由 dict,这里做拆解 + 可见性判断。

两处用途必须与 `store.acl_filter`(Qdrant 服务端硬过滤)**严格同语义**:
  - `acl_split`:索引期把 acl 拆成 store 的 4 个可过滤 payload 字段;
  - `acl_admits`:检索期 small-to-big 取材的逐元素可见性谓词。
若两者语义漂移,small-to-big(big.text 按 idx 范围重取原文)会绕过硬过滤捞到无权明文 —— INTEGRATION 铁律5。
fail-closed:空/缺字段一律按最严(= chunker RESTRICTED_ACL: unset=True/空 allow/restricted)。
deny 不生效(与 INTEGRATION 示例 filter 一致:要排除某组从 allow 移除)。"""
from __future__ import annotations

from .types import User


def acl_split(acl: dict) -> dict:
    """chunk.acl -> store payload 的 4 个可过滤 ACL 字段(fail-closed 默认)。"""
    acl = acl or {}
    tenant = acl.get("tenant") or ""
    return {
        # 空/缺 tenant 视同 unset(seal#1 fail-closed):空串 tenant 会与"空 tenant 用户"自匹配,
        # 让租户隔离形同虚设、只剩 public 把关 = fail-open。无法安全归属租户的文档一律默认拒绝;
        # 真要"全局公开"应走专门机制(如显式 tenant),绝不靠"无 tenant"。
        "acl_unset": bool(acl.get("unset")) or not acl or not tenant,
        "acl_tenant": tenant,
        "acl_allow": list(acl.get("allow") or []),
        "acl_visibility": acl.get("visibility") or "restricted",    # 无 -> restricted(非 public)
    }


def acl_admits(acl: dict, user: User) -> bool:
    """单个 acl 对 user 是否可见 —— store.acl_filter 的客户端等价(must: 非 unset ∧ 同租户 ∧ (allow∩principals ∨ public))。"""
    f = acl_split(acl)
    if f["acl_unset"] or not f["acl_tenant"] or not user.tenant:    # 空 tenant 双向拒绝(seal#1)
        return False
    if f["acl_tenant"] != user.tenant:
        return False
    return bool(set(f["acl_allow"]) & set(user.principals or [])) or f["acl_visibility"] == "public"  # principals=None 容错(B2 review)
