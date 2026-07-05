"""多身份(DESIGN D10):API key → 身份。

三种模式(service.py 按配置自动选):
  keys   —— 团队:PHAROS_KEYS_FILE 指向 JSON,每请求 X-API-Key 解析成身份;未知/缺失一律 401。
  legacy —— 单人/一道门槛:只设 PHAROS_API_KEY,单密钥 + 启动绑定身份。
  open   —— 都不设:仅限回环地址的无鉴权模式。

fail-closed 铁律:keys 文件格式错 → 拒绝启动(不静默降级);非回环绑定必须 keys 模式
(service 启动守卫)。身份只回答"谁在问";"能看什么"由引擎 ACL 用该身份的 User 兑现。
"""
from __future__ import annotations

import json
import os
import secrets
from dataclasses import dataclass, field

KEY_MIN_LEN = 16


@dataclass(frozen=True)
class Identity:
    name: str
    tenant: str
    principals: list[str] = field(default_factory=list)
    admin: bool = False


def load_keys(path: str) -> dict[str, Identity]:
    """keys 文件 → {key: Identity}。任何格式问题都响亮失败(SystemExit),绝不静默降级成开放模式。"""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        raise SystemExit(f"PHAROS_KEYS_FILE 不存在:{path}(用 `pharos keys new <name>` 生成)")
    except json.JSONDecodeError as e:
        raise SystemExit(f"keys 文件不是合法 JSON:{path}({e})")
    entries = data.get("keys")
    if not isinstance(entries, list) or not entries:
        raise SystemExit(f"keys 文件需含非空 keys 数组:{path}")
    out: dict[str, Identity] = {}
    seen_names: set[str] = set()
    for i, k in enumerate(entries):
        key = str(k.get("key") or "").strip()
        name = str(k.get("name") or "").strip()
        tenant = str(k.get("tenant") or "").strip()
        if len(key) < KEY_MIN_LEN:
            raise SystemExit(f"keys[{i}].key 过短(<{KEY_MIN_LEN} 字符),拒绝启动")
        if not name or not tenant:
            raise SystemExit(f"keys[{i}] 缺 name/tenant(fail-closed:身份不完整不上线)")
        if "|" in name:
            # name 是会话去重登记键 f"{name}|{sid}" 的前缀(service._session_keys):含 '|' 会让
            # 命名空间碰撞(a + "b|c" 与 "a|b" + c 同键)。禁 '|' 保证前缀无歧义。
            raise SystemExit(f"keys[{i}].name 不得含 '|'(会话隔离前缀分隔符)")
        if name in seen_names:
            # name 唯一:它既是日志的身份标识,也是会话去重前缀;重名会让两个不同身份共享
            # 会话命名空间(去重串味)、且日志无法区分谁是谁。fail-closed 拒绝。
            raise SystemExit(f"keys[{i}].name '{name}' 重复(身份名必须唯一)")
        if key in out:
            raise SystemExit(f"keys[{i}] 与前面的条目 key 重复")
        seen_names.add(name)
        principals = [str(p).strip() for p in (k.get("principals") or []) if str(p).strip()]
        out[key] = Identity(name=name, tenant=tenant, principals=principals, admin=bool(k.get("admin")))
    return out


def new_key(name: str) -> str:
    """生成一个密钥(pk_<name>_<32hex>,secrets 级随机)。只在生成时打印一次,文件里是唯一持久化处。"""
    safe = "".join(c for c in name if c.isalnum() or c in "-_") or "user"
    return f"pk_{safe}_{secrets.token_hex(16)}"


def append_key(path: str, *, name: str, tenant: str, principals: list[str], admin: bool = False) -> str:
    """向 keys 文件追加一个新身份(文件不存在则创建),返回生成的 key。尽力 chmod 600。"""
    path = os.path.expanduser(path)
    data = {"keys": []}
    if os.path.exists(path):
        load_keys(path)                             # 复用 serve 侧校验:损坏/重名/含'|' 都响亮拒绝,不裸抛 traceback
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    if "|" in name:
        raise SystemExit("name 不得含 '|'(会话隔离前缀分隔符)")
    if any(e.get("name") == name for e in data["keys"]):
        raise SystemExit(f"name '{name}' 已存在(身份名必须唯一)")
    key = new_key(name)
    data["keys"].append({"key": key, "name": name, "tenant": tenant,
                         "principals": principals, "admin": admin})
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    try:
        os.chmod(path, 0o600)                       # 密钥文件收紧权限(Windows 上尽力而为)
    except OSError:
        pass
    return key


def is_loopback(host: str) -> bool:
    return (host or "").strip().lower() in ("127.0.0.1", "localhost", "::1")
