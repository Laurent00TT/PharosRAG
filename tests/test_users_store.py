"""T1a tests — UsersStore + key generation.

Co-located in one file because UsersStore CRUD and key_gen are tightly
coupled (UsersStore persists key_hash and key_prefix; key_gen produces them).
Splitting would force every test to import both modules anyway."""
from __future__ import annotations

import hashlib

from kb.auth.key_gen import generate_key, hash_token


def test_generate_key_returns_three_parts():
    plaintext, key_prefix, key_hash = generate_key("alice")
    assert plaintext.startswith("kb_alice_")
    assert key_prefix.startswith("kb_alice_")
    assert len(key_hash) == 64  # sha256 hex digest


def test_generate_key_has_high_entropy():
    """secrets.token_urlsafe(32) → 43 chars ≈ 256 bits entropy.
    Anti-regression for the earlier 'kb_{user}_{16字符随机}' spec which
    only had ~96 bits."""
    plaintext, _, _ = generate_key("alice")
    # Strip prefix 'kb_alice_' to count just the random part
    random_part = plaintext[len("kb_alice_"):]
    assert len(random_part) >= 40, f"token_urlsafe(32) should produce ≥40 chars, got {len(random_part)}"


def test_generate_key_prefix_is_8_random_chars():
    """key_prefix is doctor-display identifier, 8 chars after 'kb_{user}_'."""
    plaintext, key_prefix, _ = generate_key("alice")
    random_part_full = plaintext[len("kb_alice_"):]
    prefix_random = key_prefix[len("kb_alice_"):]
    assert len(prefix_random) == 8
    assert prefix_random == random_part_full[:8]


def test_generate_key_two_calls_produce_different_secrets():
    """Sanity check — secrets.token_urlsafe should be unique per call."""
    a, _, _ = generate_key("alice")
    b, _, _ = generate_key("alice")
    assert a != b


def test_hash_token_matches_sha256():
    """hash_token wraps sha256 so callers don't have to remember .encode()."""
    token = "kb_alice_xyz"
    assert hash_token(token) == hashlib.sha256(token.encode("utf-8")).hexdigest()


def test_hash_token_is_deterministic():
    """Two calls with same input must hash to same digest (so DB lookups work)."""
    assert hash_token("foo") == hash_token("foo")


import pytest

from kb.auth.users import User, UsersStore


@pytest.fixture
async def store(tmp_path):
    db_url = f"sqlite+aiosqlite:///{tmp_path}/users.db"
    s = UsersStore(db_url=db_url)
    await s.init()
    yield s
    await s.aclose()


async def test_create_user_returns_plaintext_once(store):
    """Plaintext is returned to caller and NEVER persisted as-is."""
    plaintext, user = await store.create_user(username="alice", role="member")
    assert plaintext.startswith("kb_alice_")
    assert user.username == "alice"
    assert user.role == "member"
    assert user.disabled_at is None
    # Stored only hash + prefix, never plaintext
    assert user.key_hash != plaintext
    assert len(user.key_hash) == 64


async def test_get_by_key_hash_returns_user(store):
    plaintext, created = await store.create_user(username="bob", role="admin")
    from kb.auth.key_gen import hash_token
    found = await store.get_by_key_hash(hash_token(plaintext))
    assert found is not None
    assert found.user_id == created.user_id
    assert found.role == "admin"


async def test_get_by_key_hash_returns_none_for_unknown(store):
    found = await store.get_by_key_hash("0" * 64)
    assert found is None


async def test_get_by_username_returns_user(store):
    _, created = await store.create_user(username="carol", role="member")
    found = await store.get_by_username("carol")
    assert found is not None
    assert found.user_id == created.user_id


async def test_create_user_rejects_duplicate_username(store):
    await store.create_user(username="alice", role="member")
    with pytest.raises(ValueError, match="exists"):
        await store.create_user(username="alice", role="admin")


async def test_disable_user_sets_disabled_at(store):
    _, alice = await store.create_user(username="alice", role="member")
    await store.disable_user(alice.user_id)
    refreshed = await store.get_by_user_id(alice.user_id)
    assert refreshed.disabled_at is not None


async def test_set_role_changes_role(store):
    _, alice = await store.create_user(username="alice", role="member")
    await store.set_role(alice.user_id, "admin")
    refreshed = await store.get_by_user_id(alice.user_id)
    assert refreshed.role == "admin"


async def test_reset_key_returns_new_plaintext_and_invalidates_old(store):
    """reset_key rotates: old key_hash no longer matches, new plaintext works."""
    from kb.auth.key_gen import hash_token
    old_plaintext, alice = await store.create_user(username="alice", role="member")
    old_hash = hash_token(old_plaintext)

    new_plaintext = await store.reset_key(alice.user_id)
    assert new_plaintext != old_plaintext
    assert new_plaintext.startswith("kb_alice_")

    # Old hash no longer maps to alice
    assert await store.get_by_key_hash(old_hash) is None
    # New hash does
    found = await store.get_by_key_hash(hash_token(new_plaintext))
    assert found is not None
    assert found.user_id == alice.user_id


async def test_list_users_returns_all_with_or_without_disabled(store):
    _, alice = await store.create_user(username="alice", role="member")
    _, bob = await store.create_user(username="bob", role="admin")
    await store.disable_user(bob.user_id)

    active_only = await store.list_users(include_disabled=False)
    assert {u.username for u in active_only} == {"alice"}

    all_users = await store.list_users(include_disabled=True)
    assert {u.username for u in all_users} == {"alice", "bob"}


async def test_init_is_idempotent(tmp_path):
    """Running init() twice should not raise — server may restart."""
    db_url = f"sqlite+aiosqlite:///{tmp_path}/users.db"
    s = UsersStore(db_url=db_url)
    await s.init()
    await s.init()  # must not raise
    await s.aclose()


async def test_create_user_rejects_invalid_role(store):
    with pytest.raises(ValueError, match="role must be"):
        await store.create_user(username="alice", role="superuser")


async def test_set_role_rejects_invalid_role(store):
    _, user = await store.create_user(username="alice", role="member")
    with pytest.raises(ValueError, match="role must be"):
        await store.set_role(user.user_id, "superuser")


async def test_set_role_raises_for_unknown_user(store):
    with pytest.raises(ValueError, match="not found"):
        await store.set_role("nonexistent_id", "member")


async def test_reset_key_raises_for_unknown_user(store):
    with pytest.raises(ValueError, match="not found"):
        await store.reset_key("nonexistent_id")
