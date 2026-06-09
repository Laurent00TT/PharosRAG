"""API key generation + hashing.

Key shape: ``kb_{username}_{43-char-random}``
  - prefix 'kb_' identifies the issuer (this KB)
  - username makes the key self-describing in logs (without exposing secret)
  - 32-byte random via secrets.token_urlsafe → ~43 URL-safe chars,
    256 bits of entropy (P2-1)

DB stores sha256(plaintext) — never plaintext. ``key_prefix`` is the
first 8 chars of the random suffix, used by doctor view to identify
"which key alice is using" without leaking the secret.
"""
from __future__ import annotations

import hashlib
import secrets

_KEY_PREFIX_LEN = 8


def generate_key(username: str) -> tuple[str, str, str]:
    """Return (plaintext, key_prefix, key_hash).

    Caller responsibilities:
      - Show ``plaintext`` to the admin ONCE (e.g. CLI print), never
        store or log it after that point.
      - Persist ``key_prefix`` and ``key_hash`` to the users table.
      - On auth, hash the incoming token via ``hash_token`` and compare
        against the stored ``key_hash``.
    """
    random_suffix = secrets.token_urlsafe(32)
    plaintext = f"kb_{username}_{random_suffix}"
    key_prefix = f"kb_{username}_{random_suffix[:_KEY_PREFIX_LEN]}"
    key_hash = hash_token(plaintext)
    return plaintext, key_prefix, key_hash


def hash_token(token: str) -> str:
    """SHA-256 hex digest of an API token. Constant-time comparison is the
    caller's responsibility (hmac.compare_digest on the hash, not raw token).
    """
    return hashlib.sha256(token.encode("utf-8")).hexdigest()
