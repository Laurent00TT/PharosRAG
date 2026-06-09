"""Truncate long fields for trace logging without losing identifiability."""

_PREVIEW_MAX = 200


def preview(s: str | None, n: int = _PREVIEW_MAX) -> str:
    """Truncate to n chars, append '...[+M chars]' marker if truncated."""
    if not s:
        return ""
    if len(s) <= n:
        return s
    return s[:n] + f"...[+{len(s) - n} chars]"
