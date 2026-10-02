"""Continuation-boundary trimming for base-model output.

A base model does not stop; it rolls on into the next document. If the judge sees that
drift it scores noise. Trim to the first turn's worth of text before judging, and
record how much was removed so the trimming itself can be audited.
"""
from __future__ import annotations

import re

# Markers that a new turn or a new document has begun.
_BOUNDARIES = [
    r"\nUser:", r"\nHuman:", r"\nAssistant:", r"\nQ:", r"\nQuestion:",
    r"\n#{1,6}\s", r"\n-{3,}\n", r"\n={3,}\n", r"\n\*{3,}\n",
    r"<\|im_start\|>", r"<\|im_end\|>", r"<\|endoftext\|>",
]
_BOUNDARY_RE = re.compile("|".join(_BOUNDARIES))


def _collapse_repetition(text: str, n: int = 12, limit: int = 3) -> str:
    """Cut the text where an n-word window starts repeating beyond `limit` times."""
    words = text.split()
    if len(words) < n * 2:
        return text
    seen = {}
    for i in range(len(words) - n + 1):
        key = " ".join(words[i : i + n])
        seen[key] = seen.get(key, 0) + 1
        if seen[key] > limit:
            return " ".join(words[: i + n])
    return text


def trim(text: str) -> dict:
    """Return {text, trimmed_chars, reason}."""
    original = text
    m = _BOUNDARY_RE.search(text)
    reason = None
    if m and m.start() > 0:
        text = text[: m.start()]
        reason = "boundary"
    collapsed = _collapse_repetition(text)
    if len(collapsed) < len(text):
        text = collapsed
        reason = "repetition" if reason is None else reason + "+repetition"
    text = text.strip()
    return {
        "text": text,
        "trimmed_chars": len(original) - len(text),
        "reason": reason or "none",
        "degenerate": len(text) < 8,
    }
