"""Secret / PII redaction for on-screen text and narration.

Screex indexes whatever is on screen, which for bug repros and demos often includes
passwords, API keys, tokens, and personal data. ``redact_line`` masks the most common
high-risk patterns (and high-entropy token-like strings) so the index and any blurred
keyframe regions are safe to share. This is best-effort, not a compliance guarantee.
"""
from __future__ import annotations

import math
import re

# (kind, compiled pattern). Order matters: more specific patterns first.
_PATTERNS = [
    ("private_key", re.compile(r"-----BEGIN[A-Z ]*PRIVATE KEY-----")),
    ("jwt", re.compile(r"eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+")),
    ("aws_key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("api_key", re.compile(
        r"\b(?:sk|pk|rk|ghp|gho|ghu|ghs|xox[baprs]|AIza)[-_][A-Za-z0-9-_]{8,}\b")),
    ("email", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    ("ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("credit_card", re.compile(r"\b(?:\d[ -]?){13,16}\b")),
]

_TOKEN = re.compile(r"\S+")


def _shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    counts: dict[str, int] = {}
    for ch in s:
        counts[ch] = counts.get(ch, 0) + 1
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def _luhn_ok(digits: str) -> bool:
    nums = [int(d) for d in digits if d.isdigit()]
    if not 13 <= len(nums) <= 19:
        return False
    total = 0
    for i, d in enumerate(reversed(nums)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _looks_like_secret_token(tok: str) -> bool:
    """A standalone token long and random-looking enough to be a credential."""
    core = tok.strip(".,;:!?()[]{}'\"")
    if len(core) < 20:
        return False
    has_alpha = any(c.isalpha() for c in core)
    has_digit = any(c.isdigit() for c in core)
    if not (has_alpha and has_digit):
        return False
    return _shannon_entropy(core) >= 3.5


def find_secrets(text: str) -> list[tuple[int, int, str]]:
    """Return non-overlapping (start, end, kind) spans of likely secrets in ``text``."""
    spans: list[tuple[int, int, str]] = []
    for kind, pat in _PATTERNS:
        for m in pat.finditer(text):
            if kind == "credit_card" and not _luhn_ok(m.group()):
                continue
            spans.append((m.start(), m.end(), kind))
    for m in _TOKEN.finditer(text):
        if _looks_like_secret_token(m.group()):
            spans.append((m.start(), m.end(), "secret"))

    # Resolve overlaps: keep the earliest, longest span.
    spans.sort(key=lambda s: (s[0], -(s[1] - s[0])))
    chosen: list[tuple[int, int, str]] = []
    last_end = -1
    for start, end, kind in spans:
        if start >= last_end:
            chosen.append((start, end, kind))
            last_end = end
    return chosen


def redact_line(text: str, mask: str = "[REDACTED:{kind}]") -> tuple[str, list[str]]:
    """Return ``(redacted_text, kinds)``. ``kinds`` lists the secret types found (empty if
    the line is clean). Each match span is replaced by ``mask`` formatted with its kind."""
    spans = find_secrets(text)
    if not spans:
        return text, []
    kinds = [kind for _, _, kind in spans]
    out = []
    cursor = 0
    for start, end, kind in spans:
        out.append(text[cursor:start])
        out.append(mask.format(kind=kind))
        cursor = end
    out.append(text[cursor:])
    return "".join(out), kinds


def has_secret(text: str) -> bool:
    return bool(find_secrets(text))
