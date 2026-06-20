"""Keyframe curation: rank settled UI states by how informative they are, and select a small,
temporally-spread budget of keyframes for an agent to escalate to.

The published GUI-World result is that *more* uniform frames make a model worse — fewer, better,
curated frames win. Screex already emits a keyframe per settled state; this module ranks them by a
transparent, dependency-free score (text-change magnitude + keyframe sharpness + a typed-event
bonus) and picks a budget that covers the recording rather than clustering on one busy moment.
"""
from __future__ import annotations

from typing import Any

# Transparent, tunable weights. Text change is the strongest "what happened" signal; sharpness
# prefers settled (non-blurry) frames; a typed event is disproportionately answer-bearing.
W_TEXT = 0.6
W_SHARP = 0.3
W_EVENT = 0.1

# How strongly to reward a candidate for being far from already-picked states (temporal coverage).
SPREAD_BONUS = 0.15


def _normalize(values: list[float]) -> list[float]:
    """Min-max normalize to 0..1; all-equal (or empty) inputs map to zeros."""
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi <= lo:
        return [0.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def score_states(states, sharpness: list[float] | None = None) -> list[float]:
    """Set ``state.salience`` for each state and return the scores.

    ``sharpness`` is a parallel list of keyframe Laplacian-variance values (higher = crisper); pass
    ``None``/zeros when unavailable (e.g. motion-only ``--fast`` runs), in which case salience comes
    from text change and events alone."""
    text_change = [
        len(getattr(s, "text_added", []) or []) + len(getattr(s, "text_removed", []) or [])
        for s in states
    ]
    nt = _normalize([float(x) for x in text_change])
    ns = _normalize([float(x) for x in sharpness]) if sharpness else [0.0] * len(states)

    scores = []
    for i, s in enumerate(states):
        event_bonus = 1.0 if getattr(s, "event", None) else 0.0
        score = round(W_TEXT * nt[i] + W_SHARP * ns[i] + W_EVENT * event_bonus, 4)
        s.salience = score
        scores.append(score)
    return scores


def select_curated(states, budget: int) -> list[dict[str, Any]]:
    """Greedily pick up to ``budget`` states by salience, rewarding temporal spread so the picks
    cover the recording instead of clustering. Returns ``[{idx, t_start, keyframe, salience}]``
    ordered by ``t_start``. ``budget <= 0`` → ``[]``; ``budget >= len(states)`` → all states."""
    n = len(states)
    if budget <= 0 or n == 0:
        return []
    if budget >= n:
        chosen = list(range(n))
    else:
        chosen = []
        while len(chosen) < budget:
            def adjusted(i: int) -> float:
                base = float(getattr(states[i], "salience", 0.0))
                if not chosen:
                    return base
                d = min(abs(i - c) for c in chosen)
                return base + SPREAD_BONUS * (1.0 - 1.0 / (1 + d))
            # Highest adjusted score wins; tie-break on earliest index for determinism.
            pick = max((i for i in range(n) if i not in chosen),
                       key=lambda i: (adjusted(i), -i))
            chosen.append(pick)

    return [
        {
            "idx": states[i].idx,
            "t_start": states[i].t_start,
            "keyframe": states[i].keyframe,
            "salience": getattr(states[i], "salience", 0.0),
        }
        for i in sorted(chosen)
    ]
