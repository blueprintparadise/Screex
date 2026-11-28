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


# --- Optional query-conditioned curation -------------------------------------------------
# The default `select_curated` is query-agnostic (best when no question is known, e.g. building a
# reusable index). When a specific question IS known, the frame-selection literature (BOLT/Q-Frame/
# AKS, 2025) shows scoring keyframes by relevance to the question beats change-magnitude alone.
# This is an additive alternative — it does not change `score_states`/`select_curated`. The CLIP
# embedder requires the optional ``[keyframes]`` extra; callers fall back to `select_curated`.


def _cosine(a, b) -> float:
    import math
    num = sum(x * y for x, y in zip(a, b))
    da = math.sqrt(sum(x * x for x in a))
    db = math.sqrt(sum(y * y for y in b))
    if da == 0 or db == 0:
        return 0.0
    return num / (da * db)


def select_curated_for_query(states, budget, base_dir, question, embedder,
                             diversity: float = 0.5) -> list[dict[str, Any]]:
    """Query-conditioned curation. Rank each state's keyframe by cosine relevance of its
    ``embedder.embed_image`` to ``embedder.embed_text(question)``, then greedily pick ``budget``
    maximizing the relevance/novelty blend
    ``(1 - diversity) * relevance + diversity * (1 - max_cosine_to_already_picked)``
    (``diversity`` in [0, 1]). Returns the same shape as :func:`select_curated`, ordered by
    ``t_start``. ``base_dir`` is the index dir so ``base_dir / state.keyframe`` resolves the image."""
    from pathlib import Path

    n = len(states)
    if budget <= 0 or n == 0:
        return []
    if budget >= n:
        chosen = list(range(n))
    else:
        base = Path(base_dir)
        paths = [str(base / s.keyframe) for s in states]
        qvec = embedder.embed_text(question)
        vecs = [embedder.embed_image(p) for p in paths]
        relevance = [_cosine(qvec, v) for v in vecs]
        chosen, remaining = [], set(range(n))
        while remaining and len(chosen) < budget:
            best_i, best_val = None, None
            for i in remaining:
                max_sim = max((_cosine(vecs[i], vecs[j]) for j in chosen), default=0.0)
                val = (1 - diversity) * relevance[i] + diversity * (1 - max_sim)
                if best_val is None or val > best_val:
                    best_val, best_i = val, i
            chosen.append(best_i)
            remaining.discard(best_i)

    return [
        {
            "idx": states[i].idx,
            "t_start": states[i].t_start,
            "keyframe": states[i].keyframe,
            "salience": getattr(states[i], "salience", 0.0),
        }
        for i in sorted(chosen)
    ]


class ClipEmbedder:
    """Image+text embedder for query-conditioned curation. Optional — requires the
    ``[keyframes]`` extra (``pip install 'screex[keyframes]'``)."""

    def __init__(self, model_name: str = "clip-ViT-B-32"):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_name)

    def embed_text(self, text: str):
        return self._model.encode(text).tolist()

    def embed_image(self, path: str):
        from PIL import Image
        return self._model.encode(Image.open(path)).tolist()
