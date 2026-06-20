"""Read-only queries over a built :class:`~screex.core.index.ScreenIndex`.

Shared by the ``screex info`` / ``screex search`` CLI subcommands and the optional MCP server, so
the same logic backs every entry point.
"""
from __future__ import annotations

from typing import Any


def summarize_index(si) -> dict[str, Any]:
    """Return a compact summary of an index: counts, duration, an event-type histogram, the number
    of UI-chrome lines shared by every state, whether curated keyframes are available, etc."""
    states = si.states
    events: dict[str, int] = {}
    for s in states:
        ev = (s.event or {}).get("type") if getattr(s, "event", None) else None
        if ev:
            events[ev] = events.get(ev, 0) + 1

    persistent = 0
    if len(states) > 1:
        line_sets = [set(s.ocr_text) for s in states]
        persistent = len(set.intersection(*line_sets)) if line_sets else 0

    return {
        "video": si.video,
        "duration_s": round(si.duration, 2),
        "sampled_fps": si.sampled_fps,
        "states": len(states),
        "ocr_lines": sum(len(s.ocr_text) for s in states),
        "events": events,
        "narration_segments": len(si.narration),
        "warnings": len(si.warnings),
        "persistent_ui_lines": persistent,
        "has_curated_keyframes": any(getattr(s, "salience", 0.0) for s in states),
    }


def search_index(si, pattern: str | None = None, since: float | None = None,
                 until: float | None = None, event_type: str | None = None) -> list[dict[str, Any]]:
    """Return states matching the given filters (all optional, AND-combined):

    - ``pattern``: case-insensitive substring against each state's ``ocr_text`` lines;
    - ``since`` / ``until``: keep states whose [t_start, t_end] overlaps the window;
    - ``event_type``: keep states whose typed ``event`` has this ``type``.

    Each hit is ``{idx, t_start, t_end, event, matches}`` where ``matches`` are the ocr lines that
    contained ``pattern`` (empty when no text pattern was given)."""
    needle = pattern.lower() if pattern else None
    hits = []
    for s in si.states:
        if since is not None and s.t_end < since:
            continue
        if until is not None and s.t_start > until:
            continue
        ev_type = (s.event or {}).get("type") if getattr(s, "event", None) else None
        if event_type is not None and ev_type != event_type:
            continue
        matches = [ln for ln in s.ocr_text if needle in ln.lower()] if needle else []
        if needle is not None and not matches:
            continue
        hits.append({
            "idx": s.idx,
            "t_start": s.t_start,
            "t_end": s.t_end,
            "event": ev_type,
            "matches": matches,
        })
    return hits
