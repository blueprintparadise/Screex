from __future__ import annotations

import json


def _fmt_time(seconds) -> str:
    s = int(round(seconds))
    return f"{s // 60}:{s % 60:02d}"


def _ts(seconds: float, sep: str) -> str:
    """Format seconds as HH:MM:SS<sep>mmm for SRT (sep=',') or WebVTT (sep='.')."""
    seconds = max(0.0, float(seconds))
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def _narration_for(narration, t_start, t_end) -> str:
    """Join narration segments overlapping [t_start, t_end] in time order."""
    spoken = [n.text for n in narration if n.start < t_end and n.end > t_start]
    return " ".join(spoken)


def _format_event(ev) -> str:
    t = ev.get("type")
    label, value, field = ev.get("label"), ev.get("value"), ev.get("field")
    if t == "click":
        return f"👆 Clicked “{label}”" if label else "👆 Clicked"
    if t == "type":
        return f"⌨ Typed “{value}” into {field}" if field else f"⌨ Typed “{value}”"
    if t == "navigate":
        return f"➡ Navigated to {value}" if value else "➡ Navigated"
    if t == "open_dialog":
        return f"🪟 Opened dialog: {label}" if label else "🪟 Opened dialog"
    if t == "error":
        return f"⚠ Error: {label}" if label else "⚠ Error"
    if t == "scroll":
        return "↕ Scrolled"
    if t == "edit":
        return "✏ Edited"
    return ""


def format_transcript(screen_index) -> str:
    """Render a ScreenIndex to a markdown step transcript."""
    lines = [f"# Transcript — {screen_index.video}  ({_fmt_time(screen_index.duration)})", ""]
    for n, st in enumerate(screen_index.states, start=1):
        lines.append(f"## {_fmt_time(st.t_start)}–{_fmt_time(st.t_end)}  ·  State {n}")
        ev = getattr(st, "event", None) or {}
        if ev:
            event_line = _format_event(ev)
            if event_line:
                lines.append(event_line)
        if st.ocr_text:
            lines.append(" · ".join(st.ocr_text))
        if st.text_added:
            lines.append(f"**Appeared:** {', '.join(st.text_added)}")
        if st.text_removed:
            lines.append(f"**Gone:** {', '.join(st.text_removed)}")
        for it in getattr(st, "interactions", None) or []:
            label = it.get("label")
            where = f" near “{label}”" if label else ""
            lines.append(f"👆 interacted{where} (≈{int(it['x'])},{int(it['y'])})")
        spoken = _narration_for(getattr(screen_index, "narration", None) or [], st.t_start, st.t_end)
        if spoken:
            lines.append(f"🗣 said: {spoken}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _state_caption(st) -> str:
    """A one-line caption for a state, used by the SRT/WebVTT cues."""
    ev = _format_event(getattr(st, "event", None) or {})
    body = " · ".join(st.ocr_text) if st.ocr_text else ""
    return " — ".join(p for p in (ev, body) if p) or f"State {st.idx}"


def format_json(screen_index) -> str:
    """Render the index as the compact, LLM-oriented JSON view (stable, machine-readable)."""
    return json.dumps(screen_index.compact_dict(), indent=2, ensure_ascii=False)


def format_srt(screen_index) -> str:
    """Render states as SubRip (.srt) cues — one cue per UI state, timed to [t_start, t_end]."""
    blocks = []
    for n, st in enumerate(screen_index.states, start=1):
        blocks.append(f"{n}\n{_ts(st.t_start, ',')} --> {_ts(st.t_end, ',')}\n"
                      f"{_state_caption(st)}")
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def format_webvtt(screen_index) -> str:
    """Render states as WebVTT (.vtt) cues — one cue per UI state."""
    out = ["WEBVTT", ""]
    for st in screen_index.states:
        out.append(f"{_ts(st.t_start, '.')} --> {_ts(st.t_end, '.')}")
        out.append(_state_caption(st))
        out.append("")
    return "\n".join(out).rstrip() + "\n"


FORMATTERS = {
    "md": format_transcript,
    "json": format_json,
    "srt": format_srt,
    "vtt": format_webvtt,
}
