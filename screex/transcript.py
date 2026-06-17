from __future__ import annotations


def _fmt_time(seconds) -> str:
    s = int(round(seconds))
    return f"{s // 60}:{s % 60:02d}"


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
