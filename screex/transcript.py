from __future__ import annotations


def _fmt_time(seconds) -> str:
    s = int(round(seconds))
    return f"{s // 60}:{s % 60:02d}"


def _narration_for(narration, t_start, t_end) -> str:
    """Join narration segments overlapping [t_start, t_end] in time order."""
    spoken = [n.text for n in narration if n.start < t_end and n.end > t_start]
    return " ".join(spoken)


def format_transcript(screen_index) -> str:
    """Render a ScreenIndex to a markdown step transcript."""
    lines = [f"# Transcript — {screen_index.video}  ({_fmt_time(screen_index.duration)})", ""]
    for n, st in enumerate(screen_index.states, start=1):
        lines.append(f"## {_fmt_time(st.t_start)}–{_fmt_time(st.t_end)}  ·  State {n}")
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
