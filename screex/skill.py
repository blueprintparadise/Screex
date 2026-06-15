from __future__ import annotations

from pathlib import Path


def skill_text() -> str:
    """Return the bundled SKILL.md content (shipped as package data)."""
    from importlib.resources import files

    return (files("screex") / "SKILL.md").read_text(encoding="utf-8")


def default_skill_dir() -> Path:
    """Default Claude Code skill directory for Screex (~/.claude/skills/screex)."""
    return Path.home() / ".claude" / "skills" / "screex"


def install_skill(dest_dir=None) -> Path:
    """Write the bundled SKILL.md into a skills directory; return the written file path."""
    target_dir = Path(dest_dir) if dest_dir else default_skill_dir()
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / "SKILL.md"
    target.write_text(skill_text(), encoding="utf-8")
    return target


def skill_status(dest_dir=None) -> str:
    """Compare the installed SKILL.md to the bundled one (by content).

    Returns "missing" (not installed), "current" (matches the package's bundled skill),
    or "stale" (installed but differs — re-run ``screex skill --install``)."""
    target_dir = Path(dest_dir) if dest_dir else default_skill_dir()
    target = target_dir / "SKILL.md"
    if not target.exists():
        return "missing"
    return "current" if target.read_text(encoding="utf-8") == skill_text() else "stale"
