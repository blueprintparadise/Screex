"""Optional Model Context Protocol (MCP) server exposing Screex to any MCP client.

Install the extra and run it over stdio:

    pip install 'screex[mcp]'
    screex mcp

The tool *logic* below reuses the same functions as the CLI (``cli.index``, ``query.*``,
``transcript.*``), so it is importable and testable without the ``mcp`` package; only
``build_server``/``serve`` need it (imported lazily)."""
from __future__ import annotations

from typing import Any


def _require_fastmcp():
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as e:  # pragma: no cover - exercised only without the extra
        raise RuntimeError(
            "the MCP server needs the 'mcp' package: pip install 'screex[mcp]'"
        ) from e
    return FastMCP


def build_index_tool(recording: str, fps: float = 2.0, fast: bool = False,
                     events: bool = False, keyframe_budget: int | None = None) -> dict[str, Any]:
    """Build a Screex index for a screen recording and return its path plus a summary."""
    from screex.cli import index
    from screex.core.index import ScreenIndex
    from screex.query import summarize_index
    path = index(recording, fps=fps, fast=fast, events=events,
                 keyframe_budget=keyframe_budget, quiet=True, audio=False)
    return {"index_path": str(path), "summary": summarize_index(ScreenIndex.load(path))}


def info_tool(index_path: str) -> dict[str, Any]:
    """Summarize a built index.json (states, duration, event histogram, warnings, ...)."""
    from screex.core.index import ScreenIndex
    from screex.query import summarize_index
    return summarize_index(ScreenIndex.load(index_path))


def search_tool(index_path: str, pattern: str | None = None, since: float | None = None,
                until: float | None = None, event_type: str | None = None) -> list[dict[str, Any]]:
    """Search states in a built index.json by on-screen text, time window, and/or event type."""
    from screex.core.index import ScreenIndex
    from screex.query import search_index
    return search_index(ScreenIndex.load(index_path), pattern=pattern, since=since,
                        until=until, event_type=event_type)


def transcript_tool(index_path: str, fmt: str = "md") -> str:
    """Render a built index.json as a transcript: fmt = md | json | srt | vtt."""
    from screex.core.index import ScreenIndex
    from screex.transcript import FORMATTERS
    if fmt not in FORMATTERS:
        raise ValueError(f"unknown format {fmt!r} (use one of: {', '.join(FORMATTERS)})")
    return FORMATTERS[fmt](ScreenIndex.load(index_path))


_TOOLS = (build_index_tool, info_tool, search_tool, transcript_tool)


def build_server():
    """Construct a FastMCP server with the Screex tools registered. Requires ``screex[mcp]``."""
    FastMCP = _require_fastmcp()
    server = FastMCP("screex")
    for tool in _TOOLS:
        server.tool()(tool)
    return server


def serve() -> None:
    """Run the MCP server over stdio (blocking). Requires ``screex[mcp]``."""
    build_server().run()
