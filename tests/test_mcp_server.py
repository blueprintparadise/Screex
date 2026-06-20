import importlib.util

import pytest

from screex import mcp_server
from screex.core.index import ScreenIndex, ScreenState

_HAS_MCP = importlib.util.find_spec("mcp") is not None


def _save_index(tmp_path):
    si = ScreenIndex(
        video="cast.mp4", duration=6.0, sampled_fps=2.0,
        states=[
            ScreenState(0, 0.0, 3.0, "t0.png", "k0.png", ocr_text=["Open Settings"]),
            ScreenState(1, 3.0, 6.0, "t1.png", "k1.png", ocr_text=["Error: bad key"],
                        event={"type": "error", "label": "bad key"}),
        ],
    )
    path = tmp_path / "index.json"
    si.save(path)
    return str(path)


def test_info_tool(tmp_path):
    out = mcp_server.info_tool(_save_index(tmp_path))
    assert out["states"] == 2
    assert out["events"] == {"error": 1}


def test_search_tool(tmp_path):
    hits = mcp_server.search_tool(_save_index(tmp_path), pattern="error")
    assert [h["idx"] for h in hits] == [1]


def test_transcript_tool_srt_and_bad_format(tmp_path):
    idx = _save_index(tmp_path)
    assert "-->" in mcp_server.transcript_tool(idx, fmt="srt")
    with pytest.raises(ValueError, match="unknown format"):
        mcp_server.transcript_tool(idx, fmt="nope")


def test_build_server_requires_extra_or_builds():
    if _HAS_MCP:
        server = mcp_server.build_server()
        assert server is not None
    else:
        with pytest.raises(RuntimeError, match="pip install 'screex\\[mcp\\]'"):
            mcp_server.build_server()
