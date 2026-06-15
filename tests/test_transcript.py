from screex.core.index import ScreenIndex, ScreenState
from screex.transcript import format_transcript


def test_format_transcript():
    si = ScreenIndex(video="cast.mp4", duration=65.0, sampled_fps=2.0, states=[
        ScreenState(0, 0.0, 3.0, "frames/0_thumb.png", "frames/0.png",
                    ocr_text=["Login"], text_added=["Login"], text_removed=[]),
        ScreenState(1, 3.0, 65.0, "frames/1_thumb.png", "frames/1.png",
                    ocr_text=["Dashboard"], text_added=["Dashboard"], text_removed=["Login"]),
    ])
    md = format_transcript(si)
    assert "# Transcript — cast.mp4  (1:05)" in md
    assert "## 0:00–0:03  ·  State 1" in md
    assert "**Appeared:** Login" in md
    assert "## 0:03–1:05  ·  State 2" in md
    assert "**Gone:** Login" in md
    # the first state has no removed text -> the only "Gone" line belongs to State 2
    assert md.index("**Gone:** Login") > md.index("## 0:03–1:05")


def test_format_transcript_interleaves_narration():
    from screex.core.index import NarrationSegment, ScreenIndex, ScreenState
    si = ScreenIndex(
        video="c.mp4", duration=6.0, sampled_fps=2.0,
        states=[
            ScreenState(0, 0.0, 3.0, "t.png", "k.png", ocr_text=["Login"], text_added=["Login"], text_removed=[]),
            ScreenState(1, 3.0, 6.0, "t.png", "k.png", ocr_text=["Home"], text_added=["Home"], text_removed=[]),
        ],
        narration=[NarrationSegment(0.5, 2.0, "first click save"),
                   NarrationSegment(4.0, 5.0, "now we are home")],
    )
    md = format_transcript(si)
    assert "🗣 said: first click save" in md
    assert "🗣 said: now we are home" in md
    assert md.index("first click save") < md.index("now we are home")
