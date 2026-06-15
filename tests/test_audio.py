from screex.core import audio
from screex.core.index import NarrationSegment


def test_is_available_returns_bool():
    assert isinstance(audio.is_available(), bool)


def test_transcribe_parses_and_strips_segments(monkeypatch):
    class Seg:
        def __init__(self, s, e, t):
            self.start, self.end, self.text = s, e, t

    class Fake:
        def transcribe(self, path, language=None):
            return iter([Seg(0.0, 1.0, " Hello "), Seg(1.0, 2.0, "   "), Seg(2.0, 3.0, "World")]), None

    monkeypatch.setattr(audio, "_get_model", lambda *a, **k: Fake())
    out = audio.transcribe("x.mp4")
    assert out == [NarrationSegment(0.0, 1.0, "Hello"), NarrationSegment(2.0, 3.0, "World")]


def test_transcribe_graceful_on_failure(monkeypatch):
    class Boom:
        def transcribe(self, *a, **k):
            raise RuntimeError("no audio stream")

    monkeypatch.setattr(audio, "_get_model", lambda *a, **k: Boom())
    assert audio.transcribe("x.mp4") == []


def test_transcribe_real_clip():
    import os

    import pytest
    pytest.importorskip("faster_whisper")  # skipped unless screex[audio] is installed
    clip = "C:/Users/RushiHiray/Pictures/whisper_crop.mp4"
    if not os.path.exists(clip):
        pytest.skip("sample clip not present")
    segs = audio.transcribe(clip, model="tiny")
    assert segs and any(s.text for s in segs)
