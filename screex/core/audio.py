from __future__ import annotations

from screex.core.index import NarrationSegment

_models: dict = {}


def is_available() -> bool:
    """True if the optional faster-whisper dependency is importable."""
    import importlib.util
    return importlib.util.find_spec("faster_whisper") is not None


def _get_model(model: str = "base"):
    """Lazily construct and cache a CPU int8 Whisper model. Callers gate with is_available()."""
    m = _models.get(model)
    if m is None:
        from faster_whisper import WhisperModel
        m = WhisperModel(model, device="cpu", compute_type="int8")
        _models[model] = m
    return m


def transcribe(path, model: str = "base", language=None) -> list:
    """Return timestamped NarrationSegments for the audio track of ``path``.
    Returns [] on any decode/transcription failure (e.g. no audio stream) so indexing
    never aborts because of audio."""
    engine = _get_model(model)
    try:
        segments, _ = engine.transcribe(str(path), language=language)
        out = []
        for s in segments:
            text = s.text.strip()
            if text:
                out.append(NarrationSegment(start=round(s.start, 3), end=round(s.end, 3), text=text))
        return out
    except Exception:
        return []
