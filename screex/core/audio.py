from __future__ import annotations

import os

from screex.core.index import NarrationSegment

_models: dict = {}


def _ensure_openmp_compat() -> None:
    """Let PyTorch (faster-whisper) and ONNX Runtime (RapidOCR) coexist in one process.

    Both ship their own OpenMP runtime. On Windows, loading two OpenMP runtimes into the same
    process aborts it with a hard native fault — which is exactly what happens when ``screex
    index --audio`` runs OCR (onnxruntime) and then whisper (torch). Setting
    ``KMP_DUPLICATE_LIB_OK`` before faster-whisper (and thus torch) is imported avoids the abort.
    ``setdefault`` is used so an explicit user-provided value always wins."""
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def is_available() -> bool:
    """True if the optional faster-whisper dependency is importable."""
    import importlib.util
    return importlib.util.find_spec("faster_whisper") is not None


def _get_model(model: str = "base"):
    """Lazily construct and cache a CPU int8 Whisper model. Callers gate with is_available()."""
    m = _models.get(model)
    if m is None:
        _ensure_openmp_compat()
        from faster_whisper import WhisperModel
        m = WhisperModel(model, device="cpu", compute_type="int8")
        _models[model] = m
    return m


def transcribe(path, model: str = "base", language=None) -> list[NarrationSegment]:
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
