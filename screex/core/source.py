from __future__ import annotations

import os


def _open(path):
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        if not os.path.exists(path):
            raise FileNotFoundError(f"no such video file: {path}")
        raise ValueError(
            f"cannot decode video (corrupt or unsupported codec/container?): {path}"
        )
    return cap


def video_info(path: str) -> dict:
    import cv2

    cap = _open(path)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        cap.release()
    duration = count / fps if fps else 0.0
    return {"fps": fps, "count": count, "width": w, "height": h, "duration": duration}


def iter_frames(path: str, sample_fps: float, max_frames: int | None = None,
                diagnostics: list[str] | None = None):
    """Yield (out_idx, t_seconds, bgr) for sampled frames.

    The timestamp prefers the container's real position (CAP_PROP_POS_MSEC), which is
    correct for variable-frame-rate recordings, and falls back to raw_idx/native only
    when the container does not report a position. ``max_frames`` caps the number of
    sampled frames returned (None = no cap).

    If a frame fails to decode mid-stream (a truncated or corrupt recording), iteration
    stops cleanly with whatever was decoded so far; when ``diagnostics`` is provided, a
    recoverable warning is appended to it rather than raising."""
    cap = _open(path)
    import cv2

    native = cap.get(cv2.CAP_PROP_FPS) or sample_fps or 1.0
    step = max(1, round(native / sample_fps)) if sample_fps else 1
    raw_idx = 0
    out_idx = 0
    try:
        while True:
            if not cap.grab():
                break
            if raw_idx % step == 0:
                ok, frame = cap.retrieve()
                if not ok:
                    if diagnostics is not None:
                        diagnostics.append(
                            f"decoding stopped early near frame {raw_idx} "
                            f"(truncated or corrupt video?)"
                        )
                    break
                pos_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
                t = pos_msec / 1000.0 if pos_msec and pos_msec > 0 else raw_idx / native
                yield out_idx, t, frame
                out_idx += 1
                if max_frames is not None and out_idx >= max_frames:
                    break
            raw_idx += 1
    finally:
        cap.release()


def _capture_frame_count(seconds: float, fps: float) -> int:
    from math import isfinite

    if not isfinite(seconds) or seconds <= 0 or not isfinite(fps) or fps <= 0:
        raise ValueError("capture duration and fps must be finite and greater than 0")
    frame_count = seconds * fps
    if not isfinite(frame_count) or frame_count < 1:
        raise ValueError("capture duration and fps must produce a finite count of at least one frame")
    return int(frame_count)


def _open_capture_writer(out_path, fps, size):
    import cv2

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, size)
    if not writer.isOpened():
        writer.release()
        raise RuntimeError(f"cannot open capture output: {out_path}")
    return writer


def capture_screen(out_path: str, seconds: float, fps: float = 10.0, monitor: int = 1) -> str:
    """Record the screen into out_path. Requires the optional `mss` dependency
    (`pip install mss`). Captures the given monitor (1 = primary)."""
    n = _capture_frame_count(seconds, fps)
    import time

    import cv2
    import numpy as np

    try:
        import mss
    except ImportError as e:  # pragma: no cover - depends on optional extra
        raise RuntimeError(
            "screen capture needs the 'mss' package: pip install mss (or screex[capture])"
        ) from e

    with mss.mss() as sct:
        mon = sct.monitors[monitor]
        w, h = mon["width"], mon["height"]
        writer = _open_capture_writer(out_path, fps, (w, h))
        try:
            period = 1.0 / fps
            for _ in range(n):
                start = time.time()
                shot = np.asarray(sct.grab(mon))  # BGRA
                writer.write(cv2.cvtColor(shot, cv2.COLOR_BGRA2BGR))
                sleep = period - (time.time() - start)
                if sleep > 0:
                    time.sleep(sleep)
        finally:
            writer.release()
    return str(out_path)


def capture_webcam(out_path: str, seconds: float, fps: float = 15.0, device: int = 0) -> str:
    """Record a short clip from a webcam into out_path. Manual/hardware path."""
    n = _capture_frame_count(seconds, fps)
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(device)
    writer = None
    try:
        if not cap.isOpened():
            raise RuntimeError("cannot open webcam")
        for _ in range(n):
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if writer is None:
                h, w = np.asarray(frame).shape[:2]
                writer = _open_capture_writer(out_path, fps, (w, h))
            writer.write(frame)
        if writer is None:
            raise RuntimeError("webcam did not return any frames")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
    return str(out_path)
