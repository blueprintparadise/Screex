from __future__ import annotations


def _open(path):
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {path}")
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


def iter_frames(path: str, sample_fps: float, max_frames: int | None = None):
    """Yield (out_idx, t_seconds, bgr) for sampled frames.

    The timestamp prefers the container's real position (CAP_PROP_POS_MSEC), which is
    correct for variable-frame-rate recordings, and falls back to raw_idx/native only
    when the container does not report a position. ``max_frames`` caps the number of
    sampled frames returned (None = no cap)."""
    cap = _open(path)
    import cv2

    native = cap.get(cv2.CAP_PROP_FPS) or sample_fps or 1.0
    step = max(1, round(native / sample_fps)) if sample_fps else 1
    raw_idx = 0
    out_idx = 0
    try:
        while True:
            pos_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
            if not cap.grab():
                break
            if raw_idx % step == 0:
                ok, frame = cap.retrieve()
                if not ok:
                    break
                t = pos_msec / 1000.0 if pos_msec and pos_msec > 0 else raw_idx / native
                yield out_idx, t, frame
                out_idx += 1
                if max_frames is not None and out_idx >= max_frames:
                    break
            raw_idx += 1
    finally:
        cap.release()


def capture_screen(out_path: str, seconds: float, fps: float = 10.0, monitor: int = 1) -> str:
    """Record the screen into out_path. Requires the optional `mss` dependency
    (`pip install mss`). Captures the given monitor (1 = primary)."""
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
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
        try:
            n = int(seconds * fps)
            period = 1.0 / fps if fps else 0.0
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
    """Record a short clip from the default webcam into out_path. Manual/hardware path."""
    import cv2

    cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        raise RuntimeError("cannot open webcam")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    try:
        for _ in range(int(seconds * fps)):
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(frame)
    finally:
        cap.release()
        writer.release()
    return str(out_path)
