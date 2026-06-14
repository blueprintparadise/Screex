from __future__ import annotations

import numpy as np

RAMP = " .:-=+*#%@"


def gray_to_ascii(gray: np.ndarray, ramp: str = RAMP) -> str:
    arr = np.asarray(gray, dtype=np.uint16)
    n = len(ramp) - 1
    idx = (arr * n // 255).astype(np.intp)
    lut = np.array(list(ramp))
    chars = lut[idx]
    return "\n".join("".join(row) for row in chars)


def auto_rows(width: int, height: int, cols: int, char_aspect: float = 2.0) -> int:
    return max(1, round(cols * (height / width) / char_aspect))


def edge_magnitude(gray: np.ndarray) -> np.ndarray:
    g = np.asarray(gray, dtype=np.float32)
    gy, gx = np.gradient(g)
    mag = np.hypot(gx, gy)
    peak = float(mag.max())
    if peak > 0:
        mag = mag / peak * 255.0
    return mag.astype(np.uint8)


def frame_to_ascii(gray: np.ndarray, cols: int, ramp: str = RAMP, edge: bool = False) -> str:
    import cv2

    h, w = gray.shape[:2]
    rows = auto_rows(w, h, cols)
    small = cv2.resize(gray, (cols, rows), interpolation=cv2.INTER_AREA)
    if edge:
        small = edge_magnitude(small)
    return gray_to_ascii(small, ramp)
