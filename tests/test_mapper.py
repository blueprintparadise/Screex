import numpy as np
from screex.core import mapper


def test_gray_to_ascii_endpoints():
    ramp = " .:-=+*#%@"  # 10 chars; index = v*9//255
    arr = np.array([[0, 255], [255, 0]], dtype=np.uint8)
    out = mapper.gray_to_ascii(arr, ramp=ramp)
    assert out == " @\n@ "


def test_auto_rows_halves_for_char_aspect():
    # square image, char cells ~2x tall -> rows ~ cols/2
    assert mapper.auto_rows(width=100, height=100, cols=120, char_aspect=2.0) == 60


def test_frame_to_ascii_shape():
    gray = np.tile(np.arange(256, dtype=np.uint8), (64, 1))  # 64x256 gradient
    out = mapper.frame_to_ascii(gray, cols=40)
    lines = out.split("\n")
    assert all(len(line) == 40 for line in lines)
    assert len(lines) == mapper.auto_rows(256, 64, 40)


def test_edge_magnitude_uniform_is_zero():
    flat = np.full((8, 8), 128, dtype=np.uint8)
    assert int(mapper.edge_magnitude(flat).max()) == 0
