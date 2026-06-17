import numpy as np

from screex.core.regiondiff import changed_region


def _blank(h=480, w=640):
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_none_on_missing_or_mismatched():
    assert changed_region(None, _blank()) is None
    assert changed_region(_blank(480, 640), _blank(240, 320)) is None


def test_none_when_identical():
    a = _blank()
    assert changed_region(a, a.copy()) is None


def test_full_change():
    a, b = _blank(), _blank()
    b[:, :] = 255
    rc = changed_region(a, b)
    assert rc.shape == "full"
    assert rc.coverage > 0.45


def test_local_change():
    a, b = _blank(), _blank()
    b[10:30, 10:30] = 255  # tiny patch
    rc = changed_region(a, b)
    assert rc.shape == "local"
    assert rc.box[0] <= 10 and rc.box[1] <= 10


def test_overlay_change():
    a, b = _blank(), _blank()
    b[140:340, 170:470] = 255  # centered mid-size block
    rc = changed_region(a, b)
    assert rc.shape == "overlay"


def test_band_change():
    a, b = _blank(), _blank()
    b[0:300, :][::3] = 255  # sparse rows across a tall full-width area
    rc = changed_region(a, b)
    assert rc.shape == "band"
