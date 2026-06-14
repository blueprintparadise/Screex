from screex.core.index import ScreenState, ScreenIndex


def test_screen_index_roundtrip(tmp_path):
    si = ScreenIndex(
        video="cast.mp4", duration=12.0, sampled_fps=2.0,
        states=[
            ScreenState(idx=0, t_start=0.0, t_end=3.0,
                        thumbnail="frames/00000_thumb.png", keyframe="frames/00000.png",
                        ocr_text=["Open Settings"], text_added=["Open Settings"], text_removed=[]),
            ScreenState(idx=1, t_start=3.0, t_end=6.0,
                        thumbnail="frames/00001_thumb.png", keyframe="frames/00001.png",
                        ocr_text=["Save Changes"], text_added=["Save Changes"], text_removed=["Open Settings"]),
        ],
    )
    path = tmp_path / "index.json"
    si.save(path)
    loaded = ScreenIndex.load(path)
    assert loaded == si
    assert loaded.states[1].text_removed == ["Open Settings"]
