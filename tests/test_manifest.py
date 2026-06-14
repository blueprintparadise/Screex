from screex.core.manifest import EventRecord, FrameRecord, Manifest


def test_manifest_roundtrip(tmp_path):
    m = Manifest(
        video="door.mp4", duration=12.4, sampled_fps=5.0, cols=120,
        frames=[
            FrameRecord(idx=0, t=0.0, score=0.0, event=False,
                        ascii="frames/00000.txt", png="frames/00000.png"),
            FrameRecord(idx=1, t=0.2, score=0.78, event=True,
                        ascii="frames/00001.txt", png="frames/00001.png"),
        ],
        events=[EventRecord(t_start=0.2, t_end=0.2, peak_frame=1, peak_score=0.78)],
    )
    path = tmp_path / "manifest.json"
    m.save(path)
    loaded = Manifest.load(path)
    assert loaded == m
    assert loaded.frames[1].event is True
    assert loaded.events[0].peak_frame == 1


def test_event_record_defaults_and_backward_compat():
    e = EventRecord(t_start=0.2, t_end=0.4, peak_frame=2, peak_score=0.6)
    assert e.type == "motion"
    assert e.confidence == 0.0

    # old-style manifest dict without type/confidence still loads
    old = {
        "video": "x.mp4", "duration": 1.0, "sampled_fps": 5.0, "cols": 10,
        "frames": [],
        "events": [{"t_start": 0.2, "t_end": 0.4, "peak_frame": 2, "peak_score": 0.6}],
    }
    m = Manifest.from_dict(old)
    assert m.events[0].type == "motion"
    assert m.events[0].confidence == 0.0

    # new event round-trips type/confidence
    e2 = EventRecord(t_start=0.0, t_end=0.0, peak_frame=0, peak_score=0.9,
                     type="cut", confidence=0.8)
    m2 = Manifest(video="y", duration=0.0, sampled_fps=5.0, cols=10, frames=[], events=[e2])
    loaded = Manifest.from_dict(m2.to_dict())
    assert loaded.events[0].type == "cut"
    assert loaded.events[0].confidence == 0.8
