import importlib.util
from pathlib import Path

from screex.core.index import ScreenIndex, ScreenState


def _load_eval_module():
    path = Path(__file__).resolve().parent.parent / "scripts" / "eval.py"
    spec = importlib.util.spec_from_file_location("screex_eval_script", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_estimate_reuses_existing_index(screencast_video, tmp_path, monkeypatch):
    eval_script = _load_eval_module()
    index_path = tmp_path / "index.json"
    ScreenIndex(
        video=screencast_video.name,
        duration=2.0,
        sampled_fps=2.0,
        states=[
            ScreenState(0, 0.0, 1.0, "frames/0_thumb.png", "frames/0.png",
                        ocr_text=["Open Settings"], text_added=["Open Settings"]),
            ScreenState(1, 1.0, 2.0, "frames/1_thumb.png", "frames/1.png",
                        ocr_text=["Save"], text_added=["Save"]),
        ],
    ).save(index_path)

    def fail_index(*args, **kwargs):
        raise AssertionError("estimate should reuse --from-index")

    monkeypatch.setattr("screex.cli.index", fail_index)
    result = eval_script.estimate(
        str(screencast_video),
        fps=2.0,
        escalate=1,
        tokens_per_image=1000,
        from_index=str(index_path),
    )

    assert result["states"] == 2
    assert result["escalated_images"] == 1
    assert result["screex_tokens"] > 1000


def test_estimate_can_disable_audio_when_building(monkeypatch, tmp_path):
    eval_script = _load_eval_module()
    calls = []
    index_path = tmp_path / "index.json"
    ScreenIndex(
        video="clip.mp4",
        duration=1.0,
        sampled_fps=1.0,
        states=[ScreenState(0, 0.0, 1.0, "t.png", "k.png", ocr_text=["Hi"])],
    ).save(index_path)

    def fake_index(recording, fps, out=None, quiet=True, audio=True):
        calls.append({"recording": recording, "fps": fps, "audio": audio})
        return index_path

    monkeypatch.setattr("screex.cli.index", fake_index)
    monkeypatch.setattr("screex.core.source.video_info", lambda path: {"duration": 1.0})

    eval_script.estimate(
        "clip.mp4",
        fps=1.0,
        escalate=1,
        tokens_per_image=1000,
        audio=False,
    )

    assert calls == [{"recording": "clip.mp4", "fps": 1.0, "audio": False}]
