import importlib.util
import json
from pathlib import Path


def _load_converter():
    path = Path(__file__).resolve().parent.parent / "scripts" / "guiworld_to_qa.py"
    spec = importlib.util.spec_from_file_location("screex_guiworld_to_qa", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _sample_rows():
    return [
        {  # desktop, list-form Options
            "video_path": "macOS/0.mov", "system": "macOS", "app": "Mail",
            "MCQA": {"Question": "How many unread emails are shown?",
                     "Options": ["A) 1", "B) 3", "C) 5"], "Correct Answer": "B"},
        },
        {  # desktop, dict-form Options
            "video_path": "Web/1.mov", "system": "Web", "app": "Safari",
            "MCQA": {"Question": "Click the settings button to do what?",
                     "Options": {"A": "open settings", "B": "close tab"},
                     "Correct Answer": "A"},
        },
        {  # mobile -> filtered out under domain=desktop
            "video_path": "IOS/2.mov", "system": "iOS", "app": "Health",
            "MCQA": {"Question": "What screen is shown?",
                     "Options": ["A) Home", "B) Profile"], "Correct Answer": "A"},
        },
        {  # malformed: single option -> skipped
            "video_path": "macOS/3.mov", "system": "macOS",
            "MCQA": {"Question": "broken?", "Options": ["only one"], "Correct Answer": "A"},
        },
    ]


def test_normalize_options_list_and_dict():
    conv = _load_converter()
    assert conv.normalize_options(["A) x", "B) y"]) == ["A) x", "B) y"]
    assert conv.normalize_options({"B": "two", "A": "one"}) == ["A) one", "B) two"]  # ordered
    assert conv.normalize_options("not options") is None
    assert conv.normalize_options(["only one"]) is None


def test_classify_bucket():
    conv = _load_converter()
    assert conv.classify_bucket("How many tabs are open?") == "count"
    assert conv.classify_bucket("Click the save button") == "action"
    assert conv.classify_bucket("What color is the icon?") == "visual"
    assert conv.classify_bucket("What is the user's name?") == "state"


def test_convert_desktop_filter_and_skips():
    conv = _load_converter()
    qa, stats = conv.convert(_sample_rows(), domain="desktop")
    clips = [item["clip"] for item in qa]
    assert "IOS/2.mov" not in clips                       # mobile filtered out
    assert stats["emitted"] == 2                          # two valid desktop MCQs
    assert stats["skipped"] == 1                          # the single-option row
    first = next(i for i in qa if i["clip"] == "macOS/0.mov")
    assert first["type"] == "count" and first["answer"] == "B"
    assert first["choices"] == ["A) 1", "B) 3", "C) 5"]


def test_convert_limit_and_all_domain():
    conv = _load_converter()
    qa, _ = conv.convert(_sample_rows(), domain="all", limit=1)
    assert len(qa) == 1
    qa_all, _ = conv.convert(_sample_rows(), domain="all")
    assert any(i["clip"] == "IOS/2.mov" for i in qa_all)  # mobile included under 'all'


def test_load_rows_list_and_wrapped(tmp_path):
    conv = _load_converter()
    p = tmp_path / "list.json"
    p.write_text(json.dumps(_sample_rows()), encoding="utf-8")
    assert len(conv.load_rows(p)) == 4
    w = tmp_path / "wrapped.json"
    w.write_text(json.dumps({"train": _sample_rows()}), encoding="utf-8")
    assert len(conv.load_rows(w)) == 4
