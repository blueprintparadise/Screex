# Screex — Screen-Recording Understanding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rename the project Squint→Screex and build `screex index <recording>`, which turns a screen recording into a queryable ScreenIndex (UI states with thumbnail, full-res keyframe, OCR text, and text-diff), with a SKILL.md driving three views (transcript / Q&A / doc).

**Architecture:** Reuse `source` (decode/sample) and `analyzer.motion_score` (change detection). New `segment` groups frames into settled UI states; new `ocr` wraps RapidOCR (`extract_text` + `text_diff`); new `index` holds the `ScreenState`/`ScreenIndex` schema; a new `index` CLI subcommand orchestrates them. ASCII modules stay in the repo but are off the critical path.

**Tech Stack:** Python 3, opencv-python, numpy, `rapidocr-onnxruntime` (pip-only OCR, verified working), pytest. Builds on the existing repo (master, 19 passing tests).

---

## File Structure

```
screex/                         # renamed from squint/ (Task 1)
  core/
    source.py     # REUSE
    analyzer.py   # REUSE (motion_score); other fns unused here
    mapper.py     # kept, unused on this path
    manifest.py   # kept, unused on this path
    segment.py    # NEW — Segment + segment_stream
    ocr.py        # NEW — extract_text + text_diff (RapidOCR)
    index.py      # NEW — ScreenState + ScreenIndex
  cli.py          # add `index` subcommand (keep analyze/capture)
SKILL.md          # rewritten for Screex (Task 6)
README.md         # rewritten for Screex (Task 6)
tests/
  conftest.py     # add screencast_video fixture (Task 5)
  test_ocr.py     # NEW
  test_segment.py # NEW
  test_index.py   # NEW
  test_cli.py     # add index integration test
```

---

### Task 1: Rename Squint → Screex and add RapidOCR

**Files:**
- Rename: `squint/` → `screex/` (directory)
- Modify: every `.py` importing `squint`, `requirements.txt`

- [ ] **Step 1: Rename the package directory**

Run:
```
git mv squint screex
```

- [ ] **Step 2: Update all imports from `squint` to `screex`**

Find every reference: `grep -rl "squint" --include=*.py .`
Expected files: `screex/core/analyzer.py`, `screex/cli.py`, `tests/test_manifest.py`,
`tests/test_mapper.py`, `tests/test_analyzer.py`, `tests/test_source.py`, `tests/test_cli.py`.

In each, replace `squint` with `screex` (the imports are all of the form `from squint.core...`
/ `from squint.cli...`). After editing, verify none remain:
Run: `grep -rn "squint" --include=*.py .`
Expected: no matches.

- [ ] **Step 3: Add RapidOCR to requirements.txt and ignore generated indexes**

Replace the contents of `requirements.txt` with:
```
opencv-python
numpy
rapidocr-onnxruntime
pytest
```

Add `*.screex/` to `.gitignore` (so generated indexes stay untracked). The `.gitignore`
"Squint generated artifacts" section should become:
```
# Generated artifacts
*.squint/
*.screex/
demo.avi
capture.mp4
```

- [ ] **Step 4: Verify the full suite still passes after the rename**

Run: `python -m pytest -q`
Expected: `19 passed`.

- [ ] **Step 5: Commit**

```
git add -A
git commit -m "refactor: rename package squint -> screex; add rapidocr dependency"
```

---

### Task 2: `ocr.py` — text extraction + diff

**Files:**
- Create: `screex/core/ocr.py`
- Test: `tests/test_ocr.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_ocr.py`:
```python
import numpy as np
import cv2
from screex.core import ocr


def test_extract_text_reads_rendered_text():
    img = np.full((140, 640, 3), 255, dtype=np.uint8)
    cv2.putText(img, "Open Settings", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 3)
    lines = ocr.extract_text(img)
    joined = " ".join(lines).lower()
    assert "open" in joined
    assert "settings" in joined


def test_text_diff_added_and_removed():
    added, removed = ocr.text_diff(["a", "b"], ["b", "c"])
    assert added == ["c"]
    assert removed == ["a"]


def test_text_diff_empty_prev():
    added, removed = ocr.text_diff([], ["x", "y"])
    assert added == ["x", "y"]
    assert removed == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_ocr.py -v`
Expected: FAIL with `ModuleNotFoundError: screex.core.ocr`.

- [ ] **Step 3: Implement**

`screex/core/ocr.py`:
```python
from __future__ import annotations

_engine = None


def _get_engine():
    global _engine
    if _engine is None:
        from rapidocr_onnxruntime import RapidOCR
        _engine = RapidOCR()
    return _engine


def extract_text(bgr) -> list:
    """Return on-screen text lines from a BGR frame, in reading order (top->bottom, left->right)."""
    engine = _get_engine()
    result, _ = engine(bgr)
    if not result:
        return []

    def sort_key(item):
        box = item[0]  # 4 corner points [[x,y],...]
        ys = [p[1] for p in box]
        xs = [p[0] for p in box]
        return (round(min(ys) / 10.0), min(xs))

    ordered = sorted(result, key=sort_key)
    return [str(item[1]).strip() for item in ordered if str(item[1]).strip()]


def text_diff(prev_lines, cur_lines):
    """Return (added, removed): lines in cur not in prev, and lines in prev not in cur."""
    prev_set = set(prev_lines)
    cur_set = set(cur_lines)
    added = [line for line in cur_lines if line not in prev_set]
    removed = [line for line in prev_lines if line not in cur_set]
    return added, removed
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_ocr.py -v`
Expected: PASS (3 tests). The first run downloads/initializes RapidOCR models (a few seconds).
If `test_extract_text_reads_rendered_text` fails to read the text, do NOT weaken the assertion —
report BLOCKED with the actual `extract_text` output so we can adjust rendering (font size) — the
synthetic text is large (scale 1.6, thickness 3) specifically so OCR can read it.

- [ ] **Step 5: Commit**

```
git add screex/core/ocr.py tests/test_ocr.py
git commit -m "feat: add OCR text extraction and text-diff"
```

---

### Task 3: `segment.py` — group frames into UI states

**Files:**
- Create: `screex/core/segment.py`
- Test: `tests/test_segment.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_segment.py`:
```python
import numpy as np
from screex.core import segment


def _frames(seq):
    # seq: list of (t, fill_value); yields (idx, t, bgr 20x20)
    for idx, (t, val) in enumerate(seq):
        yield idx, t, np.full((20, 20, 3), val, dtype=np.uint8)


def test_segment_stream_two_states_on_big_change():
    seq = [(0.0, 0), (0.5, 0), (1.0, 0), (1.5, 255), (2.0, 255), (2.5, 255)]
    segs = list(segment.segment_stream(_frames(seq), change_threshold=0.5))
    assert len(segs) == 2
    assert segs[0].t_start == 0.0
    assert segs[0].t_end == 1.0
    assert segs[1].t_start == 1.5
    assert segs[1].t_end == 2.5
    assert int(segs[0].frame_bgr.mean()) == 0     # settled (black) keyframe
    assert int(segs[1].frame_bgr.mean()) == 255   # settled (white) keyframe


def test_segment_stream_single_state_when_static():
    seq = [(0.0, 100), (1.0, 100), (2.0, 100), (3.0, 100)]
    segs = list(segment.segment_stream(_frames(seq), change_threshold=0.5))
    assert len(segs) == 1
    assert segs[0].t_start == 0.0
    assert segs[0].t_end == 3.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_segment.py -v`
Expected: FAIL with `ModuleNotFoundError: screex.core.segment`.

- [ ] **Step 3: Implement**

`screex/core/segment.py`:
```python
from __future__ import annotations

from dataclasses import dataclass

from screex.core.analyzer import motion_score


@dataclass
class Segment:
    idx: int
    t_start: float
    t_end: float
    frame_bgr: object  # numpy BGR array: the settled keyframe of this UI state


def segment_stream(frames, change_threshold: float = 0.04):
    """Yield one Segment per UI state. A new state begins when frame-to-frame motion
    crosses change_threshold; the representative keyframe is the last (settled) frame
    of the state. Holds only the current state's last frame (memory-bounded)."""
    import cv2

    prev_gray = None
    seg_idx = 0
    cur_start_t = None
    last_t = None
    last_bgr = None

    for idx, t, bgr in frames:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        if prev_gray is None:
            cur_start_t = t
        elif motion_score(prev_gray, gray) >= change_threshold:
            yield Segment(seg_idx, cur_start_t, last_t, last_bgr)
            seg_idx += 1
            cur_start_t = t
        prev_gray = gray
        last_t = t
        last_bgr = bgr

    if last_bgr is not None:
        yield Segment(seg_idx, cur_start_t, last_t, last_bgr)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_segment.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```
git add screex/core/segment.py tests/test_segment.py
git commit -m "feat: add UI-state segmentation"
```

---

### Task 4: `index.py` — ScreenState / ScreenIndex schema

**Files:**
- Create: `screex/core/index.py`
- Test: `tests/test_index.py`

- [ ] **Step 1: Write the failing test**

`tests/test_index.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_index.py -v`
Expected: FAIL with `ModuleNotFoundError: screex.core.index`.

- [ ] **Step 3: Implement**

`screex/core/index.py`:
```python
from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path


@dataclass
class ScreenState:
    idx: int
    t_start: float
    t_end: float
    thumbnail: str
    keyframe: str
    ocr_text: list = field(default_factory=list)
    text_added: list = field(default_factory=list)
    text_removed: list = field(default_factory=list)


@dataclass
class ScreenIndex:
    video: str
    duration: float
    sampled_fps: float
    states: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "video": self.video,
            "duration": self.duration,
            "sampled_fps": self.sampled_fps,
            "states": [asdict(s) for s in self.states],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ScreenIndex":
        return cls(
            video=d["video"],
            duration=d["duration"],
            sampled_fps=d["sampled_fps"],
            states=[ScreenState(**s) for s in d["states"]],
        )

    def save(self, path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "ScreenIndex":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_index.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```
git add screex/core/index.py tests/test_index.py
git commit -m "feat: add ScreenState/ScreenIndex schema"
```

---

### Task 5: `index` CLI subcommand (wire the pipeline)

**Files:**
- Modify: `screex/cli.py`
- Modify: `tests/conftest.py` (add `screencast_video` fixture)
- Test: `tests/test_cli.py` (append integration test)

- [ ] **Step 1: Add the synthetic-screencast fixture** — append to `tests/conftest.py`:

```python
@pytest.fixture
def screencast_video(tmp_path):
    """A synthetic screencast: two visually-distinct UI states (different bg tint + text),
    so the change detector fires deterministically and OCR can read the text."""
    import cv2
    import numpy as np

    path = tmp_path / "cast.avi"
    w, h, fps = 360, 240, 4
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
    assert vw.isOpened(), "MJPG VideoWriter failed to open"
    states = [((255, 255, 255), "Open Settings"), ((225, 235, 255), "Save Changes")]
    for bg, text in states:
        for _ in range(6):
            frame = np.full((h, w, 3), bg, dtype=np.uint8)
            cv2.putText(frame, text, (15, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3)
            vw.write(frame)
    vw.release()
    return path
```

- [ ] **Step 2: Write the failing integration test** — append to `tests/test_cli.py`:

```python
def test_index_builds_states_with_ocr(screencast_video, tmp_path):
    from screex.cli import index
    from screex.core.index import ScreenIndex

    out = tmp_path / "work_index"
    index_path = index(str(screencast_video), fps=4.0, change_threshold=0.03, out=str(out))
    assert index_path.exists()

    si = ScreenIndex.load(index_path)
    assert si.video == screencast_video.name
    assert len(si.states) >= 2

    for s in si.states:
        assert (out / s.thumbnail).exists()
        assert (out / s.keyframe).exists()

    all_text = " ".join(" ".join(s.ocr_text) for s in si.states).lower()
    assert any(word in all_text for word in ("settings", "save", "open", "changes"))
    # a later state registers newly-appeared text
    assert any(s.text_added for s in si.states[1:])
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/test_cli.py::test_index_builds_states_with_ocr -v`
Expected: FAIL with `ImportError: cannot import name 'index'`.

- [ ] **Step 4: Implement** — add to `screex/cli.py`.

4a. Add the `index` function (place after the existing `analyze` function):

```python
def index(recording, fps=2.0, change_threshold=0.04, thumb_width=320, out=None):
    import cv2
    from screex.core import source, segment, ocr
    from screex.core.index import ScreenState, ScreenIndex

    recording = Path(recording)
    info = source.video_info(str(recording))
    out_dir = Path(out) if out else recording.parent / f"{recording.stem}.screex"
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    states = []
    prev_ocr = []
    for seg in segment.segment_stream(source.iter_frames(str(recording), fps), change_threshold):
        bgr = seg.frame_bgr
        text = ocr.extract_text(bgr)
        added, removed = ocr.text_diff(prev_ocr, text)
        prev_ocr = text

        name = f"{seg.idx:05d}"
        key_rel = f"frames/{name}.png"
        thumb_rel = f"frames/{name}_thumb.png"
        cv2.imwrite(str(out_dir / key_rel), bgr)
        th = int(bgr.shape[0] * thumb_width / bgr.shape[1])
        cv2.imwrite(str(out_dir / thumb_rel), cv2.resize(bgr, (thumb_width, th)))

        states.append(ScreenState(
            idx=seg.idx, t_start=round(seg.t_start, 3), t_end=round(seg.t_end, 3),
            thumbnail=thumb_rel, keyframe=key_rel,
            ocr_text=text, text_added=added, text_removed=removed,
        ))

    screen_index = ScreenIndex(
        video=recording.name, duration=round(info["duration"], 3),
        sampled_fps=fps, states=states,
    )
    index_path = out_dir / "index.json"
    screen_index.save(index_path)
    return index_path
```

4b. In `main()`, add the `index` subparser. Find the line that creates the `capture` subparser
(`c = sub.add_parser("capture", ...)`) and add this block immediately before it:

```python
    ix = sub.add_parser("index", help="build a ScreenIndex from a screen recording")
    ix.add_argument("recording")
    ix.add_argument("--fps", type=float, default=2.0, help="frames sampled per second")
    ix.add_argument("--change-threshold", type=float, default=0.04,
                    help="motion fraction (0..1) that marks a new UI state")
    ix.add_argument("--thumb-width", type=int, default=320, help="thumbnail width in px")
    ix.add_argument("--out", default=None, help="output dir (default <recording>.screex)")
```

4c. In `main()`, add dispatch handling. Find the `if args.cmd == "analyze":` block and add an
`elif` branch for `index` immediately after the analyze branch (before the `capture` branch):

```python
    elif args.cmd == "index":
        path = index(args.recording, fps=args.fps, change_threshold=args.change_threshold,
                     thumb_width=args.thumb_width, out=args.out)
        print(f"index: {path}")
```

- [ ] **Step 5: Run the test and the full suite**

Run: `python -m pytest tests/test_cli.py::test_index_builds_states_with_ocr -v`
Expected: PASS.
Run: `python -m pytest -q`
Expected: all PASS (24 tests).
Run: `python -m screex.cli index --help`
Expected: shows the `index` subcommand with `--fps`, `--change-threshold`, `--thumb-width`, `--out`.

- [ ] **Step 6: Commit**

```
git add screex/cli.py tests/conftest.py tests/test_cli.py
git commit -m "feat: add `screex index` building a ScreenIndex from a recording"
```

---

### Task 6: Rewrite SKILL.md and README for Screex

**Files:**
- Overwrite: `SKILL.md`
- Overwrite: `README.md`

- [ ] **Step 1: Overwrite `SKILL.md`**

`SKILL.md`:
```markdown
---
name: screex
description: Use when the user wants Claude to understand a screen recording / screencast / demo / bug-repro video — e.g. "what are the steps in this recording?", "turn this into a how-to doc", "write a bug report from this repro", "what URL did they open?". Screex builds a queryable index of UI states (with on-screen text) and Claude reads it to produce a transcript, answer questions, or generate docs.
---

# Screex — screen-recording understanding

## When to use
The user points you at a screen recording (a screencast, demo, tutorial, or bug repro) and
wants a step transcript, a how-to doc, a bug report, or answers to questions about it.

## Build the index
Run:
`python -m screex.cli index <recording> --fps 2`
(raise `--fps` for fast-moving recordings; lower `--change-threshold` to split states more
eagerly.) This writes `<recording>.screex/index.json` plus per-state `frames/NNNNN.png`
(full-res keyframe) and `frames/NNNNN_thumb.png` (thumbnail).

## Read the index
`Read` `index.json`. It is an ordered list of UI `states`, each with `t_start`/`t_end`,
`ocr_text` (the on-screen text), `text_added` / `text_removed` (what text appeared or
disappeared vs the previous state — the strongest signal of what the user did), and paths to
a `thumbnail` and full-res `keyframe`. The on-screen text is plain text — reading it across
states is cheap.

## Produce one of three views

- **Action transcript:** walk the states in order; use `text_added`/`text_removed` plus the
  thumbnail to narrate timestamped steps, e.g. "0:04 opened Settings; 0:09 entered an API
  key; 0:14 an 'invalid key' error appeared."
- **Q&A:** answer the user's question by scanning `ocr_text` across states (cheap). `Read`
  the full-res `keyframe` PNG for a state only when the text is insufficient (small icons,
  layout, colour).
- **Doc / bug report:** format the transcript into a how-to guide, or a structured
  reproduction report (steps to reproduce, expected vs actual).

## Cost discipline
The `ocr_text` and `text_*` fields are text and nearly free to read. Escalate to a
`keyframe` image only for the few states where the text doesn't answer the question.
```

- [ ] **Step 2: Overwrite `README.md`**

`README.md`:
```markdown
# Screex

Screen-recording understanding for Claude. Screex turns a screencast into a queryable
**index** of UI states — each with the on-screen text (OCR), what text changed since the
previous state, a thumbnail, and a full-resolution keyframe — so Claude can produce an
action transcript, answer questions, or generate a how-to / bug report from a recording.

Training-free, model-agnostic, and `pip install`-only (OCR via `rapidocr-onnxruntime`, no
system binaries).

## Install
```
pip install -r requirements.txt
```

## Use (standalone)
```
python -m screex.cli index path/to/recording.mp4 --fps 2
```
Produces `path/to/recording.screex/index.json` + `frames/` (keyframes + thumbnails).

Options: `--fps`, `--change-threshold` (0..1; lower = more states), `--thumb-width`, `--out`.

## Use (as a Claude skill)
`SKILL.md` drives Claude through: build index → read `index.json` → produce a transcript,
answer questions, or generate a how-to / bug report.

## Architecture
`screex/core/`: `source` (decode/sample), `segment` (group frames into settled UI states),
`ocr` (RapidOCR text + text-diff), `index` (ScreenState/ScreenIndex schema). The `index` CLI
wires them into `index.json`.
```

- [ ] **Step 3: Sanity-check and run the suite**

Run: `python -c "t=open('SKILL.md',encoding='utf-8').read(); assert t.startswith('---'); assert 'screex' in t; print('ok')"`
Expected: prints `ok`.
Run: `python -m pytest -q`
Expected: all PASS (24 tests).

- [ ] **Step 4: Commit**

```
git add SKILL.md README.md
git commit -m "docs: rewrite SKILL.md and README for Screex"
```

---

## Self-Review

- **Spec coverage:**
  - Rename Squint→Screex + RapidOCR dep (spec §0, §3, §4) → Task 1. ✓
  - `ocr.extract_text` + `text_diff` (spec §3.1) → Task 2. ✓
  - `segment.segment_stream` + `Segment`, settled keyframe, memory-bounded (spec §3.2) → Task 3. ✓
  - `ScreenState`/`ScreenIndex` schema + save/load (spec §3.3) → Task 4. ✓
  - `index` CLI subcommand + flags, thumbnail + keyframe + OCR + text-diff (spec §3.4) → Task 5. ✓
  - Three views in SKILL.md (spec §3.5) + README → Task 6. ✓
  - Testing (spec §5): ocr/text_diff → Task 2; segment_stream → Task 3; index roundtrip → Task 4; CLI integration on synthetic screencast → Task 5. Manual real-recording proof → noted post-build. ✓
  - Out-of-scope (spec §4): no audio/cursor/UI-grounding/trained-models/PySceneDetect; old ASCII path left intact. ✓
- **Placeholder scan:** no TBD/TODO; every code/edit step shows full code. ✓
- **Type consistency:** `extract_text(bgr)->list`, `text_diff(prev,cur)->(added,removed)`,
  `Segment(idx,t_start,t_end,frame_bgr)`, `segment_stream(frames,change_threshold)`,
  `ScreenState(idx,t_start,t_end,thumbnail,keyframe,ocr_text,text_added,text_removed)`,
  `ScreenIndex(video,duration,sampled_fps,states)`, `index(recording,fps,change_threshold,thumb_width,out)`
  are used identically across Tasks 2–6. The `index` CLI reads `seg.frame_bgr`, `seg.idx`,
  `seg.t_start`, `seg.t_end` — all defined on `Segment` in Task 3. ✓

**Manual real-recording proof (run after Task 5/6, not a committed test):**
```
python -m screex.cli index <a-real-screen-recording.mp4> --fps 2 --out demo.screex
```
Inspect `demo.screex/index.json`: states should align with real UI changes and `ocr_text`
should capture visible on-screen text. The `.screex` suffix is git-ignored (added in Task 1);
remove with `rm -rf demo.screex` afterward.
