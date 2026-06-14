# Screex — Screen-Recording Understanding Design Spec

**Date:** 2026-06-14
**Status:** Approved (pending spec review)
**Supersedes (as the project's primary direction):** the ASCII-perception north star
(`2026-06-14-squint-design.md`). Repositioning rationale below.

> **Project rename:** the project is renamed from **Squint** to **Screex** (screen + index;
> verified free on PyPI). Throughout this document, read the Python package `squint/` as
> **`screex/`**, the CLI command `squint ...` as **`screex ...`**, and default output dirs
> `<file>.squint/` as **`<recording>.screex/`**. The implementation plan renames the package
> as its first task and uses the new names authoritatively.

## 0. Why we repositioned

Research + two evals showed the original "ASCII as cheap LLM perception" thesis is not
defensible:
- The "ASCII → text for AI" framing is ASCILINE's own pitch, not novel.
- Frame-selection, adaptive cut detection, and coarse-to-fine escalation are all
  commoditized (VideoAgent, AKS, PySceneDetect, CARES).
- A controlled eval (independent judges, matched-or-favorable token budget) showed **ASCII
  is the worst skim layer**: low-res thumbnails won ~6/0 on scene identification and ~37×
  on token cost, and even read on-screen text ASCII could not.

Squint is repositioned to a concrete, underserved, NLP-relevant use case where the same
triage + coarse-to-fine machinery is genuinely useful and the inputs are text-rich:
**understanding screen recordings.**

## 1. Goal

`squint index <recording>` turns a screen recording into a structured, queryable
**ScreenIndex**: an ordered list of UI states, each with a timestamp range, a thumbnail, a
full-resolution keyframe, the on-screen text (OCR), and a text-diff vs the previous state.
Claude then drives three thin **views** over that index: an **action transcript**, **Q&A**,
and **doc/bug-report generation**.

The index is the substance; the three views are prompt/format layers in `SKILL.md` (little
or no extra code).

### Dual objective
- **Adoption (stars):** a frictionless (`pip install`-only) Claude skill that gives Claude
  eyes on screen recordings — demo-able in one GIF (recording → how-to doc / repro report).
- **Paper (EMNLP-style demo):** a training-free, model-agnostic, agent-native, open-source
  system for long-screen-recording understanding (the "history of screens" gap), backed by
  a reproducible cost/accuracy study.

## 2. Key signal insight

In screencasts, hard cuts are rare; the real events are **local UI changes**. The cheapest
strong signal is **OCR text-diff between consecutive states** — a dialog's text appears, a
URL changes, an error string shows up. So: a cheap **visual diff** finds candidate change
points, and OCR runs only on each segment's keyframe (not every frame); the **text-diff**
between consecutive keyframes is the primary action signal.

## 3. Architecture

Evolves the existing core. Reuses `source.py` (decode/sample) and `analyzer.motion_score`
(for change detection). The ASCII `mapper.py` is dropped from this path (kept in the repo,
unused). The old `manifest.py`/`analyze` path is left intact for backward compatibility but
superseded by the new index.

```
squint/core/
  source.py      # REUSE — decode/sample frames
  segment.py     # NEW — group frames into UI states; pick a settled keyframe per state
  ocr.py         # NEW — RapidOCR wrapper: extract_text(bgr) + text_diff(prev, cur)
  index.py       # NEW — ScreenState / ScreenIndex schema + save/load
cli.py           # NEW `index` subcommand (keeps existing `analyze`/`capture`)
SKILL.md         # rewritten — index → transcript / Q&A / doc views
```

### 3.1 `ocr.py`
- `extract_text(bgr) -> list[str]`: run RapidOCR on a BGR frame, return detected text
  strings in reading order (top-to-bottom, then left-to-right). Empty list if none. The
  RapidOCR engine is constructed lazily once (module-level singleton) because init is heavy.
- `text_diff(prev_lines, cur_lines) -> tuple[list[str], list[str]]`: returns
  `(added, removed)` — lines present in `cur` but not `prev`, and vice versa, order
  preserved, compared case-sensitively after stripping surrounding whitespace.

### 3.2 `segment.py`
- `Segment` (dataclass, internal): `idx`, `t_start`, `t_end`, `frame_bgr` (the settled
  representative keyframe of the state).
- `segment_stream(frames, change_threshold=0.04)`: consumes an iterator of `(idx, t, bgr)`
  (from `source.iter_frames`) and yields `Segment`s. State machine:
  - Convert each frame to grayscale; `score = motion_score(prev_gray, cur_gray)`.
  - A new state begins when `score >= change_threshold` (a meaningful UI change).
  - The representative keyframe of a state is the **last** frame before the next boundary
    (the settled UI). `t_start` = time of the state's first frame, `t_end` = time of its
    last frame.
  - Emits one `Segment` per state, including the final state at stream end.
  - Holds only the current state's last frame (memory-bounded for long recordings).

### 3.3 `index.py`
```python
@dataclass
class ScreenState:
    idx: int
    t_start: float
    t_end: float
    thumbnail: str        # rel path to small skim image
    keyframe: str         # rel path to full-res keyframe
    ocr_text: list        # detected on-screen text lines
    text_added: list      # lines vs previous state
    text_removed: list    # lines vs previous state

@dataclass
class ScreenIndex:
    video: str
    duration: float
    sampled_fps: float
    states: list          # list[ScreenState]
    # to_dict / from_dict / save(path) / load(path)  — JSON, same pattern as manifest.py
```

### 3.4 `cli.py` — `index` subcommand
`squint index <recording> [--fps 2] [--change-threshold 0.04] [--thumb-width 320]
[--out DIR]`:
1. For each `Segment` from `segment_stream(source.iter_frames(recording, fps), change_threshold)`:
   - `ocr_text = ocr.extract_text(segment.frame_bgr)`.
   - `text_added, text_removed = ocr.text_diff(prev_ocr, ocr_text)`.
   - Write full-res keyframe PNG and a `--thumb-width` thumbnail PNG.
   - Build a `ScreenState`.
2. Save `ScreenIndex` to `<out>/index.json` (default `<recording>.squint/`).

### 3.5 `SKILL.md` — the three views
After `squint index`, Claude `Read`s `index.json` and:
- **Transcript:** walk states in order; use `text_added`/`text_removed` + the thumbnail to
  narrate timestamped steps ("0:04 opened Settings; 0:09 typed 'api_key' into the token
  field; 0:14 an error 'invalid key' appeared").
- **Q&A:** find the answer by scanning `ocr_text` across states (it's text — cheap); escalate
  the `keyframe` PNG only when the text is insufficient.
- **Doc / bug-report:** format the transcript into a how-to guide or a structured
  reproduction report (steps, expected vs actual).

## 4. Scope

### In (v1)
- `ocr.py`, `segment.py`, `index.py`, the `index` CLI subcommand, rewritten `SKILL.md`.
- RapidOCR (`rapidocr-onnxruntime`) added to `requirements.txt` (pip-only, no system binary).
- Unit + integration tests; a synthetic-screencast proof.

### Out (YAGNI / later)
- Audio transcription, cursor/click detection, UI-element grounding/bounding boxes.
- Trained UI models (we are deliberately training-free / model-agnostic).
- PySceneDetect (our `motion_score` change detection suffices for v1; note as an option).
- The curated eval set + paper writeup (separate effort once the system works).
- Removing the old ASCII `analyze` path (left intact; not the focus).

## 5. Testing & success criteria

- **`ocr.extract_text`:** on a synthetic BGR image with known rendered text, returns a list
  containing that text (substring match, OCR may split/merge tokens).
- **`ocr.text_diff`:** `text_diff(["a","b"], ["b","c"])` → `(["c"], ["a"])`.
- **`segment.segment_stream`:** a synthetic frame iterator that is constant for N frames,
  changes, then constant again yields the expected number of segments with correct
  `t_start`/`t_end` and settled representative frames.
- **`index.py`:** `ScreenIndex` JSON round-trip (save → load → equality).
- **CLI integration:** on a generated synthetic screencast (frames rendering text that
  changes at known times), `index.json` has ≥2 states; each state's `thumbnail`/`keyframe`
  exist on disk; `ocr_text` contains the rendered text; `text_added` captures the change.
- **Manual real-recording proof:** run `squint index` on an actual screen recording; the
  states align with real UI changes and `ocr_text` captures visible text. (Synthetic
  screencast used if no real recording is available at build time.)

## 6. Environment
Windows 11, Python, `opencv-python`, `numpy`, `rapidocr-onnxruntime` (verified: installs
pip-only, correctly OCR'd a synthetic frame). Tests via `python -m pytest`. Commits on
`master`, TDD + frequent commits.
