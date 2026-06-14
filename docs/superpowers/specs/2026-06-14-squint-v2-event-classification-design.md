# Squint v2 — Event Classification Design Spec

**Date:** 2026-06-14
**Status:** Approved (pending spec review)
**Builds on:** Squint v1 (see `2026-06-14-squint-design.md`)

## 1. Problem

In v1 the analyzer scores motion as the mean absolute pixel difference between
consecutive grayscale frames (`motion_score`), thresholds it (`flag_events`), and groups
contiguous flagged frames into `EventRecord`s (`group_events`). A hard **scene cut** and
real **in-scene motion** both produce a high `motion_score`, so events are untyped and
Claude cannot tell "the scene changed" from "something moved within the scene."

The v1 demo on `whisper_crop.mp4` confirmed this: both flagged events were scene cuts, but
nothing in the manifest said so.

## 2. Goal

Tag each `EventRecord` with a `type` (`"cut"` or `"motion"`) and a `confidence` in `[0,1]`,
using a signal orthogonal to `motion_score`, so Claude escalates more intelligently
(skim is usually enough for a cut; in-scene motion is what matters for "did X happen?").

## 3. Approach: histogram-similarity discriminator

A cut swaps the entire scene, so the **global luminance histogram** between consecutive
frames decorrelates sharply. In-scene motion keeps a similar global histogram (same
lighting/palette) even as pixels move locally. Therefore, at an event's transition frame:

- **low** histogram similarity → `cut`
- **high** histogram similarity (but high motion) → `motion`

This signal is orthogonal to `motion_score`, stays in the existing numpy/grayscale
pipeline, and is deterministic and unit-testable without OpenCV.

Rejected alternatives:
- **Temporal-shape** (cut = one-frame spike, motion = sustained): fragile at 5 fps sampling;
  quick gestures fool it.
- **Optical flow**: most accurate but far heavier per frame; overkill for v2.

## 4. Components

### 4.1 `squint/core/analyzer.py` — new functions

**`histogram_similarity(prev_gray, cur_gray, bins=64) -> float`**
- Compute a `bins`-bucket histogram of each grayscale frame over `[0, 256)`.
- Normalize each histogram to sum 1.
- Return the Pearson correlation of the two histogram vectors, with negative values
  clamped to `0.0`. `1.0` = identical distribution; `~0.0` = unrelated.
- Pure numpy (`np.histogram`, `np.corrcoef`). If either histogram has zero variance
  (e.g. a uniform frame), correlation is undefined — return `1.0` when both frames are
  bit-identical, else `0.0`.

**`classify_events(events, similarities, cut_threshold=0.5) -> list[EventRecord]`**
- `similarities` is a per-frame list where `similarities[i]` is the histogram similarity
  between frame `i` and frame `i-1` (`similarities[0] == 1.0`).
- For each event, read `s = similarities[event.peak_frame]` (the transition that caused the
  spike).
- If `s < cut_threshold`: `type = "cut"`, `confidence = round((cut_threshold - s) / cut_threshold, 3)`.
- Else: `type = "motion"`, `confidence = round((s - cut_threshold) / (1 - cut_threshold), 3)`.
- Mutates and returns the same `EventRecord` objects.

### 4.2 `squint/core/manifest.py` — schema change

`EventRecord` gains two fields **with defaults** (backward-compatible: old manifests
lacking these keys still load via `EventRecord(**e)`):

```python
@dataclass
class EventRecord:
    t_start: float
    t_end: float
    peak_frame: int
    peak_score: float
    type: str = "motion"
    confidence: float = 0.0
```

`to_dict`/`from_dict` already use `asdict`/`**e`, so they pick up the new fields with no
change. A short test confirms an old-style event dict (without `type`/`confidence`) still
loads.

### 4.3 `squint/cli.py` — wiring

In the existing per-frame loop, alongside `motion_score`, compute the histogram similarity
to the previous frame and collect it into a parallel `similarities` list
(`similarities[0] = 1.0`). After `group_events`, call
`analyzer.classify_events(events, similarities, cut_threshold)` before building the
manifest. Add a `--cut-threshold` CLI flag (default `0.5`) on the `analyze` subcommand.

### 4.4 `SKILL.md` — guidance update

Document that each event now carries `type`:
- `cut` — a scene change; the ASCII skim is usually enough, escalate only if the user
  asks about that scene's content.
- `motion` — something moved within a continuous scene; escalate these for "did X happen?"
  / activity questions.

## 5. Data flow

```
per frame i:  gray_i
  score_i      = motion_score(gray_{i-1}, gray_i)       # existing
  sim_i        = histogram_similarity(gray_{i-1}, gray_i)  # NEW (sim_0 = 1.0)
flags          = flag_events(scores, sensitivity)        # existing
events         = group_events(scores, times, flags)      # existing (untyped)
events         = classify_events(events, sims, cut_threshold)  # NEW (sets type/confidence)
manifest.events = events
```

## 6. Scope

### In
- `histogram_similarity` and `classify_events` in `analyzer.py`.
- `EventRecord.type` / `EventRecord.confidence` (defaulted) in `manifest.py`.
- `similarities` computation + `classify_events` call + `--cut-threshold` flag in `cli.py`.
- `SKILL.md` event-type guidance.
- Unit tests for both new functions + a CLI assertion that events carry a valid `type`.

### Out (YAGNI / future)
- Color/HSV histograms (we stay on the grayscale pipeline; note as future).
- Optical-flow or temporal-shape classifiers.
- Sub-types beyond `cut`/`motion` (e.g. `fade`, `pan`).
- Any change to v1 motion scoring, mapper, source, or the manifest frame schema.

## 7. Testing & success criteria

- **`histogram_similarity`:**
  - identical frames → `1.0`.
  - black frame vs white frame → `0.0`.
  - two different frames with the *same* global intensity distribution (e.g. a block
    shifted in position) → stays high (well above `0.5`), demonstrating orthogonality to
    `motion_score`.
- **`classify_events`:**
  - an event whose `peak_frame` similarity is below threshold → `type="cut"` with
    `confidence` in `(0,1]`.
  - an event whose `peak_frame` similarity is above threshold → `type="motion"`.
- **CLI:** every event in the produced manifest has `type in {"cut","motion"}` and
  `0.0 <= confidence <= 1.0`.
- **Backward compat:** an `EventRecord` dict without `type`/`confidence` loads with the
  defaults.
- **Real-clip proof (manual):** re-run `analyze` on `whisper_crop.mp4`; the two known scene
  cuts are labeled `type="cut"`.

## 8. Environment
Windows 11, Python, numpy + opencv-python already installed. Tests via `python -m pytest`.
All commits on the existing `master` branch following v1's TDD + frequent-commit style.
