# Squint v2 — Event Classification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Tag each manifest event as a `cut` (scene change) or `motion` (in-scene movement) with a confidence score, using grayscale-histogram similarity as a signal orthogonal to motion.

**Architecture:** Add `histogram_similarity` and `classify_events` to the existing `analyzer.py`; extend `EventRecord` with defaulted `type`/`confidence` fields (backward-compatible); compute a per-frame similarity list in the CLI loop and classify events after grouping. No changes to motion scoring, mapper, or source.

**Tech Stack:** Python 3, numpy (`np.histogram`, `np.corrcoef`), existing pytest suite. Builds on Squint v1 (master branch, 12 passing tests).

---

## File Structure

- Modify `squint/core/manifest.py` — `EventRecord` gains `type: str = "motion"`, `confidence: float = 0.0`.
- Modify `squint/core/analyzer.py` — add `histogram_similarity`, `classify_events`.
- Modify `squint/cli.py` — compute `similarities` in the loop, call `classify_events`, add `--cut-threshold`.
- Modify `SKILL.md` — event-type guidance.
- Modify `tests/test_manifest.py`, `tests/test_analyzer.py`, `tests/test_cli.py` — new tests appended.

Each change is small and additive; existing v1 functions and signatures are untouched.

---

### Task 1: Extend EventRecord (schema, backward-compatible)

**Files:**
- Modify: `squint/core/manifest.py` (the `EventRecord` dataclass)
- Test: `tests/test_manifest.py` (append one test)

- [ ] **Step 1: Write the failing test** — append to `tests/test_manifest.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_manifest.py::test_event_record_defaults_and_backward_compat -v`
Expected: FAIL with `TypeError` (unexpected keyword `type`) or `AttributeError` (`type`).

- [ ] **Step 3: Implement** — in `squint/core/manifest.py`, replace the `EventRecord` dataclass:

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

Leave `to_dict`/`from_dict` unchanged — they use `asdict`/`EventRecord(**e)`, so new manifests serialize the fields and old manifests load with defaults.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_manifest.py -v`
Expected: PASS (original roundtrip test + new test).

- [ ] **Step 5: Commit**

```
git add squint/core/manifest.py tests/test_manifest.py
git commit -m "feat: add type/confidence fields to EventRecord"
```

---

### Task 2: `histogram_similarity` (the discriminator signal)

**Files:**
- Modify: `squint/core/analyzer.py` (add function)
- Test: `tests/test_analyzer.py` (append tests)

- [ ] **Step 1: Write the failing tests** — append to `tests/test_analyzer.py`:

```python
def test_histogram_similarity_identical_is_one():
    f = np.arange(100, dtype=np.uint8).reshape(10, 10)
    assert analyzer.histogram_similarity(f, f) == 1.0


def test_histogram_similarity_black_vs_white_is_zero():
    black = np.zeros((8, 8), dtype=np.uint8)
    white = np.full((8, 8), 255, dtype=np.uint8)
    assert analyzer.histogram_similarity(black, white) == 0.0


def test_histogram_similarity_same_distribution_stays_high():
    # identical intensity distribution, different spatial layout -> high similarity
    # (orthogonal to motion: pixels moved, histogram unchanged)
    a = np.zeros((8, 8), dtype=np.uint8); a[:, :4] = 200
    b = np.zeros((8, 8), dtype=np.uint8); b[:, 4:] = 200
    assert analyzer.histogram_similarity(a, b) > 0.9
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_analyzer.py -k histogram -v`
Expected: FAIL with `AttributeError: module ... has no attribute 'histogram_similarity'`.

- [ ] **Step 3: Implement** — add to `squint/core/analyzer.py` (after `motion_score`):

```python
def histogram_similarity(prev_gray, cur_gray, bins: int = 64) -> float:
    a = np.asarray(prev_gray)
    b = np.asarray(cur_gray)
    if a.shape == b.shape and np.array_equal(a, b):
        return 1.0
    ha, _ = np.histogram(a, bins=bins, range=(0, 256))
    hb, _ = np.histogram(b, bins=bins, range=(0, 256))
    ha = ha.astype(np.float64)
    hb = hb.astype(np.float64)
    if ha.sum() > 0:
        ha /= ha.sum()
    if hb.sum() > 0:
        hb /= hb.sum()
    if ha.std() == 0 or hb.std() == 0:
        return 1.0 if np.array_equal(ha, hb) else 0.0
    corr = float(np.corrcoef(ha, hb)[0, 1])
    return max(0.0, corr)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_analyzer.py -v`
Expected: PASS (all analyzer tests, including the 3 new histogram tests).

- [ ] **Step 5: Commit**

```
git add squint/core/analyzer.py tests/test_analyzer.py
git commit -m "feat: add grayscale histogram_similarity"
```

---

### Task 3: `classify_events` (assign type + confidence)

**Files:**
- Modify: `squint/core/analyzer.py` (add function)
- Test: `tests/test_analyzer.py` (append tests)

- [ ] **Step 1: Write the failing tests** — append to `tests/test_analyzer.py`:

```python
def test_classify_events_cut_and_motion():
    events = [
        EventRecord(t_start=1.0, t_end=1.0, peak_frame=5, peak_score=0.16),
        EventRecord(t_start=2.2, t_end=2.4, peak_frame=11, peak_score=0.15),
    ]
    sims = [1.0] * 12
    sims[5] = 0.1    # low similarity at peak -> cut
    sims[11] = 0.9   # high similarity at peak -> motion
    out = analyzer.classify_events(events, sims, cut_threshold=0.5)
    assert out[0].type == "cut"
    assert out[0].confidence == 0.8     # (0.5 - 0.1) / 0.5
    assert out[1].type == "motion"
    assert out[1].confidence == 0.8     # (0.9 - 0.5) / 0.5


def test_classify_events_returns_same_objects():
    e = EventRecord(t_start=0.0, t_end=0.0, peak_frame=0, peak_score=0.2)
    out = analyzer.classify_events([e], [0.2], cut_threshold=0.5)
    assert out[0] is e
    assert e.type == "cut"
```

(`EventRecord` is already imported at the top of `tests/test_analyzer.py` from Task 4 of v1.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_analyzer.py -k classify -v`
Expected: FAIL with `AttributeError: ... has no attribute 'classify_events'`.

- [ ] **Step 3: Implement** — add to `squint/core/analyzer.py` (after `group_events`):

```python
def classify_events(events, similarities, cut_threshold: float = 0.5):
    for e in events:
        s = similarities[e.peak_frame]
        if s < cut_threshold:
            e.type = "cut"
            e.confidence = round((cut_threshold - s) / cut_threshold, 3) if cut_threshold > 0 else 1.0
        else:
            denom = 1.0 - cut_threshold
            e.type = "motion"
            e.confidence = round((s - cut_threshold) / denom, 3) if denom > 0 else 1.0
    return events
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_analyzer.py -v`
Expected: PASS (all analyzer tests).

- [ ] **Step 5: Commit**

```
git add squint/core/analyzer.py tests/test_analyzer.py
git commit -m "feat: classify events as cut or motion via histogram similarity"
```

---

### Task 4: Wire classification into the CLI

**Files:**
- Modify: `squint/cli.py` (`analyze` loop + `main` argparse)
- Test: `tests/test_cli.py` (append a test)

- [ ] **Step 1: Write the failing test** — append to `tests/test_cli.py`:

```python
def test_analyze_events_have_type(moving_square_video, tmp_path):
    out = tmp_path / "work_typed"
    manifest_path = analyze(
        str(moving_square_video), fps=5.0, cols=40,
        sensitivity=0.01, edge=False, out=str(out),
    )
    m = Manifest.load(manifest_path)
    assert len(m.events) >= 1
    for e in m.events:
        assert e.type in {"cut", "motion"}
        assert 0.0 <= e.confidence <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_cli.py::test_analyze_events_have_type -v`
Expected: FAIL — events default to `type="motion"`/`confidence=0.0` and the assertions on a *populated* type would still pass for type, but confidence stays 0.0 (valid). To make this a true red, FIRST confirm by running; if it already passes because defaults satisfy the assertions, that is acceptable — the test locks the contract. Proceed to implement the real classification regardless so confidence is actually computed.

> Note: this test guards the contract (every event has a valid type/confidence). The substantive behavior (correct cut/motion labels) is covered by the analyzer unit tests in Task 3 and the manual real-clip proof below.

- [ ] **Step 3: Implement** — edit `squint/cli.py`.

3a. Change the `analyze` signature to add `cut_threshold`:

```python
def analyze(video, fps=5.0, cols=120, sensitivity=0.06, edge=False, out=None, cut_threshold=0.5):
```

3b. Before the frame loop, initialize a similarities list. Replace this block:

```python
    records = []
    prev_gray = None
    for idx, t, bgr in source.iter_frames(str(video), fps):
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        score = 0.0 if prev_gray is None else analyzer.motion_score(prev_gray, gray)
        prev_gray = gray
```

with:

```python
    records = []
    similarities = []
    prev_gray = None
    for idx, t, bgr in source.iter_frames(str(video), fps):
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        score = 0.0 if prev_gray is None else analyzer.motion_score(prev_gray, gray)
        sim = 1.0 if prev_gray is None else analyzer.histogram_similarity(prev_gray, gray)
        prev_gray = gray
        similarities.append(sim)
```

3c. After `events = analyzer.group_events(scores, times, flags)`, add the classify call:

```python
    events = analyzer.group_events(scores, times, flags)
    events = analyzer.classify_events(events, similarities, cut_threshold)
```

3d. In `main`, add the `--cut-threshold` argument to the `analyze` subparser (after the `--out` line):

```python
    a.add_argument("--cut-threshold", type=float, default=0.5,
                   help="histogram-similarity below which an event is a scene cut (0..1)")
```

3e. In `main`, pass it through in the `analyze` dispatch — replace:

```python
        path = analyze(args.video, fps=args.fps, cols=args.cols,
                       sensitivity=args.sensitivity, edge=args.edge, out=args.out)
```

with:

```python
        path = analyze(args.video, fps=args.fps, cols=args.cols,
                       sensitivity=args.sensitivity, edge=args.edge, out=args.out,
                       cut_threshold=args.cut_threshold)
```

- [ ] **Step 4: Run the full suite and the help**

Run: `python -m pytest -q`
Expected: all PASS (15 → now 16 tests).
Run: `python -m squint.cli analyze --help`
Expected: help text now lists `--cut-threshold`.

- [ ] **Step 5: Commit**

```
git add squint/cli.py tests/test_cli.py
git commit -m "feat: classify events in analyze CLI with --cut-threshold"
```

---

### Task 5: Update SKILL.md guidance

**Files:**
- Modify: `SKILL.md`

- [ ] **Step 1: Edit SKILL.md.**

5a. Replace the `--cols 120` analyze command line in step 1 to mention the new flag. Replace:

```
   `python -m squint.cli analyze <video> --fps 5 --cols 120`
   (add `--edge` for scenes where structure/outlines matter more than shading; lower
   `--sensitivity` to catch subtler motion, raise it to ignore noise.)
```

with:

```
   `python -m squint.cli analyze <video> --fps 5 --cols 120`
   (add `--edge` for scenes where structure/outlines matter more than shading; lower
   `--sensitivity` to catch subtler motion, raise it to ignore noise; `--cut-threshold`
   tunes how aggressively scene cuts are detected.)
```

5b. Replace the step-2 manifest description:

```
2. **Read the manifest.** `Read` `manifest.json`. It gives you `duration`, the sampled
   `frames` (each with `t`, motion `score`, `event` flag, and paths to its ASCII + PNG),
   and `events` — the contiguous high-motion segments worth your attention. Do NOT read
   every frame; start from `events`.
```

with:

```
2. **Read the manifest.** `Read` `manifest.json`. It gives you `duration`, the sampled
   `frames` (each with `t`, motion `score`, `event` flag, and paths to its ASCII + PNG),
   and `events` — the contiguous high-motion segments worth your attention. Each event has
   a `type` and `confidence`:
   - `cut` — a scene change. The ASCII skim is usually enough; escalate to the PNG only if
     the user asks about that scene's content.
   - `motion` — something moved within a continuous scene. Escalate these for "did X
     happen?" / activity questions.
   Do NOT read every frame; start from `events`, prioritizing `motion` events for activity
   questions and using `cut` events to segment the video into scenes.
```

- [ ] **Step 2: Sanity-check the frontmatter is intact**

Run: `python -c "t=open('SKILL.md',encoding='utf-8').read(); assert t.startswith('---'); assert 'cut' in t and 'motion' in t; print('ok')"`
Expected: prints `ok`.

- [ ] **Step 3: Commit**

```
git add SKILL.md
git commit -m "docs: document event types in SKILL.md"
```

---

## Self-Review

- **Spec coverage:**
  - `histogram_similarity` (§4.1) → Task 2. ✓
  - `classify_events` (§4.1) → Task 3. ✓
  - `EventRecord.type`/`confidence` defaulted + backward-compat (§4.2, §7) → Task 1. ✓
  - CLI `similarities` + `classify_events` call + `--cut-threshold` (§4.3) → Task 4. ✓
  - SKILL.md event-type guidance (§4.4) → Task 5. ✓
  - Data flow (§5) → Task 4 wiring matches exactly (sim computed pre-`prev_gray` reassign, `sim_0=1.0`). ✓
  - Tests (§7): hist identical/black-white/same-distribution → Task 2; classify cut/motion → Task 3; CLI type/confidence validity → Task 4; backward-compat load → Task 1; real-clip proof → manual note below. ✓
  - Out-of-scope items (§6): no color histograms, no flow, no new sub-types, no v1 changes — none introduced. ✓
- **Placeholder scan:** no TBD/TODO; every code/edit step shows full code. ✓
- **Type consistency:** `histogram_similarity(prev_gray, cur_gray, bins=64)`, `classify_events(events, similarities, cut_threshold=0.5)`, and `EventRecord(..., type="motion", confidence=0.0)` are used identically across Tasks 1–4. `peak_frame` indexes the per-frame `similarities` list, consistent with v1's established `idx == list position` (CLI appends one `similarities` entry per sampled frame in order). ✓

**Manual real-clip proof (run after Task 4, not a committed test):**
```
python -m squint.cli analyze "C:/Users/RushiHiray/Pictures/whisper_crop.mp4" --fps 5 --cols 100 --out work_v2check.squint
```
Then inspect `work_v2check.squint/manifest.json` events — the two known scene cuts (~t=1.0
and ~t=2.2) should have `type="cut"`. The `.squint` suffix means the dir is already
git-ignored; remove it afterward with `rm -rf work_v2check.squint`.
