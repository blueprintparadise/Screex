# Screex-on-Sharingan (ACTREAL) Study Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible harness that scores Screex's "index → Claude → typed actions" pipeline against a raw-frames baseline on a reconstructed ACTREAL set, using Sharingan's Precision/Recall metric.

**Architecture:** A standalone `study/` package imports `screex` as a library. Pure logic (metric matching, dataset filter, frame windowing, prompt assembly, report aggregation) is unit-tested with injected fakes; network/LLM/HF boundaries are thin wrappers behind protocols so tests never hit the network. Two arms (DF-equivalent baseline, Screex) emit `Action` sequences; `metric.py` reimplements Sharingan's greedy-chronological BERT-cosine matching; `report.py` aggregates.

**Tech Stack:** Python 3.9+, screex (local), opencv (via screex), anthropic SDK (Claude), sentence-transformers (BERT embeddings), datasets/huggingface_hub (GUI-World), pytest.

---

## File Structure

```
study/
  __init__.py
  config.py          # constants: OPS, model id, fps, window sizes, threshold
  actions.py         # Action dataclass + JSON load/save
  embedder.py        # Embedder protocol + BertEmbedder + cosine
  metric.py          # Sharingan P/R (greedy chronological match, 2 levels)
  claude_client.py   # ClaudeClient protocol + AnthropicClient wrapper
  frames.py          # frame sampling (2fps) + windowing (10/overlap 5)
  prompts.py         # prompt builders for DF arm, Screex arm, GT generator
  data_prep.py       # GUI-World -> ACTREAL filter -> manifest.json
  gt_generate.py     # Claude-propose GT -> gt/*.json (human-edited after)
  arms/__init__.py
  arms/df.py         # raw-frames -> Claude -> actions
  arms/screex_arm.py # screex index -> Claude -> actions
  run.py             # orchestrate arms over manifest, cache outputs
  report.py          # aggregate scores + token cost; build tables
  requirements.txt   # eval-only deps
  README.md          # how to reproduce
  manifest.json      # frozen (produced by data_prep)
  gt/                # frozen, human-verified GT (committed)
  cache/             # frames, indexes, model outputs (gitignored)
tests/study/
  test_actions.py
  test_metric.py
  test_embedder.py
  test_frames.py
  test_data_prep.py
  test_prompts.py
  test_report.py
  test_arms.py
```

Shared types (defined in early tasks, referenced later):
- `Action(op: str, details: str, context: str)` — `op` ∈ `OPS`.
- `OPS = ("click", "select", "scroll", "drag", "type")`.
- `Embedder` protocol: `embed(texts: list[str]) -> list[list[float]]`.
- `ClaudeClient` protocol: `complete(system: str, user: str, images: list[str] | None = None) -> tuple[str, dict]` (returns text, usage dict `{"input_tokens", "output_tokens"}`).
- `Scores(op_precision, op_recall, all_precision, all_recall)`.

---

## Task 0: Scaffold the study package

**Files:**
- Create: `study/__init__.py`
- Create: `study/config.py`
- Create: `study/requirements.txt`
- Create: `tests/study/__init__.py`
- Create: `study/.gitignore`

- [ ] **Step 1: Create package files**

`study/__init__.py`:
```python
"""Screex-on-Sharingan benchmark study harness (not part of the shipped package)."""
```

`study/config.py`:
```python
"""Central knobs for the study. Recorded in reports for reproducibility."""

OPS = ("click", "select", "scroll", "drag", "type")

# Base model used for BOTH arms (controlled comparison). Fixed and recorded.
CLAUDE_MODEL = "claude-opus-4-8"
# GT generator deliberately uses a config distinct from the arms (different prompt;
# dense-frame input) to avoid biasing GT toward either arm.
GT_MODEL = "claude-opus-4-8"

# Sharingan ACTREAL sampling + DF windowing.
SAMPLE_FPS = 2.0
WINDOW_SIZE = 10
WINDOW_OVERLAP = 5

# Sharingan semantic-match threshold for details/context.
MATCH_THRESHOLD = 0.70
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ACTREAL reconstruction filters.
ACTREAL_CATEGORIES = ("Software", "Website", "Multi")
ACTREAL_MIN_ACTIONS = 6
ACTREAL_MAX_ACTIONS = 10
ACTREAL_MIN_UNIQUE_TYPES = 3
ACTREAL_TARGET_COUNT = 41
```

`study/requirements.txt`:
```
anthropic>=0.40
sentence-transformers>=2.2
datasets>=2.0
huggingface_hub>=0.20
```

`study/.gitignore`:
```
cache/
```

`tests/study/__init__.py`:
```python
```

- [ ] **Step 2: Verify the package imports**

Run: `python -c "import study, study.config; print(study.config.OPS)"`
Expected: `('click', 'select', 'scroll', 'drag', 'type')`

- [ ] **Step 3: Commit**

```bash
git add study/__init__.py study/config.py study/requirements.txt study/.gitignore tests/study/__init__.py
git commit -m "chore(study): scaffold benchmark harness package"
```

---

## Task 1: Action model + JSON I/O

**Files:**
- Create: `study/actions.py`
- Test: `tests/study/test_actions.py`

- [ ] **Step 1: Write the failing test**

`tests/study/test_actions.py`:
```python
import json
from study.actions import Action, load_actions, save_actions


def test_action_roundtrip_dict():
    a = Action(op="click", details="Styles dropdown menu", context="Microsoft Word")
    assert Action.from_dict(a.to_dict()) == a


def test_action_rejects_unknown_op():
    import pytest
    with pytest.raises(ValueError):
        Action(op="hover", details="x", context="y")


def test_save_and_load_actions(tmp_path):
    seq = [
        Action("click", "File menu", "VSCode"),
        Action("type", "hello world", "VSCode"),
    ]
    p = tmp_path / "seq.json"
    save_actions(p, seq)
    assert load_actions(p) == seq
    # On-disk shape is a JSON list of {op, details, context}.
    raw = json.loads(p.read_text(encoding="utf-8"))
    assert raw[0] == {"op": "click", "details": "File menu", "context": "VSCode"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/study/test_actions.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'study.actions'`

- [ ] **Step 3: Write minimal implementation**

`study/actions.py`:
```python
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from study.config import OPS


@dataclass(frozen=True)
class Action:
    op: str
    details: str
    context: str

    def __post_init__(self):
        if self.op not in OPS:
            raise ValueError(f"unknown op {self.op!r}; expected one of {OPS}")

    def to_dict(self) -> dict:
        return {"op": self.op, "details": self.details, "context": self.context}

    @classmethod
    def from_dict(cls, d: dict) -> "Action":
        return cls(op=d["op"], details=d["details"], context=d["context"])


def save_actions(path, actions) -> None:
    Path(path).write_text(
        json.dumps([a.to_dict() for a in actions], indent=2), encoding="utf-8"
    )


def load_actions(path) -> list:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return [Action.from_dict(d) for d in raw]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/study/test_actions.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add study/actions.py tests/study/test_actions.py
git commit -m "feat(study): Action model with validated ops and JSON I/O"
```

---

## Task 2: Embedder protocol + cosine

**Files:**
- Create: `study/embedder.py`
- Test: `tests/study/test_embedder.py`

- [ ] **Step 1: Write the failing test**

`tests/study/test_embedder.py`:
```python
import math
from study.embedder import cosine, DictEmbedder


def test_cosine_identical_is_one():
    assert math.isclose(cosine([1.0, 0.0], [1.0, 0.0]), 1.0, abs_tol=1e-9)


def test_cosine_orthogonal_is_zero():
    assert math.isclose(cosine([1.0, 0.0], [0.0, 1.0]), 0.0, abs_tol=1e-9)


def test_cosine_zero_vector_is_zero():
    assert cosine([0.0, 0.0], [1.0, 1.0]) == 0.0


def test_dict_embedder_returns_configured_vectors():
    emb = DictEmbedder({"a": [1.0, 0.0], "b": [0.0, 1.0]})
    assert emb.embed(["a", "b"]) == [[1.0, 0.0], [0.0, 1.0]]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/study/test_embedder.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'study.embedder'`

- [ ] **Step 3: Write minimal implementation**

`study/embedder.py`:
```python
from __future__ import annotations

import math
from typing import Protocol


class Embedder(Protocol):
    def embed(self, texts: list) -> list: ...


def cosine(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


class DictEmbedder:
    """Test embedder: returns pre-set vectors for known strings."""

    def __init__(self, table: dict):
        self._table = table

    def embed(self, texts: list) -> list:
        return [self._table[t] for t in texts]


class BertEmbedder:
    """Real embedder backed by sentence-transformers. Lazy-loaded."""

    def __init__(self, model_name=None):
        from study.config import EMBED_MODEL
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name or EMBED_MODEL)

    def embed(self, texts: list) -> list:
        return [v.tolist() for v in self._model.encode(list(texts))]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/study/test_embedder.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add study/embedder.py tests/study/test_embedder.py
git commit -m "feat(study): Embedder protocol, cosine, and test/real implementations"
```

---

## Task 3: Sharingan metric (greedy chronological match, 2 levels)

**Files:**
- Create: `study/metric.py`
- Test: `tests/study/test_metric.py`

- [ ] **Step 1: Write the failing test**

`tests/study/test_metric.py`:
```python
import math
from study.actions import Action
from study.embedder import DictEmbedder
from study.metric import score


# Embedding table: equal strings -> identical vectors (cosine 1.0);
# "near" pairs -> cosine ~0.8 (>=0.7); unrelated -> orthogonal (0.0).
TABLE = {
    "File menu": [1.0, 0.0, 0.0],
    "the File menu": [0.8, 0.6, 0.0],     # cosine vs "File menu" = 0.8
    "Quantum tunneling": [0.0, 0.0, 1.0],  # unrelated
    "VSCode": [0.0, 1.0, 0.0],
    "the VSCode editor": [0.0, 1.0, 0.0],  # identical context vec
}


def test_perfect_match_scores_one():
    gt = [Action("click", "File menu", "VSCode")]
    pred = [Action("click", "File menu", "VSCode")]
    s = score(gt, pred, DictEmbedder(TABLE), threshold=0.7)
    assert s.op_precision == 1.0 and s.op_recall == 1.0
    assert s.all_precision == 1.0 and s.all_recall == 1.0


def test_operation_level_ignores_details():
    # Same op, unrelated details -> Operation matches, All does not.
    gt = [Action("click", "File menu", "VSCode")]
    pred = [Action("click", "Quantum tunneling", "VSCode")]
    s = score(gt, pred, DictEmbedder(TABLE), threshold=0.7)
    assert s.op_recall == 1.0          # op matched
    assert s.all_recall == 0.0         # details cosine 0 < 0.7


def test_near_details_above_threshold_matches_all():
    gt = [Action("click", "File menu", "VSCode")]
    pred = [Action("click", "the File menu", "the VSCode editor")]
    s = score(gt, pred, DictEmbedder(TABLE), threshold=0.7)
    assert s.all_recall == 1.0         # 0.8 >= 0.7 on details, context identical


def test_extra_prediction_lowers_precision():
    gt = [Action("click", "File menu", "VSCode")]
    pred = [
        Action("click", "File menu", "VSCode"),
        Action("scroll", "File menu", "VSCode"),  # spurious
    ]
    s = score(gt, pred, DictEmbedder(TABLE), threshold=0.7)
    assert s.op_recall == 1.0
    assert math.isclose(s.op_precision, 0.5)


def test_each_prediction_matches_at_most_once():
    gt = [Action("click", "File menu", "VSCode"), Action("click", "File menu", "VSCode")]
    pred = [Action("click", "File menu", "VSCode")]  # only one available
    s = score(gt, pred, DictEmbedder(TABLE), threshold=0.7)
    assert math.isclose(s.op_recall, 0.5)   # only 1 of 2 GT matched
    assert s.op_precision == 1.0


def test_empty_prediction_is_zero_not_crash():
    gt = [Action("click", "File menu", "VSCode")]
    s = score(gt, [], DictEmbedder(TABLE), threshold=0.7)
    assert s.op_precision == 0.0 and s.op_recall == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/study/test_metric.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'study.metric'`

- [ ] **Step 3: Write minimal implementation**

`study/metric.py`:
```python
from __future__ import annotations

from dataclasses import dataclass

from study.embedder import cosine


@dataclass
class Scores:
    op_precision: float
    op_recall: float
    all_precision: float
    all_recall: float


def _embed_all(actions, embedder):
    """Return parallel lists of (details_vec, context_vec) for each action."""
    if not actions:
        return [], []
    details = embedder.embed([a.details for a in actions])
    contexts = embedder.embed([a.context for a in actions])
    return details, contexts


def _count_matches(gt, pred, gt_vecs, pred_vecs, threshold, level):
    """Greedy chronological match. For each GT action in order, take the first
    still-unmatched predicted action that aligns under `level`."""
    gt_d, gt_c = gt_vecs
    pred_d, pred_c = pred_vecs
    used = [False] * len(pred)
    matched = 0
    for i, g in enumerate(gt):
        for j, p in enumerate(pred):
            if used[j]:
                continue
            if g.op != p.op:
                continue
            if level == "all":
                if cosine(gt_d[i], pred_d[j]) < threshold:
                    continue
                if cosine(gt_c[i], pred_c[j]) < threshold:
                    continue
            used[j] = True
            matched += 1
            break
    return matched


def _pr(matched, n_gt, n_pred):
    precision = matched / n_pred if n_pred else 0.0
    recall = matched / n_gt if n_gt else 0.0
    return precision, recall


def score(gt, pred, embedder, threshold=0.70) -> Scores:
    gt_vecs = _embed_all(gt, embedder)
    pred_vecs = _embed_all(pred, embedder)
    op_matched = _count_matches(gt, pred, gt_vecs, pred_vecs, threshold, "op")
    all_matched = _count_matches(gt, pred, gt_vecs, pred_vecs, threshold, "all")
    op_p, op_r = _pr(op_matched, len(gt), len(pred))
    all_p, all_r = _pr(all_matched, len(gt), len(pred))
    return Scores(op_p, op_r, all_p, all_r)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/study/test_metric.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add study/metric.py tests/study/test_metric.py
git commit -m "feat(study): Sharingan P/R metric with greedy chronological matching"
```

---

## Task 4: ACTREAL reconstruction filter

**Files:**
- Create: `study/data_prep.py`
- Test: `tests/study/test_data_prep.py`

The download side (GUI-World from HF) is a thin wrapper; the **filter predicate** is pure and is what we test.

- [ ] **Step 1: Write the failing test**

`tests/study/test_data_prep.py`:
```python
from study.data_prep import passes_actreal_filter, VideoMeta


def _meta(**kw):
    base = dict(
        video_id="v1", category="Software", n_actions=8,
        action_types=("click", "type", "scroll"),
        has_hover=False, has_noninput_key=False,
    )
    base.update(kw)
    return VideoMeta(**base)


def test_accepts_canonical_video():
    assert passes_actreal_filter(_meta()) is True


def test_rejects_wrong_category():
    assert passes_actreal_filter(_meta(category="Mobile")) is False


def test_rejects_too_few_actions():
    assert passes_actreal_filter(_meta(n_actions=5)) is False


def test_rejects_too_many_actions():
    assert passes_actreal_filter(_meta(n_actions=11)) is False


def test_rejects_too_few_unique_types():
    assert passes_actreal_filter(_meta(action_types=("click", "click"))) is False


def test_rejects_hover():
    assert passes_actreal_filter(_meta(has_hover=True)) is False


def test_rejects_noninput_keyboard():
    assert passes_actreal_filter(_meta(has_noninput_key=True)) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/study/test_data_prep.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'study.data_prep'`

- [ ] **Step 3: Write minimal implementation**

`study/data_prep.py`:
```python
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

from study.config import (
    ACTREAL_CATEGORIES,
    ACTREAL_MAX_ACTIONS,
    ACTREAL_MIN_ACTIONS,
    ACTREAL_MIN_UNIQUE_TYPES,
)


@dataclass
class VideoMeta:
    video_id: str
    category: str
    n_actions: int
    action_types: tuple
    has_hover: bool
    has_noninput_key: bool


def passes_actreal_filter(m: VideoMeta) -> bool:
    if m.category not in ACTREAL_CATEGORIES:
        return False
    if not (ACTREAL_MIN_ACTIONS <= m.n_actions <= ACTREAL_MAX_ACTIONS):
        return False
    if len(set(m.action_types)) < ACTREAL_MIN_UNIQUE_TYPES:
        return False
    if m.has_hover or m.has_noninput_key:
        return False
    return True


def write_manifest(metas, out_path) -> None:
    """Persist the frozen, filtered ACTREAL set."""
    kept = [asdict(m) for m in metas if passes_actreal_filter(m)]
    Path(out_path).write_text(json.dumps(kept, indent=2), encoding="utf-8")


def build_metas_from_guiworld(records):
    """Map raw GUI-World records (dicts) to VideoMeta. `records` is any iterable of
    dicts with keys: id, category, actions (list of {type, ...}). Adapt field names
    here once the exact HF schema is confirmed at runtime."""
    metas = []
    for r in records:
        types = tuple(a["type"] for a in r.get("actions", []))
        metas.append(VideoMeta(
            video_id=r["id"],
            category=r.get("category", ""),
            n_actions=len(types),
            action_types=types,
            has_hover=any(t == "hover" for t in types),
            has_noninput_key=bool(r.get("has_noninput_key", False)),
        ))
    return metas
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/study/test_data_prep.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add study/data_prep.py tests/study/test_data_prep.py
git commit -m "feat(study): ACTREAL reconstruction filter + manifest writer"
```

- [ ] **Step 6: Manual data pull (runtime, not unit-tested)**

After the HF GUI-World schema is confirmed interactively, append a `main()` to `data_prep.py` that downloads GUI-World, maps records via `build_metas_from_guiworld`, calls `write_manifest(..., "study/manifest.json")`, and caps to `ACTREAL_TARGET_COUNT`. Run it, eyeball `study/manifest.json`, then commit the frozen manifest:

```bash
python -m study.data_prep   # writes study/manifest.json
git add study/manifest.json
git commit -m "data(study): freeze reconstructed ACTREAL manifest"
```

---

## Task 5: Frame sampling + windowing

**Files:**
- Create: `study/frames.py`
- Test: `tests/study/test_frames.py`

The pure logic is the windowing. Frame extraction reuses `screex.core.source.iter_frames`.

- [ ] **Step 1: Write the failing test**

`tests/study/test_frames.py`:
```python
from study.frames import windows


def test_windows_overlap():
    items = list(range(12))  # 0..11
    w = windows(items, size=10, overlap=5)
    assert w[0] == list(range(0, 10))
    assert w[1] == list(range(5, 12))   # step = size - overlap = 5


def test_windows_shorter_than_size_returns_single():
    items = [1, 2, 3]
    assert windows(items, size=10, overlap=5) == [[1, 2, 3]]


def test_windows_exact_multiple_no_empty_tail():
    items = list(range(10))
    w = windows(items, size=10, overlap=5)
    assert w == [list(range(10))]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/study/test_frames.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'study.frames'`

- [ ] **Step 3: Write minimal implementation**

`study/frames.py`:
```python
from __future__ import annotations

from pathlib import Path

from study.config import SAMPLE_FPS, WINDOW_OVERLAP, WINDOW_SIZE


def windows(items, size=WINDOW_SIZE, overlap=WINDOW_OVERLAP):
    """Sliding windows of `size` with `overlap` shared items (step = size - overlap)."""
    if size <= overlap:
        raise ValueError("size must exceed overlap")
    items = list(items)
    if len(items) <= size:
        return [items]
    step = size - overlap
    out = []
    i = 0
    while i < len(items):
        chunk = items[i:i + size]
        out.append(chunk)
        if i + size >= len(items):
            break
        i += step
    return out


def sample_frame_paths(video_path, out_dir, fps=SAMPLE_FPS):
    """Write sampled frames at `fps` to out_dir; return ordered list of (t, path).
    Reuses screex's frame iterator + opencv writer."""
    import cv2

    from screex.core import source

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = []
    for i, fr in enumerate(source.iter_frames(str(video_path), fps)):
        p = out_dir / f"{i:05d}.png"
        cv2.imwrite(str(p), fr.bgr if hasattr(fr, "bgr") else fr.frame_bgr)
        t = getattr(fr, "t", getattr(fr, "t_end", i / fps))
        result.append((t, str(p)))
    return result
```

> Note: confirm the field name on `source.iter_frames` items at integration time (`.bgr`/`.frame_bgr`, `.t`/`.t_end`); the getattr fallbacks above cover the observed `frame_bgr`/`t_end` shape used in `screex/cli.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/study/test_frames.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add study/frames.py tests/study/test_frames.py
git commit -m "feat(study): frame sampling + sliding-window helper"
```

---

## Task 6: Claude client wrapper

**Files:**
- Create: `study/claude_client.py`
- Test: `tests/study/test_arms.py` (shared fake added here, used in Tasks 7–8)

- [ ] **Step 1: Write the failing test**

`tests/study/test_arms.py`:
```python
from study.claude_client import FakeClient


def test_fake_client_returns_scripted_text_and_usage():
    c = FakeClient(responses=["hello"])
    text, usage = c.complete(system="s", user="u")
    assert text == "hello"
    assert usage == {"input_tokens": 0, "output_tokens": 0}


def test_fake_client_records_calls():
    c = FakeClient(responses=["a", "b"])
    c.complete(system="s1", user="u1")
    c.complete(system="s2", user="u2", images=["img.png"])
    assert [call["user"] for call in c.calls] == ["u1", "u2"]
    assert c.calls[1]["images"] == ["img.png"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/study/test_arms.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'study.claude_client'`

- [ ] **Step 3: Write minimal implementation**

`study/claude_client.py`:
```python
from __future__ import annotations

import base64
from pathlib import Path
from typing import Protocol


class ClaudeClient(Protocol):
    def complete(self, system: str, user: str, images=None) -> tuple: ...


class FakeClient:
    """Deterministic client for tests. Pops scripted responses in order."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def complete(self, system: str, user: str, images=None) -> tuple:
        self.calls.append({"system": system, "user": user, "images": images})
        text = self._responses.pop(0)
        return text, {"input_tokens": 0, "output_tokens": 0}


class AnthropicClient:
    """Real client. Sends optional images as base64 image blocks."""

    def __init__(self, model):
        import anthropic

        self._client = anthropic.Anthropic()
        self._model = model

    def complete(self, system: str, user: str, images=None) -> tuple:
        content = []
        for img in images or []:
            data = base64.standard_b64encode(Path(img).read_bytes()).decode()
            media = "image/png" if str(img).endswith(".png") else "image/jpeg"
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": media, "data": data},
            })
        content.append({"type": "text", "text": user})
        msg = self._client.messages.create(
            model=self._model,
            max_tokens=4096,
            system=system,
            messages=[{"role": "user", "content": content}],
        )
        text = "".join(b.text for b in msg.content if b.type == "text")
        usage = {"input_tokens": msg.usage.input_tokens,
                 "output_tokens": msg.usage.output_tokens}
        return text, usage
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/study/test_arms.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add study/claude_client.py tests/study/test_arms.py
git commit -m "feat(study): Claude client protocol with fake + Anthropic impl"
```

---

## Task 7: Prompts + action parsing

**Files:**
- Create: `study/prompts.py`
- Test: `tests/study/test_prompts.py`

- [ ] **Step 1: Write the failing test**

`tests/study/test_prompts.py`:
```python
from study.actions import Action
from study.prompts import parse_actions, df_user_prompt, screex_user_prompt


def test_parse_actions_from_json_block():
    text = '''Here you go:
```json
[{"op": "click", "details": "File menu", "context": "VSCode"},
 {"op": "type", "details": "hi", "context": "VSCode"}]
```'''
    actions = parse_actions(text)
    assert actions == [
        Action("click", "File menu", "VSCode"),
        Action("type", "hi", "VSCode"),
    ]


def test_parse_actions_bare_array():
    text = '[{"op": "scroll", "details": "down", "context": "Chrome"}]'
    assert parse_actions(text) == [Action("scroll", "down", "Chrome")]


def test_parse_actions_skips_invalid_ops():
    text = '[{"op": "hover", "details": "x", "context": "y"}, {"op": "click", "details": "a", "context": "b"}]'
    # Unknown ops are dropped, not crashed on.
    assert parse_actions(text) == [Action("click", "a", "b")]


def test_df_prompt_mentions_five_ops_and_json():
    p = df_user_prompt(n_frames=10)
    for op in ("click", "select", "scroll", "drag", "type"):
        assert op in p
    assert "json" in p.lower()


def test_screex_prompt_includes_index_json():
    p = screex_user_prompt('{"states": []}')
    assert '{"states": []}' in p
    assert "json" in p.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/study/test_prompts.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'study.prompts'`

- [ ] **Step 3: Write minimal implementation**

`study/prompts.py`:
```python
from __future__ import annotations

import json
import re

from study.actions import Action
from study.config import OPS

_OPS_LINE = ", ".join(OPS)

_SCHEMA = (
    'Return ONLY a JSON array. Each element is '
    '{"op": <one of: ' + _OPS_LINE + '>, '
    '"details": <UI element interacted with>, '
    '"context": <application in use>}.'
)


def df_user_prompt(n_frames: int) -> str:
    return (
        f"You are given {n_frames} sequential screenshots (1 every 0.5s) of a desktop "
        "screen recording. Infer the user's action sequence.\n"
        "Reason step by step about what changed between frames, then output the actions.\n"
        "Drop redundant or invalid actions; merge fragments split across the window.\n"
        + _SCHEMA
    )


def screex_user_prompt(index_json: str) -> str:
    return (
        "Below is a Screex index of a desktop screen recording: a list of UI states, "
        "each with on-screen OCR text and the text that appeared/disappeared vs the "
        "previous state. Infer the user's action sequence from these state transitions.\n"
        + _SCHEMA
        + "\n\nINDEX:\n" + index_json
    )


def gt_user_prompt(n_frames: int) -> str:
    # Distinct wording + dense frames; used only for GT proposal.
    return (
        f"These {n_frames} dense frames capture a desktop recording. Carefully and "
        "exhaustively annotate every discrete user action, in order.\n" + _SCHEMA
    )


def _extract_json_array(text: str) -> str:
    fence = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
    if fence:
        return fence.group(1)
    bare = re.search(r"\[.*\]", text, re.DOTALL)
    return bare.group(0) if bare else "[]"


def parse_actions(text: str) -> list:
    raw = json.loads(_extract_json_array(text))
    out = []
    for d in raw:
        if d.get("op") in OPS:
            out.append(Action.from_dict(d))
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/study/test_prompts.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add study/prompts.py tests/study/test_prompts.py
git commit -m "feat(study): prompt builders + robust action parser"
```

---

## Task 8: The two arms

**Files:**
- Create: `study/arms/__init__.py`
- Create: `study/arms/df.py`
- Create: `study/arms/screex_arm.py`
- Test: append to `tests/study/test_arms.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/study/test_arms.py`:
```python
import json
from study.actions import Action
from study.claude_client import FakeClient
from study.arms.df import run_df_arm
from study.arms.screex_arm import run_screex_arm


def test_df_arm_merges_window_outputs_and_sums_usage(monkeypatch):
    # 12 frames -> 2 windows (size 10, overlap 5). One response per window.
    frames = [(i * 0.5, f"{i}.png") for i in range(12)]
    monkeypatch.setattr(
        "study.arms.df.sample_frame_paths", lambda *a, **k: frames
    )
    client = FakeClient(responses=[
        '[{"op":"click","details":"A","context":"App"}]',
        '[{"op":"type","details":"B","context":"App"}]',
    ])
    actions, usage = run_df_arm("video.mp4", "cache/df", client)
    assert actions == [Action("click", "A", "App"), Action("type", "B", "App")]
    assert len(client.calls) == 2
    assert "input_tokens" in usage and "output_tokens" in usage


def test_screex_arm_reads_index_and_calls_once(tmp_path, monkeypatch):
    index_path = tmp_path / "index.json"
    index_path.write_text(json.dumps({"states": [{"ocr_text": ["File"]}]}), encoding="utf-8")
    monkeypatch.setattr(
        "study.arms.screex_arm.build_index", lambda *a, **k: index_path
    )
    client = FakeClient(responses=['[{"op":"click","details":"File","context":"App"}]'])
    actions, usage = run_screex_arm("video.mp4", "cache/screex", client)
    assert actions == [Action("click", "File", "App")]
    assert len(client.calls) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/study/test_arms.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'study.arms'`

- [ ] **Step 3: Write minimal implementation**

`study/arms/__init__.py`:
```python
```

`study/arms/df.py`:
```python
from __future__ import annotations

from pathlib import Path

from study.config import SAMPLE_FPS, WINDOW_OVERLAP, WINDOW_SIZE
from study.frames import sample_frame_paths, windows
from study.prompts import df_user_prompt, parse_actions

_SYSTEM = "You extract user action sequences from desktop screen recordings."


def run_df_arm(video_path, cache_dir, client, fps=SAMPLE_FPS):
    """Raw-frames -> Claude (windowed propose/merge) -> actions. Returns (actions, usage)."""
    frames = sample_frame_paths(video_path, Path(cache_dir) / "frames", fps=fps)
    paths = [p for _, p in frames]
    all_actions = []
    usage = {"input_tokens": 0, "output_tokens": 0}
    for win in windows(paths, size=WINDOW_SIZE, overlap=WINDOW_OVERLAP):
        text, u = client.complete(
            system=_SYSTEM, user=df_user_prompt(len(win)), images=win
        )
        all_actions.extend(parse_actions(text))
        usage["input_tokens"] += u["input_tokens"]
        usage["output_tokens"] += u["output_tokens"]
    return all_actions, usage
```

`study/arms/screex_arm.py`:
```python
from __future__ import annotations

from pathlib import Path

from screex.cli import index as build_index
from study.config import SAMPLE_FPS
from study.prompts import parse_actions, screex_user_prompt

_SYSTEM = "You extract user action sequences from a Screex UI-state index."


def run_screex_arm(video_path, cache_dir, client, fps=SAMPLE_FPS):
    """Screex index -> Claude -> actions. Returns (actions, usage)."""
    out_dir = Path(cache_dir)
    index_path = build_index(video_path, fps=fps, out=str(out_dir), quiet=True)
    index_json = Path(index_path).read_text(encoding="utf-8")
    text, usage = client.complete(
        system=_SYSTEM, user=screex_user_prompt(index_json)
    )
    return parse_actions(text), usage
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/study/test_arms.py -v`
Expected: PASS (4 tests total in file)

- [ ] **Step 5: Commit**

```bash
git add study/arms/ tests/study/test_arms.py
git commit -m "feat(study): DF baseline arm and Screex arm"
```

---

## Task 9: GT generator

**Files:**
- Create: `study/gt_generate.py`
- Test: append to `tests/study/test_prompts.py` (covers `gt_user_prompt`; generator wiring tested via the arms' fake pattern)

- [ ] **Step 1: Write the failing test**

Append to `tests/study/test_prompts.py`:
```python
from study.prompts import gt_user_prompt
from study.gt_generate import propose_gt


def test_gt_prompt_is_distinct_from_df():
    from study.prompts import df_user_prompt
    assert gt_user_prompt(10) != df_user_prompt(10)


def test_propose_gt_writes_editable_json(tmp_path, monkeypatch):
    from study.actions import Action, load_actions
    from study.claude_client import FakeClient

    frames = [(i * 0.5, f"{i}.png") for i in range(4)]
    monkeypatch.setattr("study.gt_generate.sample_frame_paths", lambda *a, **k: frames)
    client = FakeClient(responses=['[{"op":"click","details":"X","context":"App"}]'])
    out = tmp_path / "v1.json"
    propose_gt("video.mp4", tmp_path / "frames", out, client)
    assert load_actions(out) == [Action("click", "X", "App")]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/study/test_prompts.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'study.gt_generate'`

- [ ] **Step 3: Write minimal implementation**

`study/gt_generate.py`:
```python
from __future__ import annotations

from pathlib import Path

from study.actions import save_actions
from study.frames import sample_frame_paths
from study.prompts import gt_user_prompt, parse_actions

_SYSTEM = "You are a meticulous annotator of desktop screen recordings."
# Dense sampling for GT (distinct from the arms' SAMPLE_FPS).
GT_FPS = 4.0


def propose_gt(video_path, frames_dir, out_path, client, fps=GT_FPS):
    """Propose GT actions from dense frames and write an editable JSON for human review."""
    frames = sample_frame_paths(video_path, frames_dir, fps=fps)
    paths = [p for _, p in frames]
    text, _ = client.complete(
        system=_SYSTEM, user=gt_user_prompt(len(paths)), images=paths
    )
    actions = parse_actions(text)
    save_actions(out_path, actions)
    return actions
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/study/test_prompts.py -v`
Expected: PASS (7 tests in file)

- [ ] **Step 5: Commit**

```bash
git add study/gt_generate.py tests/study/test_prompts.py
git commit -m "feat(study): GT proposer (distinct config) writing editable JSON"
```

> **Human gate (runtime, not a code step):** after running `propose_gt` over every manifest video into `study/gt/*.json`, a human MUST review and correct each file before any scoring. Then commit the frozen GT:
> ```bash
> git add study/gt/
> git commit -m "data(study): freeze human-verified ACTREAL ground truth"
> ```

---

## Task 10: Report aggregation

**Files:**
- Create: `study/report.py`
- Test: `tests/study/test_report.py`

- [ ] **Step 1: Write the failing test**

`tests/study/test_report.py`:
```python
import math
from study.metric import Scores
from study.report import aggregate, format_controlled_table


def test_aggregate_means_across_videos():
    scores = [
        Scores(1.0, 1.0, 1.0, 1.0),
        Scores(0.0, 0.0, 0.0, 0.0),
    ]
    agg = aggregate(scores)
    assert math.isclose(agg["op_precision"], 0.5)
    assert math.isclose(agg["all_recall"], 0.5)


def test_aggregate_empty_is_zeroed():
    agg = aggregate([])
    assert agg["op_precision"] == 0.0


def test_controlled_table_has_both_arms_and_token_cost():
    df = {"op_precision": 0.5, "op_recall": 0.5, "all_precision": 0.4,
          "all_recall": 0.4, "tokens": 100000}
    screex = {"op_precision": 0.7, "op_recall": 0.7, "all_precision": 0.6,
              "all_recall": 0.6, "tokens": 12000}
    table = format_controlled_table(df, screex)
    assert "DF" in table and "Screex" in table
    assert "100000" in table and "12000" in table
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/study/test_report.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'study.report'`

- [ ] **Step 3: Write minimal implementation**

`study/report.py`:
```python
from __future__ import annotations

_FIELDS = ("op_precision", "op_recall", "all_precision", "all_recall")


def aggregate(scores) -> dict:
    if not scores:
        return {f: 0.0 for f in _FIELDS}
    return {f: sum(getattr(s, f) for s in scores) / len(scores) for f in _FIELDS}


def _row(name, d):
    return (f"| {name} | {d['op_precision']:.3f} | {d['op_recall']:.3f} | "
            f"{d['all_precision']:.3f} | {d['all_recall']:.3f} | {d.get('tokens', 0)} |")


def format_controlled_table(df: dict, screex: dict) -> str:
    head = ("| Arm | Op-P | Op-R | All-P | All-R | Tokens |\n"
            "|-----|------|------|-------|-------|--------|")
    return "\n".join([head, _row("DF (raw frames)", df), _row("Screex (index)", screex)])


def format_vs_published_table(screex: dict, published: dict) -> str:
    """published: {"DF": {...}, "DiffF": {...}} from Sharingan's paper (GPT-4o)."""
    head = ("| System | Op-P | Op-R | All-P | All-R |\n"
            "|--------|------|------|-------|-------|")
    rows = [head]
    for name, d in published.items():
        rows.append(f"| Sharingan {name} (GPT-4o) | {d['op_precision']:.3f} | "
                    f"{d['op_recall']:.3f} | {d['all_precision']:.3f} | {d['all_recall']:.3f} |")
    rows.append(f"| Screex (Claude, ours) | {screex['op_precision']:.3f} | "
                f"{screex['op_recall']:.3f} | {screex['all_precision']:.3f} | "
                f"{screex['all_recall']:.3f} |")
    return "\n".join(rows)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/study/test_report.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add study/report.py tests/study/test_report.py
git commit -m "feat(study): score aggregation + controlled/vs-published tables"
```

---

## Task 11: Orchestrator + study README

**Files:**
- Create: `study/run.py`
- Create: `study/README.md`

`run.py` is glue over already-tested units; it is exercised at runtime, not unit-tested (its parts are all covered).

- [ ] **Step 1: Write the orchestrator**

`study/run.py`:
```python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from study.actions import load_actions
from study.arms.df import run_df_arm
from study.arms.screex_arm import run_screex_arm
from study.claude_client import AnthropicClient
from study.config import CLAUDE_MODEL, MATCH_THRESHOLD
from study.embedder import BertEmbedder
from study.metric import score
from study.report import aggregate, format_controlled_table


def _video_path(video_id: str) -> str:
    return str(Path("study/cache/videos") / f"{video_id}.mp4")


def main():
    ap = argparse.ArgumentParser(description="Run the Screex-on-ACTREAL study.")
    ap.add_argument("--manifest", default="study/manifest.json")
    ap.add_argument("--gt-dir", default="study/gt")
    ap.add_argument("--cache", default="study/cache")
    args = ap.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    client = AnthropicClient(CLAUDE_MODEL)
    embedder = BertEmbedder()

    df_scores, screex_scores = [], []
    df_tokens = screex_tokens = 0

    for m in manifest:
        vid = m["video_id"]
        gt = load_actions(Path(args.gt_dir) / f"{vid}.json")
        vpath = _video_path(vid)

        df_actions, df_u = run_df_arm(vpath, f"{args.cache}/df/{vid}", client)
        sx_actions, sx_u = run_screex_arm(vpath, f"{args.cache}/screex/{vid}", client)

        df_scores.append(score(gt, df_actions, embedder, MATCH_THRESHOLD))
        screex_scores.append(score(gt, sx_actions, embedder, MATCH_THRESHOLD))
        df_tokens += df_u["input_tokens"] + df_u["output_tokens"]
        screex_tokens += sx_u["input_tokens"] + sx_u["output_tokens"]

    df_agg = {**aggregate(df_scores), "tokens": df_tokens}
    sx_agg = {**aggregate(screex_scores), "tokens": screex_tokens}
    print(format_controlled_table(df_agg, sx_agg))
    Path(f"{args.cache}/results.json").write_text(
        json.dumps({"df": df_agg, "screex": sx_agg}, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write the README**

`study/README.md`:
```markdown
# Screex-on-Sharingan (ACTREAL) study

Reproduces the Sharingan action-extraction metric and compares a raw-frames baseline
(DF-equivalent) against Screex's index→Claude pipeline. See design:
`docs/superpowers/specs/2026-06-15-screex-sharingan-study-design.md`.

## Setup
    pip install -r study/requirements.txt
    export ANTHROPIC_API_KEY=...   # PowerShell: $env:ANTHROPIC_API_KEY="..."

## Pipeline
1. `python -m study.data_prep`  -> writes/commits `study/manifest.json`
2. Download the manifest's videos into `study/cache/videos/<video_id>.mp4`
3. Generate GT, then HUMAN-VERIFY each `study/gt/<video_id>.json`
4. `python -m study.run` -> prints the controlled table, writes `cache/results.json`

## Caveats
Reconstructed ACTREAL != Sharingan's original 41 videos; GT is LLM-assisted + human-
verified; base model is Claude not GPT-4o. The controlled arm (same model both sides) is
the rigorous claim; vs-published is indicative.
```

- [ ] **Step 3: Run the full study test suite**

Run: `python -m pytest tests/study/ -v`
Expected: PASS (all tasks' tests green)

- [ ] **Step 4: Lint (project requires ruff before committing)**

Run: `ruff check study tests/study`
Expected: no errors (fix any reported)

- [ ] **Step 5: Commit**

```bash
git add study/run.py study/README.md
git commit -m "feat(study): orchestrator + reproducibility README"
```

---

## Runtime Sequence (after code lands)

These are operator steps, not code tasks — run in order:

1. `python -m study.data_prep` → freeze `study/manifest.json` (commit).
2. Download manifest videos → `study/cache/videos/`.
3. `propose_gt` over all videos → `study/gt/*.json`; **human-verify each**; freeze (commit).
4. `python -m study.run` → controlled table + `cache/results.json`.
5. Fill `format_vs_published_table` with Sharingan's reported DF/DiffF numbers; add both tables to the paper writeup.

---

## Self-Review Notes (author check)

- **Spec coverage:** data reconstruction (Task 4), GT propose+verify (Task 9 + human gate), DF arm (Task 8), Screex arm (Task 8), Sharingan metric (Task 3), controlled + vs-published tables (Task 10), token cost (Tasks 8/10/11), caveats (README). All spec sections mapped.
- **Type consistency:** `Action(op, details, context)`, `score(gt, pred, embedder, threshold) -> Scores`, `ClaudeClient.complete(system, user, images) -> (text, usage)`, `run_df_arm/run_screex_arm(video, cache, client) -> (actions, usage)` used consistently across tasks.
- **Known runtime confirmations (flagged inline):** exact HF GUI-World field names (Task 4 Step 6), `iter_frames` item attribute names (Task 5 note). Both isolated behind adapters with documented fallbacks.
```
