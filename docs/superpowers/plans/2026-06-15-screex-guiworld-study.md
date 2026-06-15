# Screex-on-GUI-World (MCQA) Study Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Build a reproducible harness that compares answering GUI-World multiple-choice questions from a Screex index vs from raw sampled frames, with the same Claude model, scored by exact-match accuracy + token cost.

**Architecture:** Standalone `study/` package imports `screex` as a library. Pure logic (MCQA model/parsing/accuracy, dataset filter/flatten/sample, prompt building, report aggregation) is unit-tested with fakes; HF/LLM/opencv boundaries are thin wrappers behind protocols so tests run offline. Two arms answer the same MCQ and emit an option letter.

**Tech Stack:** Python 3.9+, screex (local), opencv (via screex), anthropic SDK, datasets/huggingface_hub, pytest.

**Supersedes the Sharingan plan.** This branch (`study/screex-sharingan-actreal`) already has Tasks 0–2 built; Task A below retires the Sharingan-specific modules (`actions.py`, `metric.py`) and repurposes `config.py`.

---

## File Structure

```
study/
  config.py          # GUI-World knobs (rewritten)
  mcqa.py            # MCQItem + parse_answer_letter + accuracy
  dataset.py         # desktop filter + flatten MCQA + deterministic sample + manifest
  claude_client.py   # ClaudeClient protocol + FakeClient + AnthropicClient
  frames.py          # uniform frame sampling (Arm A)
  screex_index.py    # thin wrapper over screex.cli.index
  prompts.py         # MCQA prompt builders for both arms
  arms/__init__.py
  arms/frames_arm.py # raw frames -> Claude -> letter
  arms/screex_arm.py # screex index -> Claude -> letter
  run.py             # orchestrate both arms over manifest, cache
  report.py          # accuracy (overall + per scenario) + token cost
  README.md
tests/study/
  test_mcqa.py  test_dataset.py  test_client.py  test_frames.py
  test_prompts.py  test_arms.py  test_report.py
```

Shared types:
- `MCQItem(item_id, scenario, video_path, question, options: list[str], answer: str)` — `answer` is a single uppercase option letter ("A".."Z").
- `ClaudeClient.complete(system, user, images=None) -> (text, usage)` where `usage = {"input_tokens", "output_tokens"}`.

---

## Task A: Retire Sharingan modules + rewrite config

**Files:**
- Delete: `study/actions.py`, `study/metric.py`, `tests/study/test_actions.py`, `tests/study/test_metric.py`
- Modify: `study/config.py`
- Keep untouched: `study/embedder.py` + its test (harmless; possible future free-form arm)

- [ ] **Step 1: Delete the Sharingan-specific modules and tests**

```bash
git rm study/actions.py study/metric.py tests/study/test_actions.py tests/study/test_metric.py
```

- [ ] **Step 2: Rewrite `study/config.py`**

```python
"""Central knobs for the GUI-World MCQA study. Recorded in reports for reproducibility."""

# Base model used for BOTH arms (controlled comparison). Fixed and recorded.
CLAUDE_MODEL = "claude-opus-4-8"

# GUI-World scenarios kept (desktop domain only; mobile + XR excluded).
DESKTOP_SCENARIOS = ("desktop", "software", "website", "multi")

# Sampling.
SAMPLE_SIZE = 300          # MCQs sampled from the desktop-domain subset
SAMPLE_SEED = 20260615

# Arm A (raw frames) perception budget: number of uniformly sampled frames per video.
FRAME_BUDGET = 8

# Screex index sampling fps for Arm B.
INDEX_FPS = 2.0

# Hugging Face dataset id.
HF_DATASET = "shuaishuaicdp/GUI-World"
```

- [ ] **Step 3: Verify the suite still green (embedder test remains)**

Run: `python -m pytest tests/study/ -v`
Expected: PASS (embedder tests only; no import errors from deleted modules)

- [ ] **Step 4: Commit**

```bash
git add -A study/ tests/study/
git commit -m "refactor(study): retire Sharingan modules, repurpose config for GUI-World"
```

---

## Task B: MCQA model + answer parsing + accuracy

**Files:**
- Create: `study/mcqa.py`
- Test: `tests/study/test_mcqa.py`

- [ ] **Step 1: Write the failing test**

`tests/study/test_mcqa.py`:
```python
import pytest
from study.mcqa import MCQItem, parse_answer_letter, accuracy


def _item(answer="B"):
    return MCQItem(
        item_id="v1#0", scenario="software", video_path="v1.mov",
        question="What did the user click?",
        options=["File", "Edit", "View"], answer=answer,
    )


def test_parse_plain_letter():
    assert parse_answer_letter("B", n_options=3) == "B"


def test_parse_letter_in_sentence():
    assert parse_answer_letter("The answer is (C).", n_options=3) == "C"


def test_parse_first_letter_wins():
    assert parse_answer_letter("A, not B", n_options=3) == "A"


def test_parse_rejects_out_of_range_letter():
    # Only 3 options (A-C); 'D' must be ignored.
    assert parse_answer_letter("D then A", n_options=3) == "A"


def test_parse_unparseable_returns_none():
    assert parse_answer_letter("I don't know", n_options=3) is None


def test_accuracy_counts_correct_letters():
    items = [_item("B"), _item("A")]
    preds = ["B", "C"]   # first correct, second wrong
    assert accuracy(items, preds) == pytest.approx(0.5)


def test_accuracy_none_prediction_is_incorrect():
    items = [_item("B")]
    assert accuracy(items, [None]) == 0.0


def test_accuracy_empty_is_zero():
    assert accuracy([], []) == 0.0
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/study/test_mcqa.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'study.mcqa'`

- [ ] **Step 3: Write minimal implementation**

`study/mcqa.py`:
```python
from __future__ import annotations

import re
import string
from dataclasses import dataclass


@dataclass(frozen=True)
class MCQItem:
    item_id: str
    scenario: str
    video_path: str
    question: str
    options: list
    answer: str  # single uppercase option letter, e.g. "B"


def parse_answer_letter(text, n_options):
    """Return the first standalone option letter within range, else None."""
    valid = set(string.ascii_uppercase[:n_options])
    for m in re.finditer(r"[A-Z]", text.upper()):
        if m.group(0) in valid:
            return m.group(0)
    return None


def accuracy(items, preds) -> float:
    """Fraction of items whose predicted letter equals the gold answer.
    `preds` is parallel to `items`; a None prediction counts as incorrect."""
    if not items:
        return 0.0
    correct = sum(1 for it, p in zip(items, preds) if p is not None and p == it.answer)
    return correct / len(items)
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/study/test_mcqa.py -v`
Expected: PASS (8 tests)

- [ ] **Step 5: ruff + commit**

```bash
ruff check study/mcqa.py tests/study/test_mcqa.py
git add study/mcqa.py tests/study/test_mcqa.py
git commit -m "feat(study): MCQItem, answer-letter parser, accuracy metric"
```

---

## Task C: Dataset filter + flatten + sample + manifest

**Files:**
- Create: `study/dataset.py`
- Test: `tests/study/test_dataset.py`

The HF load is a thin wrapper; the pure functions (scenario classification, MCQA flatten, deterministic sample) are tested on synthetic records. **Runtime confirm:** exact GUI-World field names are validated against the live HF dataset before the real pull (flagged in Step 6).

- [ ] **Step 1: Write the failing test**

`tests/study/test_dataset.py`:
```python
from study.dataset import scenario_of, is_desktop_domain, flatten_mcqa, sample_items, letter_of
from study.mcqa import MCQItem


def test_scenario_desktop_from_macos():
    assert scenario_of({"system": "macOS", "multi": False}) == "desktop"


def test_scenario_multi_flag_wins():
    assert scenario_of({"system": "Windows", "multi": True}) == "multi"


def test_is_desktop_domain_excludes_mobile():
    assert is_desktop_domain({"system": "iOS", "multi": False}) is False
    assert is_desktop_domain({"system": "macOS", "multi": False}) is True


def test_letter_of_handles_letter_and_text():
    opts = ["File", "Edit", "View"]
    assert letter_of("B", opts) == "B"
    assert letter_of("Edit", opts) == "B"      # gold given as option text
    assert letter_of("(C)", opts) == "C"


def test_flatten_mcqa_builds_items():
    record = {
        "video_path": "IOS/0.mov", "system": "macOS", "multi": False,
        "MCQA": [
            {"Question": "Q1?", "Options": ["File", "Edit", "View"], "Correct Answer": "Edit"},
            {"Question": "Q2?", "Options": ["Yes", "No"], "Correct Answer": "A"},
        ],
    }
    items = flatten_mcqa(record, "macrec0")
    assert len(items) == 2
    assert items[0] == MCQItem("macrec0#0", "desktop", "IOS/0.mov", "Q1?",
                               ["File", "Edit", "View"], "B")
    assert items[1].answer == "A"


def test_flatten_skips_malformed():
    record = {"video_path": "x.mov", "system": "macOS", "multi": False,
              "MCQA": [{"Question": "Q?"}]}  # missing Options/Correct Answer
    assert flatten_mcqa(record, "r") == []


def test_sample_items_deterministic():
    items = [f"i{n}" for n in range(100)]
    a = sample_items(items, n=10, seed=1)
    b = sample_items(items, n=10, seed=1)
    assert a == b and len(a) == 10


def test_sample_items_caps_at_population():
    items = ["a", "b"]
    assert len(sample_items(items, n=10, seed=1)) == 2
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/study/test_dataset.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'study.dataset'`

- [ ] **Step 3: Write minimal implementation**

`study/dataset.py`:
```python
from __future__ import annotations

import json
import random
import string
from pathlib import Path

from study.config import DESKTOP_SCENARIOS
from study.mcqa import MCQItem

_DESKTOP_SYSTEMS = {"macos", "windows", "linux"}


def scenario_of(record) -> str:
    """Classify a GUI-World record into a coarse scenario label."""
    if record.get("multi"):
        return "multi"
    system = str(record.get("system", "")).lower()
    if system in _DESKTOP_SYSTEMS:
        return "desktop"
    if system in {"ios", "android"}:
        return "mobile"
    if system in {"xr", "visionos", "vision pro"}:
        return "xr"
    # Fall back to an explicit scenario/type field if present.
    return str(record.get("scenario") or record.get("type") or "website").lower()


def is_desktop_domain(record) -> bool:
    return scenario_of(record) in DESKTOP_SCENARIOS


def letter_of(correct, options) -> str:
    """Normalize a gold 'Correct Answer' (a letter, '(C)', or option text) to a letter."""
    valid = list(string.ascii_uppercase[:len(options)])
    s = str(correct).strip()
    # Direct letter (possibly wrapped like "(C)" or "C.")
    for ch in s.upper():
        if ch in valid:
            return ch
        break
    # Otherwise match against option text.
    for i, opt in enumerate(options):
        if str(opt).strip().lower() == s.lower():
            return valid[i]
    raise ValueError(f"cannot map correct answer {correct!r} to options {options!r}")


def flatten_mcqa(record, base_id) -> list:
    """Turn one GUI-World record's MCQA list into MCQItems. Skips malformed entries."""
    scenario = scenario_of(record)
    video = record.get("video_path", "")
    out = []
    for i, q in enumerate(record.get("MCQA") or []):
        question = q.get("Question")
        options = q.get("Options")
        correct = q.get("Correct Answer")
        if not question or not options or correct is None:
            continue
        try:
            answer = letter_of(correct, options)
        except ValueError:
            continue
        out.append(MCQItem(f"{base_id}#{i}", scenario, video, question, list(options), answer))
    return out


def sample_items(items, n, seed):
    items = list(items)
    if len(items) <= n:
        return items
    return random.Random(seed).sample(items, n)


def write_manifest(items, out_path) -> None:
    payload = [
        {"item_id": it.item_id, "scenario": it.scenario, "video_path": it.video_path,
         "question": it.question, "options": it.options, "answer": it.answer}
        for it in items
    ]
    Path(out_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_manifest(path) -> list:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return [MCQItem(**d) for d in raw]
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/study/test_dataset.py -v`
Expected: PASS (8 tests)

- [ ] **Step 5: ruff + commit**

```bash
ruff check study/dataset.py tests/study/test_dataset.py
git add study/dataset.py tests/study/test_dataset.py
git commit -m "feat(study): GUI-World desktop filter, MCQA flatten, deterministic sampling"
```

- [ ] **Step 6: Runtime data pull (operator, not unit-tested)**

After confirming the live HF schema interactively (field names `MCQA`, `Question`, `Options`,
`Correct Answer`, `system`, `multi`, `video_path` — adapt in `dataset.py` if they differ),
add a `main()` that: streams `datasets.load_dataset(HF_DATASET, split="test")`, builds items
via `flatten_mcqa`, filters `is_desktop_domain`, samples `SAMPLE_SIZE`/`SAMPLE_SEED`, and
calls `write_manifest("study/manifest.json")`. Run it, eyeball, then:

```bash
python -m study.dataset
git add study/manifest.json
git commit -m "data(study): freeze sampled GUI-World desktop MCQA manifest"
```

---

## Task D: Claude client wrapper

**Files:**
- Create: `study/claude_client.py`
- Test: `tests/study/test_client.py`

- [ ] **Step 1: Write the failing test**

`tests/study/test_client.py`:
```python
from study.claude_client import FakeClient


def test_fake_client_returns_scripted_text_and_usage():
    c = FakeClient(responses=["B"])
    text, usage = c.complete(system="s", user="u")
    assert text == "B"
    assert usage == {"input_tokens": 0, "output_tokens": 0}


def test_fake_client_records_calls():
    c = FakeClient(responses=["A", "B"])
    c.complete(system="s1", user="u1")
    c.complete(system="s2", user="u2", images=["f.png"])
    assert [call["user"] for call in c.calls] == ["u1", "u2"]
    assert c.calls[1]["images"] == ["f.png"]
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/study/test_client.py -v`
Expected: FAIL with `ModuleNotFoundError`

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
        return self._responses.pop(0), {"input_tokens": 0, "output_tokens": 0}


class AnthropicClient:
    """Real client. Sends optional images as base64 blocks before the text."""

    def __init__(self, model):
        import anthropic

        self._client = anthropic.Anthropic()
        self._model = model

    def complete(self, system: str, user: str, images=None) -> tuple:
        content = []
        for img in images or []:
            data = base64.standard_b64encode(Path(img).read_bytes()).decode()
            media = "image/png" if str(img).endswith(".png") else "image/jpeg"
            content.append({"type": "image",
                            "source": {"type": "base64", "media_type": media, "data": data}})
        content.append({"type": "text", "text": user})
        msg = self._client.messages.create(
            model=self._model, max_tokens=1024, system=system,
            messages=[{"role": "user", "content": content}],
        )
        text = "".join(b.text for b in msg.content if b.type == "text")
        return text, {"input_tokens": msg.usage.input_tokens,
                      "output_tokens": msg.usage.output_tokens}
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/study/test_client.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: ruff + commit**

```bash
ruff check study/claude_client.py tests/study/test_client.py
git add study/claude_client.py tests/study/test_client.py
git commit -m "feat(study): Claude client protocol with fake + Anthropic impl"
```

---

## Task E: Uniform frame sampling (Arm A perception)

**Files:**
- Create: `study/frames.py`
- Test: `tests/study/test_frames.py`

- [ ] **Step 1: Write the failing test**

`tests/study/test_frames.py`:
```python
from study.frames import pick_uniform_indices


def test_pick_uniform_indices_basic():
    # 100 frames, budget 5 -> evenly spaced, includes endpoints region, sorted, unique.
    idx = pick_uniform_indices(total=100, budget=5)
    assert idx == sorted(set(idx))
    assert len(idx) == 5
    assert idx[0] >= 0 and idx[-1] <= 99


def test_pick_uniform_indices_budget_exceeds_total():
    assert pick_uniform_indices(total=3, budget=8) == [0, 1, 2]


def test_pick_uniform_indices_zero_total():
    assert pick_uniform_indices(total=0, budget=8) == []
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/study/test_frames.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

`study/frames.py`:
```python
from __future__ import annotations

from pathlib import Path

from study.config import FRAME_BUDGET


def pick_uniform_indices(total, budget=FRAME_BUDGET):
    """Evenly spaced unique frame indices across [0, total)."""
    if total <= 0:
        return []
    if total <= budget:
        return list(range(total))
    step = total / budget
    idx = sorted({int(i * step) for i in range(budget)})
    return idx[:budget]


def sample_frame_images(video_path, out_dir, budget=FRAME_BUDGET):
    """Decode the video, write `budget` uniformly-spaced frames as PNGs, return their paths."""
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    wanted = set(pick_uniform_indices(total, budget))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths, i = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i in wanted:
            p = out_dir / f"{i:06d}.png"
            cv2.imwrite(str(p), frame)
            paths.append(str(p))
        i += 1
    cap.release()
    return paths
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/study/test_frames.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: ruff + commit**

```bash
ruff check study/frames.py tests/study/test_frames.py
git add study/frames.py tests/study/test_frames.py
git commit -m "feat(study): uniform frame sampling for the raw-frames arm"
```

---

## Task F: Prompts + MCQA option formatting

**Files:**
- Create: `study/prompts.py`
- Test: `tests/study/test_prompts.py`

- [ ] **Step 1: Write the failing test**

`tests/study/test_prompts.py`:
```python
from study.prompts import format_options, frames_prompt, index_prompt


def test_format_options_letters():
    assert format_options(["File", "Edit"]) == "A. File\nB. Edit"


def test_frames_prompt_contains_question_options_and_letter_instruction():
    p = frames_prompt("What was clicked?", ["File", "Edit"])
    assert "What was clicked?" in p
    assert "A. File" in p and "B. Edit" in p
    assert "letter" in p.lower()


def test_index_prompt_embeds_index_json_and_question():
    p = index_prompt("Q?", ["Yes", "No"], '{"states": []}')
    assert "Q?" in p
    assert '{"states": []}' in p
    assert "A. Yes" in p
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/study/test_prompts.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

`study/prompts.py`:
```python
from __future__ import annotations

import string

_ANSWER_RULE = "Answer with ONLY the single letter of the correct option."


def format_options(options) -> str:
    letters = string.ascii_uppercase
    return "\n".join(f"{letters[i]}. {opt}" for i, opt in enumerate(options))


def frames_prompt(question, options) -> str:
    return (
        "You are shown sequential frames from a GUI screen recording.\n"
        f"Question: {question}\n\nOptions:\n{format_options(options)}\n\n{_ANSWER_RULE}"
    )


def index_prompt(question, options, index_json) -> str:
    return (
        "Below is a Screex index of a GUI screen recording: a list of UI states with "
        "on-screen OCR text and the text that appeared/disappeared between states.\n"
        f"Question: {question}\n\nOptions:\n{format_options(options)}\n\n{_ANSWER_RULE}"
        "\n\nINDEX:\n" + index_json
    )
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/study/test_prompts.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: ruff + commit**

```bash
ruff check study/prompts.py tests/study/test_prompts.py
git add study/prompts.py tests/study/test_prompts.py
git commit -m "feat(study): MCQA prompt builders for both arms"
```

---

## Task G: Screex index wrapper

**Files:**
- Create: `study/screex_index.py`
- Test: `tests/study/test_arms.py` (created here; arms appended in Task H)

- [ ] **Step 1: Write the failing test**

`tests/study/test_arms.py`:
```python
import json
from study.screex_index import build_index_json


def test_build_index_json_reads_index_file(tmp_path, monkeypatch):
    index_path = tmp_path / "index.json"
    index_path.write_text(json.dumps({"states": [{"ocr_text": ["File"]}]}), encoding="utf-8")
    # Stub screex.cli.index so no video decoding happens.
    monkeypatch.setattr("study.screex_index._screex_index",
                        lambda video, fps, out, quiet: index_path)
    js = build_index_json("v.mov", tmp_path / "idx", fps=2.0)
    assert json.loads(js)["states"][0]["ocr_text"] == ["File"]
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/study/test_arms.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'study.screex_index'`

- [ ] **Step 3: Write minimal implementation**

`study/screex_index.py`:
```python
from __future__ import annotations

from pathlib import Path

from study.config import INDEX_FPS


def _screex_index(video, fps, out, quiet):
    """Indirection seam so tests can stub the heavy screex call."""
    from screex.cli import index as screex_cli_index

    return screex_cli_index(video, fps=fps, out=out, quiet=quiet)


def build_index_json(video_path, cache_dir, fps=INDEX_FPS) -> str:
    """Build a Screex index for the video and return its index.json text."""
    index_path = _screex_index(video_path, fps, str(cache_dir), True)
    return Path(index_path).read_text(encoding="utf-8")
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/study/test_arms.py -v`
Expected: PASS (1 test)

- [ ] **Step 5: ruff + commit**

```bash
ruff check study/screex_index.py tests/study/test_arms.py
git add study/screex_index.py tests/study/test_arms.py
git commit -m "feat(study): screex index wrapper with test seam"
```

---

## Task H: The two arms

**Files:**
- Create: `study/arms/__init__.py`, `study/arms/frames_arm.py`, `study/arms/screex_arm.py`
- Test: append to `tests/study/test_arms.py`

- [ ] **Step 1: Write the failing test (append)**

Append to `tests/study/test_arms.py`:
```python
from study.arms.frames_arm import answer_with_frames
from study.arms.screex_arm import answer_with_index
from study.claude_client import FakeClient
from study.mcqa import MCQItem


def _item():
    return MCQItem("v1#0", "software", "v1.mov", "What was clicked?",
                   ["File", "Edit", "View"], "B")


def test_frames_arm_returns_letter_and_usage(tmp_path, monkeypatch):
    monkeypatch.setattr("study.arms.frames_arm.sample_frame_images",
                        lambda *a, **k: ["0.png", "1.png"])
    client = FakeClient(responses=["The answer is B"])
    letter, usage = answer_with_frames(_item(), tmp_path, client)
    assert letter == "B"
    assert client.calls[0]["images"] == ["0.png", "1.png"]
    assert "input_tokens" in usage


def test_screex_arm_returns_letter_and_usage(tmp_path, monkeypatch):
    monkeypatch.setattr("study.arms.screex_arm.build_index_json",
                        lambda *a, **k: '{"states": []}')
    client = FakeClient(responses=["C"])
    letter, usage = answer_with_index(_item(), tmp_path, client)
    assert letter == "C"
    assert client.calls[0]["images"] is None  # index arm is text-only
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/study/test_arms.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'study.arms'`

- [ ] **Step 3: Write minimal implementation**

`study/arms/__init__.py`:
```python
```

`study/arms/frames_arm.py`:
```python
from __future__ import annotations

from study.frames import sample_frame_images
from study.mcqa import parse_answer_letter
from study.prompts import frames_prompt

_SYSTEM = "You answer multiple-choice questions about GUI screen recordings."


def answer_with_frames(item, cache_dir, client):
    """Arm A: uniform raw frames -> Claude -> option letter. Returns (letter, usage)."""
    images = sample_frame_images(item.video_path, cache_dir)
    text, usage = client.complete(
        system=_SYSTEM, user=frames_prompt(item.question, item.options), images=images
    )
    return parse_answer_letter(text, len(item.options)), usage
```

`study/arms/screex_arm.py`:
```python
from __future__ import annotations

from study.mcqa import parse_answer_letter
from study.prompts import index_prompt
from study.screex_index import build_index_json

_SYSTEM = "You answer multiple-choice questions from a Screex GUI-state index."


def answer_with_index(item, cache_dir, client):
    """Arm B: Screex index -> Claude -> option letter. Returns (letter, usage)."""
    index_json = build_index_json(item.video_path, cache_dir)
    text, usage = client.complete(
        system=_SYSTEM, user=index_prompt(item.question, item.options, index_json)
    )
    return parse_answer_letter(text, len(item.options)), usage
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/study/test_arms.py -v`
Expected: PASS (3 tests in file)

- [ ] **Step 5: ruff + commit**

```bash
ruff check study/arms tests/study/test_arms.py
git add study/arms tests/study/test_arms.py
git commit -m "feat(study): raw-frames arm and Screex-index arm for MCQA"
```

---

## Task I: Report aggregation

**Files:**
- Create: `study/report.py`
- Test: `tests/study/test_report.py`

- [ ] **Step 1: Write the failing test**

`tests/study/test_report.py`:
```python
import pytest
from study.mcqa import MCQItem
from study.report import per_scenario_accuracy, format_table


def _item(scenario, answer):
    return MCQItem(f"{scenario}#0", scenario, "v.mov", "q", ["A", "B"], answer)


def test_per_scenario_accuracy():
    items = [_item("software", "A"), _item("software", "B"), _item("website", "A")]
    preds = ["A", "A", "A"]  # software 1/2, website 1/1
    acc = per_scenario_accuracy(items, preds)
    assert acc["software"] == pytest.approx(0.5)
    assert acc["website"] == pytest.approx(1.0)
    assert acc["overall"] == pytest.approx(2 / 3)


def test_format_table_has_both_arms_and_tokens():
    frames = {"overall": 0.5, "tokens": 100000}
    screex = {"overall": 0.7, "tokens": 12000}
    table = format_table(frames, screex)
    assert "Frames" in table and "Screex" in table
    assert "100000" in table and "12000" in table
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/study/test_report.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

`study/report.py`:
```python
from __future__ import annotations


def per_scenario_accuracy(items, preds) -> dict:
    """Accuracy per scenario plus an 'overall' key."""
    buckets = {}
    correct_total = 0
    for it, p in zip(items, preds):
        ok = 1 if (p is not None and p == it.answer) else 0
        c, n = buckets.get(it.scenario, (0, 0))
        buckets[it.scenario] = (c + ok, n + 1)
        correct_total += ok
    acc = {s: c / n for s, (c, n) in buckets.items()}
    acc["overall"] = correct_total / len(items) if items else 0.0
    return acc


def format_table(frames: dict, screex: dict) -> str:
    head = "| Arm | Accuracy | Tokens |\n|-----|----------|--------|"
    rows = [
        head,
        f"| Frames (baseline) | {frames['overall']:.3f} | {frames.get('tokens', 0)} |",
        f"| Screex (index) | {screex['overall']:.3f} | {screex.get('tokens', 0)} |",
    ]
    return "\n".join(rows)
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/study/test_report.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: ruff + commit**

```bash
ruff check study/report.py tests/study/test_report.py
git add study/report.py tests/study/test_report.py
git commit -m "feat(study): per-scenario accuracy + controlled table"
```

---

## Task J: Orchestrator + README

**Files:**
- Create: `study/run.py`, `study/README.md`

`run.py` is glue over tested units; exercised at runtime.

- [ ] **Step 1: Write the orchestrator**

`study/run.py`:
```python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from study.arms.frames_arm import answer_with_frames
from study.arms.screex_arm import answer_with_index
from study.claude_client import AnthropicClient
from study.config import CLAUDE_MODEL
from study.dataset import load_manifest
from study.report import format_table, per_scenario_accuracy


def main():
    ap = argparse.ArgumentParser(description="Run the Screex-on-GUI-World MCQA study.")
    ap.add_argument("--manifest", default="study/manifest.json")
    ap.add_argument("--videos", default="study/cache/videos")
    ap.add_argument("--cache", default="study/cache")
    args = ap.parse_args()

    items = load_manifest(args.manifest)
    client = AnthropicClient(CLAUDE_MODEL)

    frame_preds, screex_preds = [], []
    frame_tokens = screex_tokens = 0
    for it in items:
        # Resolve the manifest's relative video_path under --videos.
        video = str(Path(args.videos) / Path(it.video_path).name)
        it_v = it.__class__(it.item_id, it.scenario, video, it.question, it.options, it.answer)

        fl, fu = answer_with_frames(it_v, f"{args.cache}/frames/{it.item_id}", client)
        sl, su = answer_with_index(it_v, f"{args.cache}/screex/{it.item_id}", client)
        frame_preds.append(fl)
        screex_preds.append(sl)
        frame_tokens += fu["input_tokens"] + fu["output_tokens"]
        screex_tokens += su["input_tokens"] + su["output_tokens"]

    frames_acc = {**per_scenario_accuracy(items, frame_preds), "tokens": frame_tokens}
    screex_acc = {**per_scenario_accuracy(items, screex_preds), "tokens": screex_tokens}
    print(format_table(frames_acc, screex_acc))
    Path(f"{args.cache}/results.json").write_text(
        json.dumps({"frames": frames_acc, "screex": screex_acc}, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write `study/README.md`**

```markdown
# Screex-on-GUI-World (MCQA) study

Compares answering GUI-World multiple-choice questions from a Screex index vs raw sampled
frames, same Claude model, scored by exact-match accuracy + token cost. Design:
`docs/superpowers/specs/2026-06-15-screex-guiworld-study-design.md`.

## Setup
    pip install -r study/requirements.txt
    # PowerShell: $env:ANTHROPIC_API_KEY="..."

## Pipeline
1. `python -m study.dataset`  -> writes/commits `study/manifest.json` (desktop MCQA sample)
2. Download the manifest's videos into `study/cache/videos/<name>.mov`
3. `python -m study.run` -> prints the accuracy/token table, writes `cache/results.json`

## Caveats
Subset of GUI-World (frozen manifest); base model is Claude (not the paper's GPT-4V/Gemini),
so cross-paper numbers are indicative — the controlled arm (same model both sides) is the
rigorous claim.
```

- [ ] **Step 3: Update `study/requirements.txt`**

Ensure it contains (Task A may have left sentence-transformers; that's fine for the kept
embedder module):
```
anthropic>=0.40
datasets>=2.0
huggingface_hub>=0.20
sentence-transformers>=2.2
```

- [ ] **Step 4: Full suite + lint**

Run: `python -m pytest tests/study/ -v`  → all green
Run: `ruff check study tests/study`  → no errors

- [ ] **Step 5: Commit**

```bash
git add study/run.py study/README.md study/requirements.txt
git commit -m "feat(study): GUI-World MCQA orchestrator + README"
```

---

## Runtime Sequence (after code lands)

1. `python -m study.dataset` → freeze `study/manifest.json` (commit).
2. Download the sampled videos into `study/cache/videos/`.
3. `python -m study.run` → accuracy/token table + `cache/results.json`.
4. (Optional) frame-budget sweep + McNemar significance; add to the paper writeup.

---

## Self-Review Notes (author check)

- **Spec coverage:** desktop filter + MCQA flatten + sample + manifest (Task C); MCQA model +
  accuracy (Task B); raw-frames arm (Task H/E); Screex-index arm (Task H/G); same-model
  controlled run + token cost (Tasks H/I/J); per-scenario accuracy + table (Task I); caveats
  (README). Retirement of Sharingan modules (Task A). All spec sections mapped.
- **Type consistency:** `MCQItem(item_id, scenario, video_path, question, options, answer)`;
  `ClaudeClient.complete(system, user, images) -> (text, usage)`;
  `answer_with_frames/answer_with_index(item, cache_dir, client) -> (letter, usage)`;
  `parse_answer_letter(text, n_options) -> str|None` used consistently.
- **Runtime confirmations (flagged inline):** GUI-World HF field names (Task C Step 6);
  `cv2.CAP_PROP_FRAME_COUNT` decode loop (Task E). Both isolated behind seams.
```
