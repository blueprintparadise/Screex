# Screex on GUI-World (MCQA) — Benchmark Study Design

**Date:** 2026-06-15
**Status:** Approved (design)
**Supersedes:** `2026-06-15-screex-sharingan-study-design.md` (Sharingan shipped no GT).
**Topic:** Evaluate Screex's "cheap index → Claude reads it" pipeline on GUI-video
multiple-choice question answering, using the GUI-World benchmark's released MCQA ground
truth.

---

## 1. Goal & Claim

Evaluate Screex on **GUI screen-recording understanding**, measured by multiple-choice
question accuracy on the **GUI-World** benchmark (arXiv 2406.10819; ICLR 2025).

**Controlled claim (same Claude model both arms):** answering GUI-World MCQA from a
**Screex index** (OCR text + keyframes/thumbnails) is **as accurate as or more accurate
than** answering from **raw sampled video frames**, at substantially **lower token cost**.

This tests Screex's core thesis — a cheap OCR+keyframe index is a better/cheaper LLM
perception layer than raw frames — on a real, released benchmark with **objective,
exact-match ground truth**. No ground-truth manufacturing, no dataset reconstruction.

## 2. Why GUI-World (vs the superseded Sharingan plan)

- **Real GT shipped.** GUI-World releases human-annotated MCQA with `Question / Options /
  Correct Answer`. Sharingan released neither data nor GT.
- **Objective metric.** MCQA is scored by **exact-match accuracy** on the chosen option —
  no LLM-judge, no semantic-threshold tuning.
- **Public + large.** 12,379 videos, **24,758 MCQs**, 6 scenarios, on Hugging Face
  (`shuaishuaicdp/GUI-World`); official repo `Dongping-Chen/GUI-World`.
- ACTREAL (the Sharingan set) was itself carved from GUI-World — this is the source.

## 3. Scope

- **Task:** MCQA only (exact-match accuracy). Free-form / LLM-judge tasks are out of scope.
- **Scenarios:** desktop-domain only — desktop OS, software, website, multi-window. Mobile
  and XR are **excluded** (outside Screex's domain).
- **Sampling:** a few hundred MCQs sampled from the desktop-domain subset (start with a
  ~30–50-video pilot, then scale). The exact sampled set is frozen to `manifest.json` and
  committed for reproducibility.

## 4. The Two Arms

Both arms answer the **same** MCQ (question + options) and use the **same Claude model**
(fixed, recorded). Both log token usage. Each arm outputs a single option letter.

- **Arm A — Raw frames (baseline):** sample N uniform frames from the video, send them as
  images to Claude with the question + options → option letter. This is the
  "frames as perception layer" baseline Screex argues against.
- **Arm B — Screex index:** run `screex index` on the video; send the index JSON
  (OCR text + text-diff per UI state; optionally a few keyframe images) to Claude with the
  question + options → option letter.

## 5. Metric

- **Accuracy** = correct / total, computed overall and **per scenario**.
- A prediction is correct iff the parsed option letter equals the gold `Correct Answer`.
- Robust answer parsing: extract the first standalone option letter (A/B/C/D/...) from the
  model's reply; unparseable replies count as incorrect.
- **Token cost** per arm (sum of input+output) reported alongside accuracy — the efficiency
  half of the claim.
- Optional secondary: McNemar's test on per-item correctness for the controlled pair.

## 6. Harness Layout

Standalone `study/` package importing `screex` as a library. Eval-only deps (datasets,
huggingface_hub, anthropic) stay out of the shipped pip package.

```
study/
  config.py          # scenarios, sample size, model, frame budget, seed
  dataset.py         # load GUI-World (HF) -> desktop filter -> flatten MCQA -> sample -> manifest
  mcqa.py            # MCQItem model + parse_answer_letter + accuracy
  claude_client.py   # ClaudeClient protocol + FakeClient + AnthropicClient
  frames.py          # uniform frame sampling for Arm A
  screex_index.py    # thin wrapper over screex.cli.index
  prompts.py         # MCQA prompt builders (frames arm, index arm)
  arms/frames_arm.py # raw frames -> Claude -> letter
  arms/screex_arm.py # screex index -> Claude -> letter
  run.py             # orchestrate both arms over manifest, cache outputs
  report.py          # accuracy (overall + per scenario) + token cost tables
  README.md
  manifest.json      # frozen sampled MCQA set (committed)
  cache/             # videos, frames, indexes, model outputs (gitignored)
```

Model outputs cached per (arm, item) so accuracy is re-scorable without re-calling Claude.

## 7. Reuse / Retire from the Sharingan build

Already built on this branch:
- **Keep:** `config.py`, `claude_client.py` (Task pending), `frames.py` sampling idea,
  `report.py` shape — all adapt directly.
- **Retire (Sharingan-specific):** `study/actions.py` and `study/metric.py` (typed-action
  P/R) + their tests — removed; MCQA needs accuracy, not action matching.
- **Keep but unused on critical path:** `study/embedder.py` (harmless; may serve a future
  free-form arm). Left in repo.

## 8. Deliverables

- Reproducible harness under `study/`.
- Frozen `manifest.json` (sampled desktop-domain MCQA set).
- Results: **controlled accuracy table** (Arm A vs Arm B, overall + per scenario) with
  **token cost**; optional McNemar significance.
- Short writeup for the demo paper.

## 9. Known Caveats (state in paper)

- We sample a subset of GUI-World, not the full benchmark; the sampled set is frozen and
  reported.
- Base model is Claude (not the GPT-4V/Gemini baselines in the GUI-World paper), so
  cross-paper numbers are indicative; the **controlled arm (same model both sides)** is the
  rigorous claim.
- GUI-World has a noted dataset-viewer schema inconsistency (some QA fields vary
  object/array); the loader normalizes fields defensively and skips malformed items,
  logging how many were dropped.
- Frame budget for Arm A (N frames) is a knob; it is fixed and recorded, and a small sweep
  may be reported to show the comparison isn't budget-cherrypicked.

## 10. Out of Scope (YAGNI)

- Free-form / captioning / conversation QA and the LLM-judge.
- Mobile and XR scenarios.
- Fine-tuning or any model training (no GUI-Vid reproduction).

## References

- GUI-World: A Video Benchmark and Dataset for Multimodal GUI-oriented Understanding —
  arXiv 2406.10819 (ICLR 2025). Data: `huggingface.co/datasets/shuaishuaicdp/GUI-World`;
  code: `github.com/Dongping-Chen/GUI-World`.
