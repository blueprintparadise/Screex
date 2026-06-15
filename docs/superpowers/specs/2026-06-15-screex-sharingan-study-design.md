# Screex on Sharingan (ACTREAL) — Benchmark Study Design

**Date:** 2026-06-15
**Status:** SUPERSEDED — Sharingan ships no public data or ground truth, forcing ACTREAL
reconstruction + self-generated GT (the spec's biggest risk). Replaced by the GUI-World
design (`2026-06-15-screex-guiworld-study-design.md`), which uses a dataset that ships
real, human-annotated MCQA ground truth. Kept for history.
**Topic:** Evaluate Screex's "cheap index → Claude reads it" pipeline on action-sequence
extraction from desktop recordings, using the Sharingan task and metric.

---

## 1. Goal & Claim

Evaluate Screex on **user-action-sequence extraction from desktop screen recordings**,
reusing the task definition and evaluation metric from **Sharingan: Extract User Action
Sequence from Desktop Recordings** (arXiv 2411.08768, Microsoft Research, 2024).

Two-part claim:

1. **Controlled (primary, rigorous):** with the *same Claude model* in both arms,
   *Screex's index beats raw frames* (Sharingan's Direct-Frame "DF" setup) on
   Precision/Recall — and does so at lower token cost. This is the defensible headline.
2. **Vs-published (secondary, indicative):** tabulate our Screex-arm Precision/Recall
   against Sharingan's *reported* DF/DiffF numbers (GPT-4o family) on ACTREAL, for
   context only. Not strictly apples-to-apples (different base model, reconstructed data,
   regenerated GT).

This directly tests Screex's core thesis (a cheap OCR + keyframe index is a better LLM
perception layer than raw frames) on a real, published action-extraction benchmark.

## 2. Data — Reconstructed ACTREAL

Sharingan's ACTREAL is **not publicly released**, but it is a documented filtered subset
of the public **GUI-World** dataset (arXiv 2406.10819; on Hugging Face). We reconstruct an
ACTREAL-equivalent:

- Source: GUI-World videos (YouTube-sourced screencasts).
- Sharingan's published filters:
  - categories **Software**, **Website**, **Multi**;
  - **6–10 actions** per video;
  - **≥ 3 unique action types**;
  - drop videos containing hover actions or non-input keyboard actions.
- Target size: **~41 videos** (Sharingan's reported ACTREAL count).
- Persist the **exact selected video IDs + a manifest** (source URL/HF id, category,
  duration, action count) so the set is fully reproducible.

**Caveat (documented in paper):** our selection will not be byte-identical to Sharingan's
original 41 videos, so the vs-published comparison is indicative, not exact.

## 3. Ground Truth — Claude-propose + Human-verify

Sharingan's ACTREAL ground-truth action sequences are not released, and GUI-World does not
ship GT in Sharingan's `(type, details, context)` form. We therefore create our own GT:

- A **GT generator** (a strong Claude model) is fed dense frames per video and proposes
  `(type, details, context)` action tuples.
- **Hard requirement:** the GT generator must use a configuration **distinct from the arms
  under test** (different prompt, and ideally dense-frame input rather than the
  windowed/index inputs the arms use) to avoid biasing GT toward either arm.
- Output is written to an **editable JSON** per video; a **human reviews and corrects**
  every video.
- GT is then **frozen and committed**. No scoring run may execute against unverified GT.

This is the study's principal fidelity risk and is gated on human verification.

## 4. The Two Arms

Both arms produce a `(type, details, context)` action sequence and use the **same Claude
model** (fixed and recorded). Both log token usage.

### Arm A — DF-equivalent baseline (raw frames → Claude)

- Sample frames at **2 fps** (Sharingan's ACTREAL rate).
- Window size **10 frames**, overlap **5** (Sharingan DF config for the GPT series).
- Claude pipeline mirrors Sharingan DF's three modules: **Action Proposer → Action
  Corrector → Action Merger** (chain-of-thought).

### Arm B — Screex (index → Claude)

- Run `screex index` on the video (OCR text + text-diff + keyframes/thumbnails per
  UI state).
- Claude reads the Screex index/transcript and emits the same `(type, details, context)`
  action sequence.

Both arms cache their model outputs to disk so the metric is re-scorable without
re-calling Claude.

## 5. Metric — Faithful Sharingan Reimplementation

Action representation: `(O = operation type, D = operation details, C = context/app)`.

- **Operation type O:** one of `click, select, scroll, drag, type`. Matched **exactly**
  (binary).
- **Details D & Context C:** matched via **BERT embedding cosine similarity ≥ 0.70**.
- **Matching algorithm:** iterate ground-truth actions in chronological order; for each GT
  action, take the **first unmatched predicted action** that aligns (greedy chronological).
- **Reported metrics:**
  - Recall = matched pairs / total GT actions
  - Precision = matched pairs / total predicted actions
  - Two levels: **Operation** (type only) and **All** (all three components).
- Aggregate as mean (± spread) across videos.

## 6. Harness Layout

A standalone, reproducible harness that imports `screex` as a library. Heavy eval-only
dependencies (datasets, sentence-transformers/BERT) stay **out** of the shipped pip
package.

```
study/
  data_prep.py     # GUI-World -> filtered ACTREAL manifest + video cache
  gt_generate.py   # Claude-propose GT  ->  gt/*.json  (then human-edited)
  arms/df.py       # 2fps frames -> Claude -> actions (DF-equivalent baseline)
  arms/screex.py   # screex index -> Claude -> actions
  metric.py        # Sharingan P/R (BERT >=0.7, greedy chronological)
  run.py           # orchestrate arms over manifest, cache outputs
  report.py        # aggregate tables (controlled + vs-published) + token costs
  requirements.txt # eval-only deps
  manifest.json    # frozen reconstructed-ACTREAL video set
  gt/              # frozen, human-verified ground truth (committed)
  cache/           # frames, indexes, model outputs (gitignored)
```

Intermediate artifacts (sampled frames, Screex indexes, raw model outputs) are cached to
disk so reruns are cheap and the metric can be re-applied without re-calling Claude.

## 7. Deliverables

- Reproducible harness under `study/`.
- Frozen `manifest.json` + human-verified `gt/`.
- Results:
  - **Controlled table:** Arm A vs Arm B — Precision/Recall (Operation + All) + token cost.
  - **Vs-published context table:** Screex arm vs Sharingan's reported DF/DiffF (GPT-4o).
- A short results writeup suitable for the demo paper.

## 8. Known Caveats (to state explicitly in the paper)

- Reconstructed ACTREAL ≠ Sharingan's original 41 videos (selection drift).
- Ground truth is LLM-assisted + human-verified, not Sharingan's original GT.
- Base model is Claude, not the GPT-4o family — so **vs-published is indicative**; the
  **controlled arm (same model both sides) is the rigorous claim**.
- BERT-threshold matching (0.70) reproduces Sharingan's semantic-match design but is
  sensitive to the embedding model choice; the exact model is fixed and recorded.

## 9. Out of Scope (YAGNI)

- ACTONE (self-curated OBS dataset) — not reconstructed for this study.
- RPA replayability validation — Sharingan's optional functional check; omitted unless the
  semantic results warrant it.
- The deterministic (no-LLM) state-diff→action layer — considered and deferred; this study
  uses the Screex-index → Claude pipeline only.

## References

- Sharingan: Extract User Action Sequence from Desktop Recordings — arXiv 2411.08768
- GUI-World: A Video Benchmark and Dataset for Multimodal GUI-oriented Understanding —
  arXiv 2406.10819
