# Screex improvement roadmap — accuracy-first

**Date:** 2026-06-17
**North star:** make the OCR/keyframe index **as accurate as raw sampled frames AND cheaper** —
i.e. *beat* raw frames, not merely undercut them on cost.

## Evidence base (GUI-World MCQA benchmark, `screex-research`)

A controlled benchmark on **GUI-World** (ICLR 2025), 300-item desktop-domain MCQA subset,
same Claude model (`claude-opus-4-8`) for both arms; only the perception input varies.

Published results (286 scored items):

| Arm | Accuracy | Input tokens | Cost |
|-----|----------|--------------|------|
| Raw frames (uniform 8) | **90.2%** | 4.69M | $23.46 |
| OCR/keyframe index | **85.0%** | 2.59M | **$12.99** |

The index is **~2× cheaper but ~5 points less accurate.**

**Frame-budget sweep (the key lever):** more frames make the model *worse* —

| frames | accuracy |
|--------|----------|
| 4 | **88.3%** |
| 8 | 85.0% |
| 16 | 83.3% |

Uniform sampling dilutes signal with redundant frames. Screex segments video into *settled,
meaningful* UI states, so its keyframes are **curated, not uniform** — the opening to beat the
frames arm.

**Latency:** median 50.5s/video, p90 135s on CPU (i7-1355U); their note: *"MX550 GPU present
but unused."* Indexing speed is the secondary pain.

**Current-Screex pilot (this session, no API):** the current pipeline builds clean on real
GUI-World videos — 11/12 pilot videos OK (1 corrupt download), mean 6.5 states, ~2,873
tokens/index, ~38s/video build on CPU. The published run used the *older vendored `videx`*, so
current Screex's `interactions`, `boxes`, and `compact_dict` were never measured.

## Priorities (accuracy-first)

### Lead bet — A2 · Curated hybrid index (text + N curated keyframes)
Attach a small budget of keyframes from the most-changed settled states to the text index.
Exploits "fewer, better frames win": curated-few should beat uniform-eight, while keeping the
cost/curation advantage. Attacks the frames arm on its home turf (pixels). **Needs its own
spec/brainstorm.** Impact: highest · Effort: med.

### A1 · Typed action events (v0.6) — already specced
Promote text-diffs into `navigate/type/click/open_dialog/error/scroll`, region-grounded. Adds
the action/temporal signal that static frames blur; targets the action-MCQs where the index
loses. Spec: `docs/superpowers/specs/2026-06-17-screex-v0.6-action-events-design.md`.
**Ship-ready.** Impact: high · Effort: med.

### A3 · Enable `interactions` + `boxes` in the eval
Already built, never benchmarked — spatial grounding for "where/what did they click." Impact:
med · Effort: low.

### A4 · Multi-frame OCR voting
OCR 2–3 frames per settled state and merge to cut garbled/false lines → more trustworthy
answers. Impact: med · Effort: med.

### E2 (+ minimal E1) · Per-question-type accuracy harness
A slim in-repo benchmark that buckets MCQs (action / state / count / visual) so each accuracy
fix is *proven* to move the right bucket. Just enough measurement to validate the bets — not
"measure everything first." Impact: high (proves the thesis) · Effort: med.

## Deprioritized (still on the roadmap, after accuracy)

These don't move accuracy, so they wait — but get measured for free once E2 exists.

- **Theme B — speed/scale:** B1 region-OCR (v0.6.1), B2 parallel OCR (`--workers`), B3 GPU OCR
  provider (the unused MX550!), B4 adaptive sampling.
- **Theme C — cost:** C1 benchmark `compact_dict` (already in current Screex, never measured),
  C2 TOON/structured compression.
- **Theme D — robustness/coverage:** D1 corrupt/truncated-video handling (pilot hit "moov atom
  not found"; published run excluded 14/300), D2 multi-window scenario (re-enable once hybrid
  keyframes + char-cap land).

## Execution order

1. **A1 (events)** — ship first; spec already exists.
2. **E2 (+ minimal E1)** — stand up measurement to validate A1 and everything after.
3. **A2 (curated hybrid keyframes)** — the bet that flips index > frames; brainstorm a spec.
4. **A3 / A4** — accuracy top-ups.
5. Then Theme B/C/D, each now measurable.

Throughline: **A1 is already designed, and E2 makes every accuracy claim provable.**
