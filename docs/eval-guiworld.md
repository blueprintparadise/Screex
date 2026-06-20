# GUI-World benchmark run (publishing real accuracy + cost numbers)

This runbook turns the in-repo accuracy harness (`scripts/eval.py --qa`) into a reproducible
comparison on a real benchmark — **GUI-World** (ICLR 2025) — to test Screex's thesis: a *curated
few* keyframes beat *uniform-N* frames at lower cost. The actual run needs an Anthropic API budget;
everything else (dataset conversion, commands) is prepared here.

## Methodology

- **Same model both arms** (`claude-opus-4-8`); only the *perception input* varies. This isolates
  perception, mirroring the published roadmap experiment.
- **Two arms:**
  - *Frames* — uniform `N` sampled frames as images (`--frames N`, no `--keyframe-budget`).
  - *Curated hybrid index* — compact index text **plus** the `N` most salient curated keyframes
    (`--keyframe-budget N`, the A2 feature surfaced through the hybrid index arm).
- **Report accuracy AND tokens per question bucket** (action/state/count/visual) — the F2C/EVS
  template. Cost is only credible alongside accuracy.

Roadmap baseline to beat (GUI-World desktop MCQA, `claude-opus-4-8`):

| Arm | Accuracy | Input tokens |
|-----|----------|--------------|
| Raw frames (uniform 8) | 90.2% | 4.69M |
| OCR/keyframe index (text only) | 85.0% | 2.59M |

…and the key lever — *more uniform frames hurt*: 4f **88.3%** > 8f 85.0% > 16f 83.3%. So the bet is
that a small curated keyframe budget added to the text index clears uniform-8 at lower cost.

## Setup

1. **Get the dataset** (HF `shuaishuaicdp/GUI-World`) — clips + annotation JSON. Keep them **outside
   the repo** (or git-ignored): GUI-World's license is unspecified upstream, so do not commit clips
   or the generated `qa.jsonl`, and verify licensing before redistributing any derived data.
2. **Install the eval extra and set a key:**
   ```bash
   pip install -e '.[eval]'
   export ANTHROPIC_API_KEY=sk-...
   ```
3. **Build a desktop MCQ set** (sized to your budget) with the converter:
   ```bash
   python scripts/guiworld_to_qa.py /data/guiworld/annotations.json \
       -o qa.jsonl --domain desktop --limit 100
   ```
   (`type` buckets are an approximate keyword heuristic — see the converter docstring.)

## Run matrix

Start tiny to validate end-to-end before scaling:

```bash
# smoke (a few items, both arms)
python scripts/eval.py --qa qa.jsonl --clips-dir /data/guiworld/clips \
    --answerer claude --limit-by-editing-qa-first --keyframe-budget 4 --json   # see note*
```
\* there is no `--limit` on the eval itself; size the run by trimming `qa.jsonl` (e.g.
`head -n 20 qa.jsonl > qa20.jsonl`).

Frames-arm sweep:
```bash
for N in 4 8 16; do
  python scripts/eval.py --qa qa.jsonl --clips-dir /data/guiworld/clips \
      --answerer claude --frames $N --json > frames_$N.json
done
```

Curated-hybrid sweep:
```bash
for B in 2 4 8; do
  python scripts/eval.py --qa qa.jsonl --clips-dir /data/guiworld/clips \
      --answerer claude --keyframe-budget $B --json > curated_$B.json
done
```

Each run prints (and `--json` archives) the per-bucket table:

```
| bucket | n | index_acc | frames_acc | index_tokens | frames_tokens |
```

## Cost control

Rough per-run cost ≈ `items × (index_call + frames_call) × tokens × Opus input price`. The frames
arm dominates tokens (≈`N × 1500` per question). Concretely: 100 items × an 8-image arm ≈ 100 × 8 ×
1500 ≈ 1.2M input tokens for that arm alone — so **start with ~20 items**, confirm the table looks
sane, then scale. Use `tokens_per_image`/`--frames` to bound spend.

## Recording results

Add a row per arm/budget to the `docs/eval.md` **Results Log** (extend it with accuracy columns):
model id, date, subset size + domain, commit SHA, per-bucket accuracy, and index-vs-frames tokens.
Keep raw `*.json` outputs out of the repo unless licensing is cleared.

## Success criterion

A curated-hybrid budget where **index accuracy ≥ uniform-8 at fewer tokens**, ideally lifting the
`visual`/`count` buckets where text-only loses. Report honestly if it does **not** beat frames — a
negative result still tells the project where curation needs work (it's why the harness exists).
