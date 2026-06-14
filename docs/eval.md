# Squint evaluation

The thesis: Claude can answer "what happened?" accurately by reading motion-selected ASCII
+ a few escalated images, far cheaper than reading every sampled frame as an image.

## Test clips
Use 3–5 short clips (5–15s) with a known ground-truth event:
- a person walking into frame,
- a screen showing a specific word/error,
- an object moving across a static scene.

## Accuracy
For each clip, ask Claude (via the skill) the event question and record whether it:
1. detects the event, and
2. timestamps it within ±1s.

## Cost (the number that matters)
Run two conditions on the same clips and compare total input tokens:
- **Squint path:** the SKILL.md loop (manifest + ASCII skim + a few escalated PNGs).
- **Baseline:** `Read` every sampled PNG as an image, no ASCII.

Record tokens for each. The win is the ratio (Squint tokens / baseline tokens) at equal or
better accuracy. Log results in a table here as you run them.
