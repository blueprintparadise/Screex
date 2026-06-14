from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from screex import __version__
from screex.core import analyzer, mapper, source
from screex.core.manifest import FrameRecord, Manifest


def _log(quiet: bool, msg: str) -> None:
    if not quiet:
        print(msg, file=sys.stderr, flush=True)


def _clean_frames_dir(frames_dir: Path) -> None:
    """Remove stale per-state images so a re-run never leaves orphans behind."""
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)


def analyze(video, fps=5.0, cols=120, sensitivity=0.06, edge=False, out=None,
            cut_threshold=0.5, max_frames=None, quiet=False):
    import cv2

    video = Path(video)
    info = source.video_info(str(video))
    out_dir = Path(out) if out else video.parent / f"{video.stem}.screex"
    frames_dir = out_dir / "frames"
    _clean_frames_dir(frames_dir)

    records = []
    similarities = []
    prev_gray = None
    last_t = 0.0
    for idx, t, bgr in source.iter_frames(str(video), fps, max_frames=max_frames):
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        score = 0.0 if prev_gray is None else analyzer.motion_score(prev_gray, gray)
        sim = 1.0 if prev_gray is None else analyzer.histogram_similarity(prev_gray, gray)
        prev_gray = gray
        last_t = t
        similarities.append(sim)

        ascii_text = mapper.frame_to_ascii(gray, cols, edge=edge)
        name = f"{idx:05d}"
        png_rel = f"frames/{name}.png"
        txt_rel = f"frames/{name}.txt"
        cv2.imwrite(str(out_dir / png_rel), bgr)
        (out_dir / txt_rel).write_text(ascii_text, encoding="utf-8")

        records.append(FrameRecord(
            idx=idx, t=round(t, 3), score=round(score, 4),
            event=False, ascii=txt_rel, png=png_rel,
        ))
        if not quiet and idx % 25 == 0:
            _log(quiet, f"analyze: {idx + 1} frames...")

    if not records:
        raise ValueError(f"no frames decoded from {video} (empty or unreadable video?)")

    scores = [r.score for r in records]
    times = [r.t for r in records]
    flags = analyzer.flag_events(scores, sensitivity)
    for r, f in zip(records, flags):
        r.event = f
    events = analyzer.group_events(scores, times, flags)
    events = analyzer.classify_events(events, similarities, cut_threshold)

    duration = max(info["duration"], last_t)
    manifest = Manifest(
        video=video.name, duration=round(duration, 3),
        sampled_fps=fps, cols=cols, frames=records, events=events,
    )
    manifest_path = out_dir / "manifest.json"
    manifest.save(manifest_path)
    _log(quiet, f"analyze: {len(records)} frames, {len(events)} events")
    return manifest_path


def index(recording, fps=2.0, change_threshold=0.04, thumb_width=320, out=None,
          max_frames=None, keyframe_format="png", keyframe_quality=90,
          dedupe_threshold=0.95, lang=None, quiet=False):
    import cv2

    from screex.core import ocr, segment
    from screex.core.index import ScreenIndex, ScreenState

    recording = Path(recording)
    info = source.video_info(str(recording))
    out_dir = Path(out) if out else recording.parent / f"{recording.stem}.screex"
    frames_dir = out_dir / "frames"
    _clean_frames_dir(frames_dir)

    ext = keyframe_format.lower().lstrip(".")
    if ext not in ("png", "jpg", "jpeg"):
        raise ValueError(f"unsupported keyframe format: {keyframe_format} (use png or jpg)")
    write_params = [cv2.IMWRITE_JPEG_QUALITY, int(keyframe_quality)] if ext in ("jpg", "jpeg") else []

    states = []
    prev_ocr = []
    last_t = 0.0
    frames = source.iter_frames(str(recording), fps, max_frames=max_frames)
    for seg in segment.segment_stream(frames, change_threshold):
        bgr = seg.frame_bgr
        last_t = seg.t_end
        text = ocr.extract_text(bgr, lang=lang)

        # Merge near-identical consecutive UI states (cheap dedup) instead of emitting
        # a new state + extra image files for what is effectively the same screen.
        if states and ocr.text_similarity(states[-1].ocr_text, text) >= dedupe_threshold:
            states[-1].t_end = round(seg.t_end, 3)
            _log(quiet, f"index: merged near-duplicate state at {seg.t_end:.1f}s")
            continue

        added, removed = ocr.text_diff(prev_ocr, text)
        prev_ocr = text

        name = f"{seg.idx:05d}"
        key_rel = f"frames/{name}.{ext}"
        thumb_rel = f"frames/{name}_thumb.{ext}"
        cv2.imwrite(str(out_dir / key_rel), bgr, write_params)
        th = max(1, int(bgr.shape[0] * thumb_width / bgr.shape[1]))
        cv2.imwrite(str(out_dir / thumb_rel), cv2.resize(bgr, (thumb_width, th)), write_params)

        states.append(ScreenState(
            idx=seg.idx, t_start=round(seg.t_start, 3), t_end=round(seg.t_end, 3),
            thumbnail=thumb_rel, keyframe=key_rel,
            ocr_text=text, text_added=added, text_removed=removed,
        ))
        _log(quiet, f"index: state {len(states)} @ {seg.t_start:.1f}-{seg.t_end:.1f}s "
                    f"({len(text)} text lines)")

    if not states:
        raise ValueError(f"no UI states produced from {recording} (empty or unreadable video?)")

    duration = max(info["duration"], last_t)
    screen_index = ScreenIndex(
        video=recording.name, duration=round(duration, 3),
        sampled_fps=fps, states=states,
    )
    index_path = out_dir / "index.json"
    screen_index.save(index_path)
    _log(quiet, f"index: {len(states)} states -> {index_path}")
    return index_path


def main(argv=None):
    p = argparse.ArgumentParser(prog="screex")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    p.add_argument("-q", "--quiet", action="store_true", help="suppress progress output")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("analyze", help="analyze a video into ASCII frames + manifest")
    a.add_argument("video")
    a.add_argument("--fps", type=float, default=5.0, help="frames sampled per second")
    a.add_argument("--cols", type=int, default=120, help="ASCII grid width")
    a.add_argument("--sensitivity", type=float, default=0.06,
                   help="motion threshold (0..1) for flagging event frames")
    a.add_argument("--edge", action="store_true", help="emphasize edges/structure")
    a.add_argument("--out", default=None, help="output dir (default <video>.screex)")
    a.add_argument("--cut-threshold", type=float, default=0.5,
                   help="histogram-similarity below which an event is a scene cut (0..1)")
    a.add_argument("--max-frames", type=int, default=None,
                   help="cap the number of sampled frames")

    ix = sub.add_parser("index", help="build a ScreenIndex from a screen recording")
    ix.add_argument("recording")
    ix.add_argument("--fps", type=float, default=2.0, help="frames sampled per second")
    ix.add_argument("--change-threshold", type=float, default=0.04,
                    help="mean frame-to-frame intensity change (0..1) that marks a new UI "
                         "state; also triggers on cumulative drift from the state anchor")
    ix.add_argument("--thumb-width", type=int, default=320, help="thumbnail width in px")
    ix.add_argument("--out", default=None, help="output dir (default <recording>.screex)")
    ix.add_argument("--max-frames", type=int, default=None,
                    help="cap the number of sampled frames")
    ix.add_argument("--keyframe-format", default="png", choices=["png", "jpg", "jpeg"],
                    help="image format for keyframes/thumbnails (jpg is much smaller)")
    ix.add_argument("--keyframe-quality", type=int, default=90,
                    help="JPEG quality 1..100 (only used with jpg)")
    ix.add_argument("--dedupe-threshold", type=float, default=0.95,
                    help="merge consecutive states whose on-screen text is at least this "
                         "similar (0..1); set to >1 to disable merging")
    ix.add_argument("--lang", default=None, help="OCR language hint (default: auto)")

    c = sub.add_parser("capture", help="record a short clip from the screen or webcam")
    c.add_argument("--screen", action="store_true", help="capture the screen (needs 'mss')")
    c.add_argument("--webcam", action="store_true", help="capture the default webcam")
    c.add_argument("--seconds", type=float, default=10.0)
    c.add_argument("--out", default="capture.mp4")

    sk = sub.add_parser("skill", help="install or locate the Screex Claude skill (SKILL.md)")
    sk.add_argument("--install", action="store_true",
                    help="copy the bundled SKILL.md into the skills dir (default action)")
    sk.add_argument("--dir", default=None,
                    help="target skills dir (default ~/.claude/skills/screex)")
    sk.add_argument("--path", action="store_true",
                    help="print the install target path without writing")

    args = p.parse_args(argv)
    quiet = getattr(args, "quiet", False)
    if args.cmd == "analyze":
        path = analyze(args.video, fps=args.fps, cols=args.cols,
                       sensitivity=args.sensitivity, edge=args.edge, out=args.out,
                       cut_threshold=args.cut_threshold, max_frames=args.max_frames,
                       quiet=quiet)
        print(f"manifest: {path}")
    elif args.cmd == "index":
        path = index(args.recording, fps=args.fps, change_threshold=args.change_threshold,
                     thumb_width=args.thumb_width, out=args.out, max_frames=args.max_frames,
                     keyframe_format=args.keyframe_format, keyframe_quality=args.keyframe_quality,
                     dedupe_threshold=args.dedupe_threshold, lang=args.lang, quiet=quiet)
        print(f"index: {path}")
    elif args.cmd == "capture":
        if args.screen:
            out = source.capture_screen(args.out, args.seconds)
        else:
            out = source.capture_webcam(args.out, args.seconds)
        print(f"captured: {out}")
    elif args.cmd == "skill":
        from screex import skill as skill_mod
        target_dir = Path(args.dir) if args.dir else skill_mod.default_skill_dir()
        if args.path:
            print(target_dir / "SKILL.md")
        else:
            target = skill_mod.install_skill(args.dir)
            print(f"installed skill: {target}")


if __name__ == "__main__":
    main()
