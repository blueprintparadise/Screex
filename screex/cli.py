from __future__ import annotations

import argparse
from pathlib import Path

from screex.core import source, mapper, analyzer
from screex.core.manifest import Manifest, FrameRecord


def analyze(video, fps=5.0, cols=120, sensitivity=0.06, edge=False, out=None, cut_threshold=0.5):
    import cv2

    video = Path(video)
    info = source.video_info(str(video))
    out_dir = Path(out) if out else video.parent / f"{video.stem}.screex"
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    records = []
    similarities = []
    prev_gray = None
    for idx, t, bgr in source.iter_frames(str(video), fps):
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        score = 0.0 if prev_gray is None else analyzer.motion_score(prev_gray, gray)
        sim = 1.0 if prev_gray is None else analyzer.histogram_similarity(prev_gray, gray)
        prev_gray = gray
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

    scores = [r.score for r in records]
    times = [r.t for r in records]
    flags = analyzer.flag_events(scores, sensitivity)
    for r, f in zip(records, flags):
        r.event = f
    events = analyzer.group_events(scores, times, flags)
    events = analyzer.classify_events(events, similarities, cut_threshold)

    manifest = Manifest(
        video=video.name, duration=round(info["duration"], 3),
        sampled_fps=fps, cols=cols, frames=records, events=events,
    )
    manifest_path = out_dir / "manifest.json"
    manifest.save(manifest_path)
    return manifest_path


def index(recording, fps=2.0, change_threshold=0.04, thumb_width=320, out=None):
    import cv2
    from screex.core import source, segment, ocr
    from screex.core.index import ScreenState, ScreenIndex

    recording = Path(recording)
    info = source.video_info(str(recording))
    out_dir = Path(out) if out else recording.parent / f"{recording.stem}.screex"
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    states = []
    prev_ocr = []
    for seg in segment.segment_stream(source.iter_frames(str(recording), fps), change_threshold):
        bgr = seg.frame_bgr
        text = ocr.extract_text(bgr)
        added, removed = ocr.text_diff(prev_ocr, text)
        prev_ocr = text

        name = f"{seg.idx:05d}"
        key_rel = f"frames/{name}.png"
        thumb_rel = f"frames/{name}_thumb.png"
        cv2.imwrite(str(out_dir / key_rel), bgr)
        th = max(1, int(bgr.shape[0] * thumb_width / bgr.shape[1]))
        cv2.imwrite(str(out_dir / thumb_rel), cv2.resize(bgr, (thumb_width, th)))

        states.append(ScreenState(
            idx=seg.idx, t_start=round(seg.t_start, 3), t_end=round(seg.t_end, 3),
            thumbnail=thumb_rel, keyframe=key_rel,
            ocr_text=text, text_added=added, text_removed=removed,
        ))

    screen_index = ScreenIndex(
        video=recording.name, duration=round(info["duration"], 3),
        sampled_fps=fps, states=states,
    )
    index_path = out_dir / "index.json"
    screen_index.save(index_path)
    return index_path


def main(argv=None):
    p = argparse.ArgumentParser(prog="screex")
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

    ix = sub.add_parser("index", help="build a ScreenIndex from a screen recording")
    ix.add_argument("recording")
    ix.add_argument("--fps", type=float, default=2.0, help="frames sampled per second")
    ix.add_argument("--change-threshold", type=float, default=0.04,
                    help="motion fraction (0..1) that marks a new UI state")
    ix.add_argument("--thumb-width", type=int, default=320, help="thumbnail width in px")
    ix.add_argument("--out", default=None, help="output dir (default <recording>.screex)")

    c = sub.add_parser("capture", help="record a short webcam clip")
    c.add_argument("--webcam", action="store_true")
    c.add_argument("--seconds", type=float, default=10.0)
    c.add_argument("--out", default="capture.mp4")

    args = p.parse_args(argv)
    if args.cmd == "analyze":
        path = analyze(args.video, fps=args.fps, cols=args.cols,
                       sensitivity=args.sensitivity, edge=args.edge, out=args.out,
                       cut_threshold=args.cut_threshold)
        print(f"manifest: {path}")
    elif args.cmd == "index":
        path = index(args.recording, fps=args.fps, change_threshold=args.change_threshold,
                     thumb_width=args.thumb_width, out=args.out)
        print(f"index: {path}")
    elif args.cmd == "capture":
        out = source.capture_webcam(args.out, args.seconds)
        print(f"captured: {out}")


if __name__ == "__main__":
    main()
