from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path


@dataclass
class FrameRecord:
    idx: int
    t: float
    score: float
    event: bool
    ascii: str
    png: str


@dataclass
class EventRecord:
    t_start: float
    t_end: float
    peak_frame: int
    peak_score: float
    type: str = "motion"
    confidence: float = 0.0


@dataclass
class Manifest:
    video: str
    duration: float
    sampled_fps: float
    cols: int
    frames: list = field(default_factory=list)
    events: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "video": self.video,
            "duration": self.duration,
            "sampled_fps": self.sampled_fps,
            "cols": self.cols,
            "frames": [asdict(f) for f in self.frames],
            "events": [asdict(e) for e in self.events],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Manifest":
        return cls(
            video=d["video"],
            duration=d["duration"],
            sampled_fps=d["sampled_fps"],
            cols=d["cols"],
            frames=[FrameRecord(**f) for f in d["frames"]],
            events=[EventRecord(**e) for e in d["events"]],
        )

    def save(self, path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "Manifest":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
