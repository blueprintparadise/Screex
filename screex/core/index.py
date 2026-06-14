from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class ScreenState:
    idx: int
    t_start: float
    t_end: float
    thumbnail: str
    keyframe: str
    ocr_text: list = field(default_factory=list)
    text_added: list = field(default_factory=list)
    text_removed: list = field(default_factory=list)


SCHEMA_VERSION = 1


@dataclass
class ScreenIndex:
    video: str
    duration: float
    sampled_fps: float
    states: list = field(default_factory=list)
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "video": self.video,
            "duration": self.duration,
            "sampled_fps": self.sampled_fps,
            "states": [asdict(s) for s in self.states],
        }

    @classmethod
    def from_dict(cls, d: dict) -> ScreenIndex:
        return cls(
            video=d["video"],
            duration=d["duration"],
            sampled_fps=d["sampled_fps"],
            states=[ScreenState(**s) for s in d["states"]],
            schema_version=d.get("schema_version", SCHEMA_VERSION),
        )

    def save(self, path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path) -> ScreenIndex:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
