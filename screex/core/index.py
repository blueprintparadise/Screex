from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ScreenState:
    idx: int
    t_start: float
    t_end: float
    thumbnail: str
    keyframe: str
    ocr_text: list[str] = field(default_factory=list)
    text_added: list[str] = field(default_factory=list)
    text_removed: list[str] = field(default_factory=list)


@dataclass
class NarrationSegment:
    start: float
    end: float
    text: str


SCHEMA_VERSION = 1


@dataclass
class ScreenIndex:
    video: str
    duration: float
    sampled_fps: float
    states: list[ScreenState] = field(default_factory=list)
    narration: list[NarrationSegment] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "video": self.video,
            "duration": self.duration,
            "sampled_fps": self.sampled_fps,
            "states": [asdict(s) for s in self.states],
            "narration": [asdict(n) for n in self.narration],
            "warnings": self.warnings,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ScreenIndex:
        schema_version = d.get("schema_version", SCHEMA_VERSION)
        if schema_version > SCHEMA_VERSION:
            raise ValueError(
                f"unsupported ScreenIndex schema_version {schema_version}; "
                f"this package supports up to {SCHEMA_VERSION}"
            )
        required = ("video", "duration", "sampled_fps", "states")
        missing = [key for key in required if key not in d]
        if missing:
            raise ValueError(f"invalid ScreenIndex: missing {', '.join(missing)}")

        def _state(payload: dict[str, Any]) -> ScreenState:
            required_state = ("idx", "t_start", "t_end", "thumbnail", "keyframe")
            missing_state = [key for key in required_state if key not in payload]
            if missing_state:
                raise ValueError(f"invalid ScreenState: missing {', '.join(missing_state)}")
            return ScreenState(
                idx=payload["idx"],
                t_start=payload["t_start"],
                t_end=payload["t_end"],
                thumbnail=payload["thumbnail"],
                keyframe=payload["keyframe"],
                ocr_text=payload.get("ocr_text", []),
                text_added=payload.get("text_added", []),
                text_removed=payload.get("text_removed", []),
            )

        return cls(
            video=d["video"],
            duration=d["duration"],
            sampled_fps=d["sampled_fps"],
            states=[_state(s) for s in d["states"]],
            narration=[NarrationSegment(**n) for n in d.get("narration", [])],
            warnings=list(d.get("warnings", [])),
            schema_version=schema_version,
        )

    def save(self, path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path) -> ScreenIndex:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
