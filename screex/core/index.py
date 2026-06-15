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
    boxes: list = field(default_factory=list)  # [{"text": str, "box": [x, y, w, h]}]
    interactions: list = field(default_factory=list)  # [{"t", "x", "y", "label"}]


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
    states: list = field(default_factory=list)
    narration: list = field(default_factory=list)
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "video": self.video,
            "duration": self.duration,
            "sampled_fps": self.sampled_fps,
            "states": [asdict(s) for s in self.states],
            "narration": [asdict(n) for n in self.narration],
        }

    def compact_dict(self, drop_diffs=True, factor_persistent=True,
                     drop_boxes=True, drop_interactions=True) -> dict:
        """A compact, LLM-oriented view of the index that cuts token cost without losing
        on-screen text. Two large redundancies dominate the verbose index: each state ships
        both its full OCR snapshot and the add/removed diffs (mutually derivable), and UI
        chrome (menus, toolbars, taskbar) repeats in every state.

        - ``drop_diffs``: drop per-state ``text_added``/``text_removed`` (recoverable by
          comparing consecutive ``ocr_text``).
        - ``factor_persistent``: hoist OCR lines present in *every* state into a single
          top-level ``persistent_ui`` block, leaving each state only its unique text. Lossless
          at the text-line-set level: a state's original line set is ``persistent_ui`` united
          with its remaining ``ocr_text``.
        - ``drop_boxes``/``drop_interactions``: omit per-line OCR coordinates and cursor
          hotspots — auxiliary metadata not needed for text-level understanding (the text in
          ``boxes`` already appears in ``ocr_text``).

        The default verbose ``to_dict`` is unchanged; this is an additive, opt-in view.
        """
        d = self.to_dict()
        states = d["states"]
        for s in states:
            if drop_diffs:
                s.pop("text_added", None)
                s.pop("text_removed", None)
            if drop_boxes:
                s.pop("boxes", None)
            if drop_interactions:
                s.pop("interactions", None)
        if factor_persistent and states:
            line_sets = [set(s.get("ocr_text", [])) for s in states]
            persistent = set.intersection(*line_sets) if line_sets else set()
            if persistent:
                d["persistent_ui"] = sorted(persistent)
                for s in states:
                    s["ocr_text"] = [ln for ln in s.get("ocr_text", []) if ln not in persistent]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> ScreenIndex:
        return cls(
            video=d["video"],
            duration=d["duration"],
            sampled_fps=d["sampled_fps"],
            states=[ScreenState(**s) for s in d["states"]],
            narration=[NarrationSegment(**n) for n in d.get("narration", [])],
            schema_version=d.get("schema_version", SCHEMA_VERSION),
        )

    def save(self, path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path) -> ScreenIndex:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
