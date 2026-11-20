"""Сегменти мовлення та робота з SRT-субтитрами."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from pathlib import Path


@dataclass
class Segment:
    start: float  # секунди
    end: float  # секунди
    text: str
    translation: str = ""

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def with_translation(self, translation: str) -> "Segment":
        return replace(self, translation=translation)


def _fmt_time(seconds: float) -> str:
    ms = int(round(max(0.0, seconds) * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


_TIME_RE = re.compile(
    r"(\d+):(\d{2}):(\d{2})[,.](\d{1,3})\s*-->\s*(\d+):(\d{2}):(\d{2})[,.](\d{1,3})"
)


def _to_seconds(h: str, m: str, s: str, ms: str) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms.ljust(3, "0")) / 1000


def write_srt(segments: list[Segment], path: str | Path, use_translation: bool = True) -> None:
    lines: list[str] = []
    for i, seg in enumerate(segments, 1):
        text = seg.translation if use_translation and seg.translation else seg.text
        lines += [str(i), f"{_fmt_time(seg.start)} --> {_fmt_time(seg.end)}", text.strip(), ""]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def read_srt(path: str | Path) -> list[Segment]:
    content = Path(path).read_text(encoding="utf-8-sig").replace("\r\n", "\n")
    segments: list[Segment] = []
    for block in re.split(r"\n\s*\n", content.strip()):
        lines = [ln for ln in block.split("\n") if ln.strip()]
        for idx, line in enumerate(lines):
            m = _TIME_RE.search(line)
            if m:
                g = m.groups()
                text = " ".join(lines[idx + 1 :]).strip()
                if text:
                    segments.append(
                        Segment(_to_seconds(*g[:4]), _to_seconds(*g[4:]), text)
                    )
                break
    return segments


def merge_short_segments(
    segments: list[Segment], min_duration: float = 1.0, max_gap: float = 0.3, max_duration: float = 12.0
) -> list[Segment]:
    """Зливає надто короткі сусідні сегменти — переклад виходить природнішим."""
    merged: list[Segment] = []
    for seg in segments:
        if merged:
            prev = merged[-1]
            gap = seg.start - prev.end
            if (
                (prev.duration < min_duration or seg.duration < min_duration)
                and gap <= max_gap
                and seg.end - prev.start <= max_duration
            ):
                merged[-1] = Segment(prev.start, seg.end, f"{prev.text} {seg.text}".strip())
                continue
        merged.append(Segment(seg.start, seg.end, seg.text))
    return merged
