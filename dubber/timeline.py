"""Синхронізація озвучених фраз із таймкодами оригіналу."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .segments import Segment


@dataclass
class Placement:
    start: float  # де фраза починається в дубльованій доріжці, с
    tempo: float  # коефіцієнт прискорення (1.0 — без змін)
    duration: float  # тривалість після прискорення, с


def plan_timeline(
    segments: list[Segment],
    clip_durations: list[float],
    total_duration: float | None = None,
    max_speed: float = 1.35,
    gap_borrow: float = 0.6,
) -> list[Placement]:
    """Розраховує, коли і з яким темпом програвати кожну фразу.

    - Доступне вікно фрази: від її початку до початку наступної (але не довше,
      ніж кінець сегмента + gap_borrow секунд паузи).
    - Якщо фраза довша за вікно — прискорюємо, але не більше max_speed.
    - Якщо й після цього не вміщається — зсуваємо наступні фрази (без накладань).
    """
    placements: list[Placement] = []
    cursor = 0.0
    n = len(segments)
    for i, (seg, clip) in enumerate(zip(segments, clip_durations)):
        start = max(seg.start, cursor)
        next_start = segments[i + 1].start if i + 1 < n else (total_duration or float("inf"))
        window_end = min(next_start, seg.end + gap_borrow)
        window_end = max(window_end, seg.end)
        window = max(0.05, window_end - start)

        tempo = 1.0
        if clip > window:
            tempo = min(max_speed, clip / window)
        duration = clip / tempo
        placements.append(Placement(start, tempo, duration))
        cursor = start + duration + 0.05  # мінімальна пауза між фразами
    return placements


def assemble(
    clips: list[np.ndarray | None],
    placements: list[Placement],
    sample_rate: int,
    total_duration: float,
    fade_ms: float = 8.0,
) -> np.ndarray:
    """Розкладає фрази на тайм-лінії та повертає моно-доріжку float32."""
    end_needed = max(
        [total_duration] + [p.start + len(c) / sample_rate for c, p in zip(clips, placements) if c is not None]
    )
    track = np.zeros(int(np.ceil(end_needed * sample_rate)) + 1, dtype=np.float32)
    fade = int(sample_rate * fade_ms / 1000)
    for clip, p in zip(clips, placements):
        if clip is None or len(clip) == 0:
            continue
        clip = clip.astype(np.float32, copy=True)
        if fade and len(clip) > 2 * fade:
            ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
            clip[:fade] *= ramp
            clip[-fade:] *= ramp[::-1]
        i0 = int(round(p.start * sample_rate))
        track[i0 : i0 + len(clip)] += clip
    # Обрізаємо до довжини відео
    track = track[: int(np.ceil(total_duration * sample_rate))]
    peak = float(np.max(np.abs(track))) if track.size else 0.0
    if peak > 0.99:
        track *= 0.99 / peak
    return track
