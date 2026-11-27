"""Розпізнавання мовлення (faster-whisper)."""

from __future__ import annotations

import os
from pathlib import Path

from . import media
from .segments import Segment


def transcribe(
    audio_path: str | Path,
    model_size: str = "small",
    language: str | None = None,
    device: str = "auto",
    vad: bool = True,
) -> tuple[list[Segment], str]:
    """Повертає (сегменти, мова_оригіналу)."""
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    try:
        from faster_whisper import WhisperModel
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("Встановіть faster-whisper: pip install faster-whisper") from e

    compute_type = "int8" if device in ("cpu", "auto") else "float16"
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    # Декодуємо через ffmpeg самі: так не залежимо від версії PyAV усередині faster-whisper
    audio = media.load_audio(audio_path, sample_rate=16_000)
    raw_segments, info = model.transcribe(
        audio, language=language, vad_filter=vad, beam_size=5
    )
    segments = [Segment(s.start, s.end, s.text.strip()) for s in raw_segments if s.text.strip()]
    return segments, info.language
