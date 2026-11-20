"""Розпізнавання мовлення (faster-whisper)."""

from __future__ import annotations

from pathlib import Path

from .segments import Segment


def transcribe(
    audio_path: str | Path,
    model_size: str = "small",
    language: str | None = None,
    device: str = "auto",
) -> tuple[list[Segment], str]:
    """Повертає (сегменти, мова_оригіналу)."""
    try:
        from faster_whisper import WhisperModel
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("Встановіть faster-whisper: pip install faster-whisper") from e

    compute_type = "int8" if device in ("cpu", "auto") else "float16"
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    raw_segments, info = model.transcribe(
        str(audio_path), language=language, vad_filter=True, beam_size=5
    )
    segments = [Segment(s.start, s.end, s.text.strip()) for s in raw_segments if s.text.strip()]
    return segments, info.language
