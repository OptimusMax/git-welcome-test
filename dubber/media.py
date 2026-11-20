"""Обгортки над ffmpeg/ffprobe та робота з PCM-аудіо через numpy."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

SAMPLE_RATE = 24_000


class MediaError(RuntimeError):
    pass


def _require(tool: str) -> str:
    path = shutil.which(tool)
    if not path:
        raise MediaError(f"Не знайдено '{tool}'. Встановіть ffmpeg: https://ffmpeg.org/download.html")
    return path


def run(cmd: list[str]) -> bytes:
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise MediaError(f"Помилка команди {' '.join(cmd[:3])}...:\n{proc.stderr.decode(errors='replace')[-2000:]}")
    return proc.stdout


def probe_duration(path: str | Path) -> float:
    out = run([_require("ffprobe"), "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)])
    return float(json.loads(out)["format"]["duration"])


def extract_audio(video: str | Path, out_wav: str | Path, sample_rate: int = 16_000) -> Path:
    run([_require("ffmpeg"), "-y", "-i", str(video), "-vn", "-ac", "1", "-ar", str(sample_rate),
         "-c:a", "pcm_s16le", str(out_wav)])
    return Path(out_wav)


def load_audio(path: str | Path, sample_rate: int = SAMPLE_RATE, tempo: float = 1.0) -> np.ndarray:
    """Декодує будь-який аудіофайл у моно float32, за потреби змінюючи темп без зміни тону."""
    cmd = [_require("ffmpeg"), "-v", "error", "-i", str(path)]
    filters = atempo_chain(tempo)
    if filters:
        cmd += ["-filter:a", filters]
    cmd += ["-f", "f32le", "-ac", "1", "-ar", str(sample_rate), "-"]
    return np.frombuffer(run(cmd), dtype=np.float32).copy()


def save_wav(samples: np.ndarray, path: str | Path, sample_rate: int = SAMPLE_RATE) -> Path:
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2").tobytes()
    proc = subprocess.run(
        [_require("ffmpeg"), "-y", "-v", "error", "-f", "s16le", "-ar", str(sample_rate), "-ac", "1",
         "-i", "-", str(path)],
        input=pcm, stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        raise MediaError(proc.stderr.decode(errors="replace"))
    return Path(path)


def atempo_chain(tempo: float) -> str:
    """ffmpeg atempo приймає 0.5..2.0 (у старих версіях), тому ланцюжимо фільтри."""
    if abs(tempo - 1.0) < 1e-3:
        return ""
    parts: list[str] = []
    t = tempo
    while t > 2.0:
        parts.append("atempo=2.0")
        t /= 2.0
    while t < 0.5:
        parts.append("atempo=0.5")
        t /= 0.5
    parts.append(f"atempo={t:.4f}")
    return ",".join(parts)


def mux(
    video: str | Path,
    dub_wav: str | Path,
    out_path: str | Path,
    original_volume: float = 0.0,
    keep_original_track: bool = False,
    subtitles: str | Path | None = None,
    language: str = "ukr",
) -> Path:
    """Зводить відео з дубльованою доріжкою.

    original_volume > 0 — підмішує оригінальний звук (фон) з цією гучністю.
    keep_original_track — додає оригінальну доріжку як другу аудіодоріжку.
    """
    ffmpeg = _require("ffmpeg")
    cmd = [ffmpeg, "-y", "-i", str(video), "-i", str(dub_wav)]
    if subtitles:
        cmd += ["-i", str(subtitles)]

    maps: list[str] = ["-map", "0:v:0"]
    if original_volume > 0:
        cmd += ["-filter_complex",
                f"[0:a:0]volume={original_volume}[bg];[1:a:0][bg]amix=inputs=2:duration=first:normalize=0[dub]"]
        maps += ["-map", "[dub]"]
    else:
        maps += ["-map", "1:a:0"]
    if keep_original_track:
        maps += ["-map", "0:a:0?"]
    if subtitles:
        maps += ["-map", "2:s:0"]

    cmd += maps + ["-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                   "-metadata:s:a:0", f"language={language}", "-metadata:s:a:0", "title=Dub"]
    if subtitles:
        sub_codec = "mov_text" if str(out_path).lower().endswith((".mp4", ".m4v", ".mov")) else "srt"
        cmd += ["-c:s", sub_codec, "-metadata:s:s:0", f"language={language}"]
    cmd.append(str(out_path))
    run(cmd)
    return Path(out_path)
