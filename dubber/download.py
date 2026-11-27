"""Завантаження відео за посиланням (YouTube та ін.) через yt-dlp."""

from __future__ import annotations

from pathlib import Path


def download_video(url: str, out_dir: str | Path = ".") -> Path:
    try:
        import yt_dlp
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("Встановіть yt-dlp: pip install yt-dlp") from e

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    opts = {
        "format": "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/bv*+ba/b",
        "merge_output_format": "mp4",
        "outtmpl": str(out_dir / "%(title).80s [%(id)s].%(ext)s"),
        "noplaylist": True,
        "quiet": False,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        path = Path(ydl.prepare_filename(info)).with_suffix(".mp4")
    if not path.exists():
        raise RuntimeError(f"Не вдалося знайти завантажений файл: {path}")
    print(f"Завантажено: {path}")
    return path
