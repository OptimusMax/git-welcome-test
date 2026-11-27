"""Командний рядок: dubber input.mp4 -o output.mp4"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from .pipeline import DubOptions, dub_video


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="dubber",
        description="Автоматичний дубляж відео: розпізнавання → переклад → озвучення → зведення.",
    )
    p.add_argument("video", help="вхідне відео (файл або посилання YouTube тощо)")
    p.add_argument("-o", "--output", type=Path, help="вихідне відео (за замовч. <назва>.<мова>.mp4)")
    p.add_argument("-t", "--target", default="uk", help="мова дубляжу (uk, en, de, ...), за замовч. uk")
    p.add_argument("-s", "--source", default=None, help="мова оригіналу (за замовч. автовизначення)")
    p.add_argument("--voice", help="голос Edge TTS, напр. uk-UA-PolinaNeural")
    p.add_argument("--female", action="store_true", help="жіночий голос за замовчуванням")
    p.add_argument("--rate", default="+0%", help="швидкість мовлення TTS, напр. +10%% або -5%%")
    p.add_argument("--model", default="small", help="модель whisper: tiny/base/small/medium/large-v3")
    p.add_argument("--device", default="auto", help="cpu / cuda / auto")
    p.add_argument("--max-speed", type=float, default=1.35, help="макс. прискорення фрази (за замовч. 1.35)")
    p.add_argument("--bg-volume", type=float, default=0.12,
                   help="гучність оригінального звуку під дубляжем, 0..1 (0 — тільки дубляж)")
    p.add_argument("--keep-original", action="store_true", help="зберегти оригінальну доріжку як другу")
    p.add_argument("--no-subs", action="store_true", help="не вбудовувати субтитри у відео")
    p.add_argument("--srt", type=Path, help="готові субтитри оригіналу (пропустити розпізнавання)")
    p.add_argument("--translated-srt", type=Path, help="готові перекладені субтитри (лише озвучення)")
    p.add_argument("--work-dir", type=Path, help="зберегти проміжні файли в цю папку")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if re.match(r"https?://", args.video):
        from .download import download_video

        try:
            args.video = download_video(args.video, Path(args.work_dir or "."))
        except Exception as e:
            print(f"Помилка завантаження: {e}", file=sys.stderr)
            return 1
    args.video = Path(args.video)
    output = args.output or args.video.with_name(f"{args.video.stem}.{args.target}{args.video.suffix or '.mp4'}")
    opts = DubOptions(
        target_lang=args.target,
        source_lang=args.source,
        voice=args.voice,
        female=args.female,
        rate=args.rate,
        whisper_model=args.model,
        device=args.device,
        max_speed=args.max_speed,
        original_volume=args.bg_volume,
        keep_original_track=args.keep_original,
        embed_subtitles=not args.no_subs,
        input_srt=args.srt,
        translated_srt=args.translated_srt,
        work_dir=args.work_dir,
    )
    try:
        dub_video(args.video, output, opts)
    except KeyboardInterrupt:
        return 130
    except Exception as e:
        print(f"Помилка: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
