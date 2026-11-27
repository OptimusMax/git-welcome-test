"""Повний конвеєр дубляжу."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from . import media
from .segments import Segment, merge_short_segments, read_srt, write_srt
from .timeline import assemble, plan_timeline


@dataclass
class DubOptions:
    target_lang: str = "uk"
    source_lang: str | None = None  # None — автовизначення
    voice: str | None = None
    female: bool = False
    rate: str = "+0%"
    whisper_model: str = "small"
    device: str = "auto"
    vad: bool = True  # відсікати немовні фрагменти (музику, шум)
    max_speed: float = 1.35
    original_volume: float = 0.12  # гучність фону оригіналу (0 — вимкнути)
    keep_original_track: bool = False
    embed_subtitles: bool = True
    input_srt: Path | None = None  # готові субтитри оригіналу (без розпізнавання)
    translated_srt: Path | None = None  # готовий переклад (без перекладу)
    work_dir: Path | None = None


def dub_video(
    video: str | Path,
    output: str | Path,
    opts: DubOptions,
    log: Callable[[str], None] = print,
) -> Path:
    video, output = Path(video), Path(output)
    if not video.exists():
        raise FileNotFoundError(f"Файл не знайдено: {video}")

    tmp_ctx = tempfile.TemporaryDirectory(prefix="dubber_") if opts.work_dir is None else None
    work = Path(tmp_ctx.name) if tmp_ctx else Path(opts.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    try:
        total = media.probe_duration(video)
        log(f"Відео: {video.name}, тривалість {total:.1f} с")

        # 1. Сегменти
        segments: list[Segment]
        source_lang = opts.source_lang
        if opts.translated_srt:
            log(f"[1/5] Беру готовий переклад: {opts.translated_srt}")
            segments = [Segment(s.start, s.end, s.text, s.text) for s in read_srt(opts.translated_srt)]
        elif not opts.input_srt and output.with_suffix(".orig.srt").exists():
            cached = output.with_suffix(".orig.srt")
            log(f"[1/5] Знайдено вже розпізнаний текст, розпізнавання пропущено: {cached.name}")
            log("      (щоб розпізнати заново, видаліть цей файл)")
            segments = read_srt(cached)
        elif opts.input_srt:
            log(f"[1/5] Беру субтитри оригіналу: {opts.input_srt}")
            segments = read_srt(opts.input_srt)
        else:
            from .transcribe import transcribe

            log(f"[1/5] Розпізнавання мовлення (whisper '{opts.whisper_model}')...")
            wav = media.extract_audio(video, work / "source.wav")
            segments, detected = transcribe(
                wav, opts.whisper_model, opts.source_lang, opts.device, opts.vad
            )
            source_lang = source_lang or detected
            log(f"      мова оригіналу: {detected}, сегментів: {len(segments)}")
            write_srt(segments, output.with_suffix(".orig.srt"), use_translation=False)

        if not segments:
            hint = (
                "Можливо, у відео немає мовлення (лише музика чи звуки). "
                "Якщо мова там є, спробуйте ще раз із параметрами --no-vad --model medium"
                if opts.vad and not (opts.input_srt or opts.translated_srt) else
                "Перевірте, чи у відео є мовлення, або спробуйте --model medium"
            )
            raise RuntimeError(f"Не знайдено жодної фрази для дубляжу. {hint}")

        # 2. Переклад
        if not opts.translated_srt:
            segments = merge_short_segments(segments)
            if source_lang and source_lang.split("-")[0] == opts.target_lang:
                log("[2/5] Мова оригіналу збігається з цільовою — переклад пропущено")
                segments = [s.with_translation(s.text) for s in segments]
            else:
                from .translate import google_translator, translate_segments

                log(f"[2/5] Переклад {source_lang or 'auto'} → {opts.target_lang}...")
                segments = translate_segments(
                    segments, google_translator(source_lang or "auto", opts.target_lang), log
                )
        else:
            log("[2/5] Переклад пропущено")
        srt_path = output.with_suffix(f".{opts.target_lang}.srt")
        write_srt(segments, srt_path)
        log(f"      субтитри: {srt_path}")

        # 3. Синтез голосу
        from .tts import default_voice, synthesize

        voice = opts.voice or default_voice(opts.target_lang, opts.female)
        log(f"[3/5] Синтез голосу ({voice})...")
        clip_paths = synthesize([s.translation for s in segments], work / "tts", voice, opts.rate)

        # 4. Синхронізація
        log("[4/5] Синхронізація з відео...")
        sr = media.SAMPLE_RATE
        raw = [media.load_audio(p, sr) if p else None for p in clip_paths]
        durations = [len(c) / sr if c is not None else 0.0 for c in raw]
        placements = plan_timeline(segments, durations, total, opts.max_speed)
        clips = [
            media.load_audio(p, sr, tempo=pl.tempo) if p and pl.tempo != 1.0 else c
            for p, c, pl in zip(clip_paths, raw, placements)
        ]
        sped = sum(1 for pl in placements if pl.tempo > 1.0)
        drift = max((pl.start - s.start for pl, s in zip(placements, segments)), default=0.0)
        log(f"      прискорено фраз: {sped}/{len(segments)}, макс. зсув: {drift:.2f} с")
        dub_wav = media.save_wav(assemble(clips, placements, sr, total), work / "dub.wav", sr)

        # 5. Зведення
        log("[5/5] Зведення відео...")
        lang3 = {"uk": "ukr", "en": "eng", "de": "deu", "fr": "fra", "es": "spa", "pl": "pol", "it": "ita"}
        media.mux(
            video, dub_wav, output,
            original_volume=opts.original_volume,
            keep_original_track=opts.keep_original_track,
            subtitles=srt_path if opts.embed_subtitles else None,
            language=lang3.get(opts.target_lang, opts.target_lang),
        )
        log(f"Готово: {output}")
        return output
    finally:
        if tmp_ctx:
            tmp_ctx.cleanup()
