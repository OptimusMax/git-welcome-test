"""Синтез мовлення (Microsoft Edge TTS — безкоштовні нейронні голоси)."""

from __future__ import annotations

import asyncio
from pathlib import Path

# Голоси за замовчуванням для поширених мов
DEFAULT_VOICES = {
    "uk": "uk-UA-OstapNeural",
    "en": "en-US-GuyNeural",
    "de": "de-DE-ConradNeural",
    "fr": "fr-FR-HenriNeural",
    "es": "es-ES-AlvaroNeural",
    "it": "it-IT-DiegoNeural",
    "pl": "pl-PL-MarekNeural",
}

FEMALE_VOICES = {
    "uk": "uk-UA-PolinaNeural",
    "en": "en-US-JennyNeural",
    "de": "de-DE-KatjaNeural",
    "fr": "fr-FR-DeniseNeural",
    "es": "es-ES-ElviraNeural",
    "it": "it-IT-ElsaNeural",
    "pl": "pl-PL-ZofiaNeural",
}


def default_voice(lang: str, female: bool = False) -> str:
    table = FEMALE_VOICES if female else DEFAULT_VOICES
    if lang not in table:
        raise ValueError(f"Немає голосу за замовчуванням для '{lang}', вкажіть --voice (див. `edge-tts --list-voices`)")
    return table[lang]


async def _synth_one(text: str, voice: str, out: Path, rate: str, retries: int = 3) -> None:
    import edge_tts

    for attempt in range(retries):
        try:
            await edge_tts.Communicate(text, voice, rate=rate).save(str(out))
            return
        except Exception:
            if attempt == retries - 1:
                raise
            await asyncio.sleep(1.5 * (attempt + 1))


async def _synth_all(items: list[tuple[str, Path]], voice: str, rate: str, concurrency: int) -> None:
    sem = asyncio.Semaphore(concurrency)

    async def worker(text: str, out: Path) -> None:
        async with sem:
            await _synth_one(text, voice, out, rate)

    await asyncio.gather(*(worker(t, p) for t, p in items))


def synthesize(
    texts: list[str], out_dir: str | Path, voice: str, rate: str = "+0%", concurrency: int = 4
) -> list[Path | None]:
    """Синтезує кожен текст в окремий mp3. Для порожніх текстів повертає None."""
    try:
        import edge_tts  # noqa: F401
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("Встановіть edge-tts: pip install edge-tts") from e

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path | None] = []
    jobs: list[tuple[str, Path]] = []
    for i, text in enumerate(texts):
        if not text.strip():
            paths.append(None)
            continue
        p = out_dir / f"seg_{i:05d}.mp3"
        paths.append(p)
        jobs.append((text, p))
    asyncio.run(_synth_all(jobs, voice, rate, concurrency))
    return paths
