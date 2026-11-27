"""Переклад сегментів."""

from __future__ import annotations

import time
from typing import Callable, Iterable

from .segments import Segment

Translator = Callable[[list[str]], list[str]]
TextTranslator = Callable[[str], str]

MAX_CHARS = 4500  # ліміт Google Translate — 5000 символів на запит


def _with_retries(fn: TextTranslator, retries: int = 5, delay: float = 2.0) -> TextTranslator:
    def wrapped(text: str) -> str:
        for attempt in range(retries):
            try:
                return fn(text)
            except Exception:
                if attempt == retries - 1:
                    raise
                time.sleep(delay * (2**attempt))  # 2, 4, 8, 16 с
        return text

    return wrapped


def batched_translator(translate_text: TextTranslator, pause: float = 0.5) -> Translator:
    """Перекладає список фраз мінімумом запитів: зшиває їх через перенос рядка.

    Якщо сервіс повернув інший рядків, ніж надіслали, перекладає цю пачку пофразно.
    """
    translate_text = _with_retries(translate_text)

    def _translate(texts: list[str]) -> list[str]:
        clean = [" ".join(t.split()) for t in texts]
        result = translate_text("\n".join(clean)) or ""
        lines = [ln.strip() for ln in result.split("\n")]
        if len(lines) == len(clean) and all(lines[i] or not clean[i] for i in range(len(clean))):
            time.sleep(pause)
            return lines
        out = []
        for t in clean:
            out.append((translate_text(t) or t).strip() if t else "")
            time.sleep(pause)
        return out

    return _translate


def google_translator(source: str, target: str) -> Translator:
    try:
        from deep_translator import GoogleTranslator
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("Встановіть deep-translator: pip install deep-translator") from e

    tr = GoogleTranslator(source=source or "auto", target=target)
    return batched_translator(tr.translate)


def _batches(items: list[str], max_chars: int = MAX_CHARS) -> Iterable[list[str]]:
    batch: list[str] = []
    size = 0
    for item in items:
        if batch and size + len(item) + 1 > max_chars:
            yield batch
            batch, size = [], 0
        batch.append(item)
        size += len(item) + 1
    if batch:
        yield batch


def translate_segments(
    segments: list[Segment], translator: Translator, log: Callable[[str], None] | None = None
) -> list[Segment]:
    texts = [s.text for s in segments]
    batches = list(_batches(texts))
    out: list[str] = []
    for i, batch in enumerate(batches, 1):
        if log:
            log(f"      пачка {i}/{len(batches)}")
        out.extend(translator(batch))
    return [seg.with_translation(t.strip()) for seg, t in zip(segments, out)]
