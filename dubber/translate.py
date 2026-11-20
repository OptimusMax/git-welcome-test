"""Переклад сегментів."""

from __future__ import annotations

from typing import Callable, Iterable

from .segments import Segment

Translator = Callable[[list[str]], list[str]]


def google_translator(source: str, target: str) -> Translator:
    try:
        from deep_translator import GoogleTranslator
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("Встановіть deep-translator: pip install deep-translator") from e

    tr = GoogleTranslator(source=source or "auto", target=target)

    def _translate(texts: list[str]) -> list[str]:
        return [r or t for r, t in zip(tr.translate_batch(texts), texts)]

    return _translate


def _batches(items: list[str], max_chars: int = 4000) -> Iterable[list[str]]:
    batch: list[str] = []
    size = 0
    for item in items:
        if batch and size + len(item) > max_chars:
            yield batch
            batch, size = [], 0
        batch.append(item)
        size += len(item)
    if batch:
        yield batch


def translate_segments(segments: list[Segment], translator: Translator) -> list[Segment]:
    texts = [s.text for s in segments]
    out: list[str] = []
    for batch in _batches(texts):
        out.extend(translator(batch))
    return [seg.with_translation(t.strip()) for seg, t in zip(segments, out)]
