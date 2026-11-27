"""Переклад сегментів."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, Iterable

from .segments import Segment

Translator = Callable[[list[str]], list[str]]
TextTranslator = Callable[[str], str]

MAX_CHARS = 1500  # невеликі пачки — менше шансів, що Google заблокує


class TranslationBlocked(RuntimeError):
    """Сервіс тимчасово заблокував запити — повтори лише погіршать ситуацію."""


def _with_retries(fn: TextTranslator, retries: int = 5, delay: float = 5.0) -> TextTranslator:
    def wrapped(text: str) -> str:
        for attempt in range(retries):
            try:
                return fn(text)
            except TranslationBlocked:
                raise
            except Exception:
                if attempt == retries - 1:
                    raise
                time.sleep(delay * (2**attempt))  # 5, 10, 20, 40 с
        return text

    return wrapped


def batched_translator(translate_text: TextTranslator, pause: float = 3.0) -> Translator:
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


GTX_URL = "https://translate.googleapis.com/translate_a/single"


def gtx_translate_text(source: str, target: str) -> TextTranslator:
    """Google Translate через публічний JSON-ендпоінт (POST, до ~5000 символів за запит)."""
    import requests

    session = requests.Session()

    def _translate(text: str) -> str:
        resp = session.post(
            GTX_URL,
            params={"client": "gtx", "sl": source or "auto", "tl": target, "dt": "t"},
            data={"q": text},
            timeout=30,
            allow_redirects=False,
        )
        if resp.status_code in (302, 429) or "/sorry/" in resp.headers.get("Location", ""):
            raise TranslationBlocked(
                "Google Translate тимчасово заблокував запити з вашої адреси "
                "(«підозрілий трафік»). Зазвичай це минає за 1–2 години."
            )
        resp.raise_for_status()
        return parse_gtx_response(resp.json())

    return _translate


def parse_gtx_response(data) -> str:
    return "".join(part[0] for part in (data[0] or []) if part and part[0])


def google_translator(source: str, target: str) -> Translator:
    return batched_translator(gtx_translate_text(source, target))


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
    segments: list[Segment],
    translator: Translator,
    log: Callable[[str], None] | None = None,
    cache_path: str | Path | None = None,
) -> list[Segment]:
    """Перекладає сегменти. Якщо задано cache_path — зберігає вже перекладене,
    і повторний запуск продовжує з місця зупинки."""
    cache: dict[str, str] = {}
    if cache_path and Path(cache_path).exists():
        try:
            cache = json.loads(Path(cache_path).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            cache = {}

    todo = list(dict.fromkeys(s.text for s in segments if s.text not in cache))
    if cache and log:
        log(f"      вже перекладено раніше: {len(segments) - len(todo)}, залишилось: {len(todo)}")
    batches = list(_batches(todo))
    for i, batch in enumerate(batches, 1):
        if log:
            log(f"      пачка {i}/{len(batches)}")
        for src, dst in zip(batch, translator(batch)):
            cache[src] = dst
        if cache_path:
            Path(cache_path).write_text(json.dumps(cache, ensure_ascii=False, indent=0), encoding="utf-8")
    return [seg.with_translation(cache.get(seg.text, "").strip()) for seg in segments]
