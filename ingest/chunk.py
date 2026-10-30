"""Split extracted pages into ~900-token passages with ~15% overlap.

Splits on paragraph and heading boundaries first, then sentences, then words for
anything still too long. Each chunk keeps the page it mostly came from (for citations).
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from ingest.extract import PageText

TARGET_TOKENS = 900
OVERLAP = 0.15

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def estimate_tokens(text: str) -> int:
    # ~4 characters per token for English; close enough for sizing chunks.
    return max(1, len(text) // 4)


@dataclass(frozen=True)
class Unit:
    text: str
    page: int


@dataclass(frozen=True)
class Chunk:
    index: int
    text: str
    page: int  # primary page, used for the citation
    pages: tuple[int, ...]


def _looks_like_heading(line: str) -> bool:
    line = line.strip()
    return 0 < len(line) <= 80 and not line.endswith((".", ",", ";", ":")) and (
        line.isupper() or line.istitle() or bool(re.match(r"^\d+(\.\d+)*\s", line))
    )


def _paragraphs(text: str) -> list[str]:
    paras: list[str] = []
    for block in re.split(r"\n\s*\n", text):
        current: list[str] = []
        for line in block.splitlines():
            if _looks_like_heading(line) and current:
                paras.append(" ".join(current))
                current = []
            current.append(line.strip())
        if current:
            paras.append(" ".join(current))
    return [p for p in (re.sub(r"\s+", " ", p).strip() for p in paras) if p]


def _split_long(text: str, limit: int) -> list[str]:
    if estimate_tokens(text) <= limit:
        return [text]
    pieces, current = [], ""
    for part in _SENTENCE_RE.split(text):
        if estimate_tokens(part) > limit:  # one giant "sentence": split on words
            words, part = part.split(), ""
            for w in words:
                if part and estimate_tokens(part + " " + w) > limit:
                    pieces.append(part)
                    part = w
                else:
                    part = f"{part} {w}".strip()
        if current and estimate_tokens(current + " " + part) > limit:
            pieces.append(current)
            current = part
        else:
            current = f"{current} {part}".strip()
    if current:
        pieces.append(current)
    return pieces


def _units(pages: list[PageText], target: int) -> list[Unit]:
    return [
        Unit(piece, p.page)
        for p in pages
        for para in _paragraphs(p.text)
        for piece in _split_long(para, target // 2)
    ]


def _make_chunk(index: int, units: list[Unit]) -> Chunk:
    weight = Counter()
    for u in units:
        weight[u.page] += estimate_tokens(u.text)
    primary = max(weight, key=lambda pg: (weight[pg], -pg))
    return Chunk(index, "\n\n".join(u.text for u in units), primary, tuple(sorted(weight)))


def chunk_pages(pages: list[PageText], target: int = TARGET_TOKENS, overlap: float = OVERLAP) -> list[Chunk]:
    chunks: list[Chunk] = []
    current: list[Unit] = []
    size = 0
    fresh = 0  # tokens added since the last emitted chunk (excludes carried overlap)
    for unit in _units(pages, target):
        tokens = estimate_tokens(unit.text)
        if current and size + tokens > target:
            chunks.append(_make_chunk(len(chunks), current))
            carried, carried_size = [], 0
            for u in reversed(current):
                t = estimate_tokens(u.text)
                if carried_size + t > overlap * target:
                    break
                carried.insert(0, u)
                carried_size += t
            current, size, fresh = carried, carried_size, 0
        current.append(unit)
        size += tokens
        fresh += tokens
    if current and fresh:
        chunks.append(_make_chunk(len(chunks), current))
    return chunks
