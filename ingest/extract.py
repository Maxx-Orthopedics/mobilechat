"""Per-page text extraction with PyMuPDF, falling back to Tesseract OCR for scanned pages."""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path

import pymupdf

log = logging.getLogger(__name__)

# Pages with less extractable text than this are treated as scanned images.
MIN_TEXT_CHARS = 40
OCR_LANGUAGE = os.environ.get("OCR_LANGUAGE", "eng")
OCR_DPI = 300

_ocr_warned = False


@dataclass(frozen=True)
class PageText:
    page: int  # 1-based
    text: str
    ocr: bool = False


def _clean(text: str) -> str:
    text = text.replace("­", "").replace("\x00", "")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # re-join hyphenated line breaks
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def page_needs_ocr(page: pymupdf.Page) -> bool:
    return len(page.get_text("text").strip()) < MIN_TEXT_CHARS


def _ocr(page: pymupdf.Page) -> str | None:
    global _ocr_warned
    try:
        tp = page.get_textpage_ocr(language=OCR_LANGUAGE, dpi=OCR_DPI, full=True)
        return page.get_text("text", textpage=tp)
    except Exception as e:  # noqa: BLE001 - Tesseract missing or failed
        if not _ocr_warned:
            log.warning("OCR unavailable (%s); scanned pages will be skipped. Install tesseract-ocr.", e)
            _ocr_warned = True
        return None


def extract_pages(path: Path, ocr: bool = True) -> list[PageText]:
    pages = []
    with pymupdf.open(path) as doc:
        for i, page in enumerate(doc, start=1):
            text, used_ocr = page.get_text("text", sort=True), False
            if ocr and len(text.strip()) < MIN_TEXT_CHARS:
                ocr_text = _ocr(page)
                if ocr_text and len(ocr_text.strip()) > len(text.strip()):
                    text, used_ocr = ocr_text, True
            text = _clean(text)
            if text:
                pages.append(PageText(page=i, text=text, ocr=used_ocr))
    return pages


def scan_profile(path: Path) -> tuple[int, int]:
    """Return (page count, pages that look scanned / need OCR)."""
    with pymupdf.open(path) as doc:
        return doc.page_count, sum(page_needs_ocr(p) for p in doc)
