"""Lightweight stand-ins for the models and the Parse backend, so tests run offline."""

from __future__ import annotations

import hashlib
import json
import math
import re
from urllib.parse import parse_qs, urlparse

import httpx
import pymupdf

_STOPWORDS = {"the", "and", "are", "for", "from", "what", "who", "there", "with", "how", "does"}


class HashEmbedder:
    """Bag-of-words hashed into a fixed-size unit vector. Shared words => higher cosine."""

    dim = 256

    def encode(self, texts: list[str]) -> list[list[float]]:
        out = []
        for text in texts:
            v = [0.0] * self.dim
            for word in re.findall(r"[a-z0-9]+", text.lower()):
                if len(word) > 2 and word not in _STOPWORDS:
                    v[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim] += 1.0
            norm = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / norm for x in v])
        return out


class FakeLLM:
    def __init__(self, reply: str = "It is described in the materials [1]."):
        self.reply = reply
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        return self.reply


def make_pdf(pages: list[str]) -> bytes:
    """A PDF with one page per string; an empty string makes an image-only-like blank page."""
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        if text:
            page.insert_textbox(pymupdf.Rect(50, 50, 550, 800), text, fontsize=10)
    data = doc.tobytes()
    doc.close()
    return data


class FakeParse:
    """Serves Parse REST queries and PDF downloads from in-memory data via httpx.MockTransport."""

    def __init__(self):
        self.classes: dict[str, list[dict]] = {"Company": [], "Products": [], "Category": [], "Media": []}
        self.files: dict[str, bytes] = {}
        self.requests: list[httpx.Request] = []

    def add_pdf(self, object_id: str, title: str, pages: list[str], **fields) -> None:
        url = f"https://files.example.com/{object_id}.pdf"
        self.files[url] = make_pdf(pages)
        self.classes["Media"].append(
            {"objectId": object_id, "type": "pdf", "title": title, "url": url, "active": 1,
             "hidden": False, "updatedAt": "2026-01-01T00:00:00.000Z", **fields}
        )

    def _matches(self, row: dict, where: dict) -> bool:
        for key, cond in where.items():
            if isinstance(cond, dict) and "$in" in cond:
                if row.get(key) not in cond["$in"]:
                    return False
            elif row.get(key) != cond:
                return False
        return True

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        url = str(request.url)
        if url in self.files:
            return httpx.Response(200, content=self.files[url], headers={"content-type": "application/pdf"})
        parsed = urlparse(url)
        cls = parsed.path.rsplit("/", 1)[-1]
        q = {k: v[0] for k, v in parse_qs(parsed.query).items()}
        rows = [r for r in self.classes.get(cls, []) if self._matches(r, json.loads(q.get("where", "{}")))]
        if q.get("count") == "1":
            return httpx.Response(200, json={"results": [], "count": len(rows)})
        skip, limit = int(q.get("skip", 0)), int(q.get("limit", 100))
        return httpx.Response(200, json={"results": rows[skip : skip + limit]})

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handler)
