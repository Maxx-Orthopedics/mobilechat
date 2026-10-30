"""Grounded answer generation: build the prompt, call the LLM, map citations back to sources."""

from __future__ import annotations

import json
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

import httpx

from app.retrieval import Passage, Retriever

SYSTEM_PROMPT = (
    "You answer only using the provided Maxx Orthopedics materials. If the answer is "
    "not in them, say you could not find it in the materials and suggest contacting a "
    "Maxx representative. Do not give medical, clinical, dosing, or surgical advice. "
    "For every claim, cite the document title and page.\n\n"
    "The materials are numbered sources like [1], [2]. Cite them by putting the source "
    "number in square brackets after each claim, e.g. \"... is available in three sizes [2].\" "
    "Do not use outside knowledge. If the materials do not answer the question, reply with "
    "exactly NOT_FOUND and nothing else."
)

NOT_FOUND_TOKEN = "NOT_FOUND"
NOT_FOUND_MESSAGE = (
    "I couldn't find that in the Maxx Orthopedics materials. "
    "Please contact your Maxx representative for help."
)

_CITE_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


class LLM(Protocol):
    def complete(self, system: str, user: str) -> str: ...


class OllamaLLM:
    def __init__(self, base_url: str, model: str, disable_thinking: bool = True, timeout: float = 180):
        self.model, self.disable_thinking = model, disable_thinking
        self._client = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout)

    def complete(self, system: str, user: str) -> str:
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "stream": False,
            "options": {"temperature": 0.1},
        }
        if self.disable_thinking:
            body["think"] = False
        r = self._client.post("/api/chat", json=body)
        if r.status_code == 400 and self.disable_thinking:
            # Models without a thinking mode reject the flag; retry without it.
            body.pop("think")
            r = self._client.post("/api/chat", json=body)
        r.raise_for_status()
        return r.json()["message"]["content"]


def build_user_prompt(question: str, passages: list[Passage]) -> str:
    sources = "\n\n".join(
        f"[{i}] {p.title} — page {p.page}\n{p.text}" for i, p in enumerate(passages, start=1)
    )
    return f"Materials:\n\n{sources}\n\nQuestion: {question}"


def _not_found() -> dict:
    return {"answer": NOT_FOUND_MESSAGE, "citations": [], "foundInMaterials": False}


def ground_answer(raw: str, passages: list[Passage]) -> dict:
    """Turn model output into the API response, renumbering [n] markers to the citations list."""
    text = _THINK_RE.sub("", raw).strip()
    if not text or text.upper().startswith(NOT_FOUND_TOKEN):
        return _not_found()

    cited = [
        int(n)
        for group in _CITE_RE.findall(text)
        for n in group.split(",")
        if 1 <= int(n) <= len(passages)
    ]
    # If the model didn't cite anything, attribute the answer to everything it was shown.
    used = list(dict.fromkeys(cited)) or list(range(1, len(passages) + 1))

    citations: list[dict] = []
    keys: dict[tuple, int] = {}
    source_to_citation: dict[int, int] = {}
    for n in used:
        p = passages[n - 1]
        key = (p.url, p.page)
        if key not in keys:
            citations.append({"title": p.title, "url": p.url, "page": p.page})
            keys[key] = len(citations)
        source_to_citation[n] = keys[key]

    def renumber(m: re.Match) -> str:
        nums = [source_to_citation.get(int(n)) for n in m.group(1).split(",")]
        nums = list(dict.fromkeys(n for n in nums if n))
        return f"[{', '.join(map(str, nums))}]" if nums else ""

    answer = re.sub(r"\s+([.,;:])", r"\1", _CITE_RE.sub(renumber, text)).strip()
    return {"answer": answer, "citations": citations, "foundInMaterials": True}


class QALog:
    """Append-only JSONL log of every question, answer, and the sources behind it (§8)."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(self, record: dict) -> None:
        line = json.dumps({"ts": datetime.now(timezone.utc).isoformat(), **record}, ensure_ascii=False)
        with self._lock, self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


class AnswerService:
    def __init__(self, retriever: Retriever, llm: LLM, qa_log: QALog | None = None):
        self.retriever, self.llm, self.qa_log = retriever, llm, qa_log

    def answer(self, question: str, session_id: str | None = None) -> dict:
        passages = self.retriever.retrieve(question)
        if passages:
            result = ground_answer(self.llm.complete(SYSTEM_PROMPT, build_user_prompt(question, passages)), passages)
        else:
            result = _not_found()

        if self.qa_log:
            self.qa_log.write(
                {
                    "sessionId": session_id,
                    "question": question,
                    **result,
                    "retrieved": [
                        {
                            "objectId": p.object_id,
                            "title": p.title,
                            "page": p.page,
                            "score": round(p.score, 4),
                            "rerankScore": None if p.rerank_score is None else round(p.rerank_score, 4),
                        }
                        for p in passages
                    ],
                }
            )
        return result
