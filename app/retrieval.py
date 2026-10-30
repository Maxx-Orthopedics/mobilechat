"""Question -> relevant passages: embed, vector search, (optionally) rerank, threshold."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ingest.embed_index import Embedder, VectorIndex


@dataclass(frozen=True)
class Passage:
    text: str
    title: str
    url: str
    page: int
    object_id: str
    score: float  # cosine similarity
    rerank_score: float | None = None


class Reranker(Protocol):
    def score(self, question: str, texts: list[str]) -> list[float]: ...


class CrossEncoderReranker:
    def __init__(self, model_name: str):
        from sentence_transformers import CrossEncoder

        # Single-logit rerankers get a sigmoid by default, so scores land in 0..1.
        self.model = CrossEncoder(model_name, max_length=1024)

    def score(self, question: str, texts: list[str]) -> list[float]:
        return [float(s) for s in self.model.predict([(question, t) for t in texts], show_progress_bar=False)]


class Retriever:
    def __init__(
        self,
        embedder: Embedder,
        index: VectorIndex,
        reranker: Reranker | None = None,
        *,
        search_top_k: int = 20,
        rerank_top_k: int = 5,
        rerank_threshold: float = 0.15,
        vector_threshold: float = 0.45,
    ):
        self.embedder, self.index, self.reranker = embedder, index, reranker
        self.search_top_k, self.rerank_top_k = search_top_k, rerank_top_k
        self.rerank_threshold, self.vector_threshold = rerank_threshold, vector_threshold

    def retrieve(self, question: str) -> list[Passage]:
        """Return up to `rerank_top_k` passages above the relevance threshold, best first.

        An empty list means nothing in the corpus is relevant enough to answer from.
        """
        vector = self.embedder.encode([question])[0]
        passages = [
            Passage(
                text=p.payload["text"],
                title=p.payload["title"],
                url=p.payload["source_url"],
                page=int(p.payload["page"]),
                object_id=p.payload["object_id"],
                score=float(p.score),
            )
            for p in self.index.search(vector, self.search_top_k)
        ]
        if not passages:
            return []

        if self.reranker is None:
            kept = [p for p in passages if p.score >= self.vector_threshold]
        else:
            scores = self.reranker.score(question, [p.text for p in passages])
            ranked = sorted(
                (Passage(**{**p.__dict__, "rerank_score": s}) for p, s in zip(passages, scores, strict=True)),
                key=lambda p: p.rerank_score,
                reverse=True,
            )
            kept = [p for p in ranked if p.rerank_score >= self.rerank_threshold]
        return kept[: self.rerank_top_k]
