"""Embed chunks with BGE-M3 and keep them in a Qdrant collection (one doc = many points)."""

from __future__ import annotations

import logging
import uuid
from typing import Protocol

from qdrant_client import QdrantClient, models

from ingest.chunk import Chunk
from ingest.enumerate import MediaPDF

log = logging.getLogger(__name__)

_POINT_NAMESPACE = uuid.UUID("4f6b1f2e-9a51-4c1e-8f2a-6d7c0b3e5a10")


class Embedder(Protocol):
    dim: int

    def encode(self, texts: list[str]) -> list[list[float]]: ...


class BGEEmbedder:
    def __init__(self, model_name: str, batch_size: int = 16):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
        self.batch_size = batch_size

    def encode(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(
            texts, batch_size=self.batch_size, normalize_embeddings=True, show_progress_bar=False
        )
        return vectors.tolist()


def make_qdrant(url: str) -> QdrantClient:
    if url == ":memory:":
        return QdrantClient(location=":memory:")
    if url.startswith("path:"):
        return QdrantClient(path=url.removeprefix("path:"))
    return QdrantClient(url=url)


def embedding_input(title: str, text: str) -> str:
    # Prefixing the title helps match questions that name the product.
    return f"{title}\n\n{text}"


def _doc_filter(object_id: str) -> models.Filter:
    return models.Filter(must=[models.FieldCondition(key="object_id", match=models.MatchValue(value=object_id))])


class VectorIndex:
    def __init__(self, client: QdrantClient, collection: str, dim: int):
        self.client, self.collection = client, collection
        if not client.collection_exists(collection):
            client.create_collection(
                collection, vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE)
            )
            try:
                client.create_payload_index(collection, "object_id", models.PayloadSchemaType.KEYWORD)
            except Exception:  # noqa: BLE001 - not supported by local mode
                pass

    def replace_document(self, pdf: MediaPDF, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        self.delete_document(pdf.object_id)
        points = [
            models.PointStruct(
                id=str(uuid.uuid5(_POINT_NAMESPACE, f"{pdf.object_id}:{c.index}")),
                vector=v,
                payload={
                    "object_id": pdf.object_id,
                    "chunk_index": c.index,
                    "text": c.text,
                    "title": pdf.title,
                    "product": pdf.product,
                    "category": pdf.category,
                    "company": pdf.company,
                    "source_url": pdf.url,
                    "page": c.page,
                    "pages": list(c.pages),
                },
            )
            for c, v in zip(chunks, vectors, strict=True)
        ]
        for start in range(0, len(points), 256):
            self.client.upsert(self.collection, points=points[start : start + 256])

    def delete_document(self, object_id: str) -> None:
        self.client.delete(self.collection, points_selector=models.FilterSelector(filter=_doc_filter(object_id)))

    def document_ids(self) -> set[str]:
        ids, offset = set(), None
        while True:
            points, offset = self.client.scroll(
                self.collection, limit=512, offset=offset, with_payload=["object_id"], with_vectors=False
            )
            ids.update(p.payload["object_id"] for p in points)
            if offset is None:
                return ids

    def search(self, vector: list[float], limit: int) -> list[models.ScoredPoint]:
        return self.client.query_points(self.collection, query=vector, limit=limit, with_payload=True).points
