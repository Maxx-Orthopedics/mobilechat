"""FastAPI service: /api/chat, /api/ingest/refresh, /api/health, and the chat page at /.

    uvicorn app.main:app --reload
"""

from __future__ import annotations

import logging
import secrets
import threading
from collections.abc import Callable
from functools import cached_property

from fastapi import FastAPI, Header, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from config import ROOT, Settings, get_settings
from app.generate import AnswerService, OllamaLLM, QALog
from app.retrieval import CrossEncoderReranker, Retriever
from ingest.embed_index import BGEEmbedder, VectorIndex, make_qdrant

log = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    sessionId: str | None = Field(default=None, max_length=200)


class Citation(BaseModel):
    title: str
    url: str
    page: int


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation]
    foundInMaterials: bool


class IngestResponse(BaseModel):
    indexed: int
    skipped: int
    removed: int = 0
    failed: int = 0


class Components:
    """Models and clients, loaded on first use so the server starts fast and /api/health works alone."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._lock = threading.Lock()

    @cached_property
    def embedder(self) -> BGEEmbedder:
        return BGEEmbedder(self.settings.embedding_model)

    @cached_property
    def index(self) -> VectorIndex:
        s = self.settings
        return VectorIndex(make_qdrant(s.qdrant_url), s.qdrant_collection, self.embedder.dim)

    @cached_property
    def service(self) -> AnswerService:
        s = self.settings
        retriever = Retriever(
            self.embedder,
            self.index,
            CrossEncoderReranker(s.rerank_model) if s.rerank_enabled else None,
            search_top_k=s.search_top_k,
            rerank_top_k=s.rerank_top_k,
            rerank_threshold=s.rerank_threshold,
            vector_threshold=s.vector_threshold,
        )
        llm = OllamaLLM(s.ollama_url, s.llm_model, s.llm_disable_thinking)
        return AnswerService(retriever, llm, QALog(s.log_dir / "qa.jsonl"))

    def get_service(self) -> AnswerService:
        with self._lock:
            return self.service

    def run_ingest(self) -> dict:
        from ingest.run_ingest import run_ingest

        with self._lock:
            embedder, index = self.embedder, self.index
        return run_ingest(self.settings, embedder=embedder, index=index)


def create_app(
    settings: Settings | None = None,
    service: AnswerService | None = None,
    ingest: Callable[[], dict] | None = None,
) -> FastAPI:
    settings = settings or get_settings()
    components = Components(settings)
    get_service = (lambda: service) if service else components.get_service
    ingest = ingest or components.run_ingest
    ingest_lock = threading.Lock()

    app = FastAPI(title="Maxx Content Q&A", version="0.1.0")

    @app.get("/api/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/api/chat", response_model=ChatResponse)
    def chat(req: ChatRequest) -> dict:
        question = req.question.strip()
        if not question:
            raise HTTPException(422, "question is empty")
        try:
            return get_service().answer(question, req.sessionId)
        except Exception as e:
            log.exception("chat failed")
            raise HTTPException(503, "The answer service is unavailable. Please try again shortly.") from e

    @app.post("/api/ingest/refresh", response_model=IngestResponse)
    async def refresh(x_admin_token: str | None = Header(default=None)) -> dict:
        if not settings.admin_token:
            raise HTTPException(403, "ingest refresh is disabled (ADMIN_TOKEN not set)")
        if not x_admin_token or not secrets.compare_digest(x_admin_token, settings.admin_token):
            raise HTTPException(401, "invalid admin token")
        if not ingest_lock.acquire(blocking=False):
            raise HTTPException(409, "an ingest is already running")
        try:
            result = await run_in_threadpool(ingest)
        finally:
            ingest_lock.release()
        return {**result, "failed": len(result.get("failed", []))}

    @app.get("/", include_in_schema=False)
    def index_page() -> FileResponse:
        return FileResponse(ROOT / "web" / "index.html")

    return app


app = create_app()
