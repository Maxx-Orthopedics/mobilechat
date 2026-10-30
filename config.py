"""Runtime configuration, loaded from environment variables (and `.env` if present)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default).strip()


def _bool(name: str, default: bool) -> bool:
    return _env(name, str(default)).lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    parse_base_url: str = "http://beanstalk.maxxortho.com/"
    parse_app_id: str = ""
    parse_rest_key: str = ""

    # A URL for a Qdrant server, ":memory:" for an in-process index (tests), or
    # "path:<dir>" for an on-disk local index (single process only).
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "maxx_media"

    ollama_url: str = "http://localhost:11434"
    llm_model: str = "qwen3"
    llm_disable_thinking: bool = True

    embedding_model: str = "BAAI/bge-m3"
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_enabled: bool = True

    search_top_k: int = 20
    rerank_top_k: int = 5
    # Minimum relevance for a passage to be used. Calibrate both against the eval set.
    rerank_threshold: float = 0.15
    vector_threshold: float = 0.45

    data_dir: Path = ROOT / "data"
    log_dir: Path = ROOT / "logs"

    # Required to call POST /api/ingest/refresh; the endpoint is disabled when empty.
    admin_token: str = ""

    @classmethod
    def from_env(cls) -> "Settings":
        d = cls()
        return cls(
            parse_base_url=_env("PARSE_BASE_URL", d.parse_base_url),
            parse_app_id=_env("PARSE_APP_ID", d.parse_app_id),
            parse_rest_key=_env("PARSE_REST_KEY", d.parse_rest_key),
            qdrant_url=_env("QDRANT_URL", d.qdrant_url),
            qdrant_collection=_env("QDRANT_COLLECTION", d.qdrant_collection),
            ollama_url=_env("OLLAMA_URL", d.ollama_url),
            llm_model=_env("LLM_MODEL", d.llm_model),
            llm_disable_thinking=_bool("LLM_DISABLE_THINKING", d.llm_disable_thinking),
            embedding_model=_env("EMBEDDING_MODEL", d.embedding_model),
            rerank_model=_env("RERANK_MODEL", d.rerank_model),
            rerank_enabled=_bool("RERANK_ENABLED", d.rerank_enabled),
            search_top_k=int(_env("SEARCH_TOP_K", str(d.search_top_k))),
            rerank_top_k=int(_env("RERANK_TOP_K", str(d.rerank_top_k))),
            rerank_threshold=float(_env("RERANK_THRESHOLD", str(d.rerank_threshold))),
            vector_threshold=float(_env("VECTOR_THRESHOLD", str(d.vector_threshold))),
            data_dir=Path(_env("DATA_DIR", str(d.data_dir))),
            log_dir=Path(_env("LOG_DIR", str(d.log_dir))),
            admin_token=_env("ADMIN_TOKEN", d.admin_token),
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings.from_env()
