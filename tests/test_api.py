import json

import pytest
from fastapi.testclient import TestClient

from app.generate import NOT_FOUND_MESSAGE, AnswerService, QALog, ground_answer
from app.main import create_app
from app.retrieval import Passage, Retriever
from config import Settings
from ingest.chunk import Chunk
from ingest.embed_index import VectorIndex, embedding_input, make_qdrant
from ingest.enumerate import MediaPDF
from tests.fakes import FakeLLM, HashEmbedder

DOCS = [
    (MediaPDF("m1", "Knee Surgical Technique", "https://f/m1.pdf"), "Tibial insert sizes range from size 1 to size 8.", 4),
    (MediaPDF("m2", "Instrument Tray Guide", "https://f/m2.pdf"), "Tray A holds the femoral trials and cutting blocks.", 2),
]


@pytest.fixture
def setup(tmp_path):
    embedder = HashEmbedder()
    index = VectorIndex(make_qdrant(":memory:"), "test", embedder.dim)
    for pdf, text, page in DOCS:
        index.replace_document(pdf, [Chunk(0, text, page, (page,))], embedder.encode([embedding_input(pdf.title, text)]))
    llm = FakeLLM()
    log_path = tmp_path / "qa.jsonl"
    retriever = Retriever(embedder, index, vector_threshold=0.3)
    service = AnswerService(retriever, llm, QALog(log_path))
    calls = []
    settings = Settings(admin_token="s3cret")
    client = TestClient(create_app(settings, service=service, ingest=lambda: calls.append(1) or {"indexed": 3, "skipped": 1, "removed": 0, "failed": []}))
    return client, llm, log_path, calls


def test_health(setup):
    client, *_ = setup
    assert client.get("/api/health").json() == {"status": "ok"}


def test_chat_answers_with_citation_and_logs(setup):
    client, llm, log_path, _ = setup
    r = client.post("/api/chat", json={"question": "What tibial insert sizes are there?", "sessionId": "abc"})
    assert r.status_code == 200
    body = r.json()
    assert body["foundInMaterials"] is True
    assert body["citations"][0] == {"title": "Knee Surgical Technique", "url": "https://f/m1.pdf", "page": 4}
    system, user = llm.calls[0]
    assert "Do not give medical" in system and "[1] Knee Surgical Technique — page 4" in user
    record = json.loads(log_path.read_text().splitlines()[-1])
    assert record["sessionId"] == "abc" and record["retrieved"][0]["objectId"] == "m1"


def test_irrelevant_question_is_not_found_without_calling_llm(setup):
    client, llm, *_ = setup
    body = client.post("/api/chat", json={"question": "Who won the 1998 world cup?"}).json()
    assert body == {"answer": NOT_FOUND_MESSAGE, "citations": [], "foundInMaterials": False}
    assert llm.calls == []


def test_model_not_found_reply_is_respected(setup):
    client, llm, *_ = setup
    llm.reply = "NOT_FOUND"
    body = client.post("/api/chat", json={"question": "What tibial insert sizes are there?"}).json()
    assert body["foundInMaterials"] is False and body["citations"] == []


def test_chat_validates_input(setup):
    client, *_ = setup
    assert client.post("/api/chat", json={"question": ""}).status_code == 422
    assert client.post("/api/chat", json={"question": "x" * 2001}).status_code == 422


def test_ingest_refresh_requires_token(setup):
    client, *_, calls = setup
    assert client.post("/api/ingest/refresh").status_code == 401
    assert client.post("/api/ingest/refresh", headers={"X-Admin-Token": "nope"}).status_code == 401
    r = client.post("/api/ingest/refresh", headers={"X-Admin-Token": "s3cret"})
    assert r.json() == {"indexed": 3, "skipped": 1, "removed": 0, "failed": 0}
    assert calls == [1]


def test_ingest_refresh_disabled_without_token_configured():
    client = TestClient(create_app(Settings(), service=object(), ingest=lambda: {}))
    assert client.post("/api/ingest/refresh", headers={"X-Admin-Token": ""}).status_code == 403


def test_index_page_served(setup):
    client, *_ = setup
    r = client.get("/")
    assert r.status_code == 200 and "Maxx Content Q&amp;A" in r.text


def _p(title, page, url=None):
    return Passage("text", title, url or f"https://f/{title}.pdf", page, title, 0.9)


def test_ground_answer_renumbers_and_dedupes_citations():
    passages = [_p("A", 1), _p("B", 3), _p("A", 1), _p("C", 2)]
    out = ground_answer("<think>hmm</think>Claim one [2]. Claim two [3, 4]. Bogus [9].", passages)
    assert out["citations"] == [
        {"title": "B", "url": "https://f/B.pdf", "page": 3},
        {"title": "A", "url": "https://f/A.pdf", "page": 1},
        {"title": "C", "url": "https://f/C.pdf", "page": 2},
    ]
    assert out["answer"] == "Claim one [1]. Claim two [2, 3]. Bogus."


def test_ground_answer_without_markers_cites_all_sources():
    out = ground_answer("An answer with no markers.", [_p("A", 1), _p("B", 2)])
    assert [c["title"] for c in out["citations"]] == ["A", "B"]
