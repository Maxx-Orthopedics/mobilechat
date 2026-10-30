import httpx
import pytest

import ingest.enumerate as enumerate_mod
from config import Settings
from ingest.download import PdfStore
from ingest.embed_index import VectorIndex, make_qdrant
from ingest.enumerate import ParseClient, enumerate_pdfs
from ingest.run_ingest import run_ingest
from tests.fakes import FakeParse, HashEmbedder


@pytest.fixture
def parse():
    fake = FakeParse()
    fake.classes["Company"] = [{"objectId": "co1", "name": "Maxx Orthopedics"}]
    fake.classes["Products"] = [{"objectId": "pr1", "name": "Knee System", "companyId": "co1"}]
    fake.classes["Category"] = [
        {"objectId": "cat1", "name": "Surgical Technique", "parent_object_id": "pr1"},
        {"objectId": "cat2", "name": "Sub-category", "parent_object_id": "cat1"},
    ]
    fake.add_pdf("m1", "Knee Surgical Technique", ["Femoral sizing guide overview.", "Tibial insert sizes range from 1 to 8."], categoryId="cat1")
    fake.add_pdf("m2", "Instrument Tray Guide", ["Tray A holds the femoral trials and cutting blocks."], categoryId="cat2")
    fake.add_pdf("m3", "Hidden Draft", ["Do not index me."], hidden=True)
    fake.add_pdf("m4", "Scanned Brochure", [""])  # no text layer, no OCR in tests
    fake.classes["Media"].append({"objectId": "v1", "type": "video", "title": "Video", "url": "x", "active": 1})
    return fake


@pytest.fixture
def env(tmp_path, parse):
    settings = Settings(parse_app_id="app", parse_rest_key="key", data_dir=tmp_path)
    transport = parse.transport()
    embedder = HashEmbedder()
    index = VectorIndex(make_qdrant(":memory:"), "test", embedder.dim)

    def run(**kw):
        return run_ingest(
            settings,
            parse=ParseClient(settings, transport=transport),
            store=PdfStore(tmp_path, client=httpx.Client(transport=transport)),
            embedder=embedder,
            index=index,
            ocr=False,
            **kw,
        )

    return run, index


def test_enumerate_filters_and_tags(parse):
    client = ParseClient(Settings(parse_app_id="a", parse_rest_key="k"), transport=parse.transport())
    pdfs = {p.object_id: p for p in enumerate_pdfs(client)}
    assert set(pdfs) == {"m1", "m2", "m4"}
    assert (pdfs["m1"].product, pdfs["m1"].category, pdfs["m1"].company) == ("Knee System", "Surgical Technique", "Maxx Orthopedics")
    assert pdfs["m2"].product == "Knee System"  # nested category walks up to the product
    req = parse.requests[-1]
    assert req.headers["X-Parse-Application-Id"] == "a" and req.headers["X-Parse-REST-API-Key"] == "k"


def test_enumerate_pages_through_results(parse, monkeypatch):
    monkeypatch.setattr(enumerate_mod, "PAGE_SIZE", 1)
    client = ParseClient(Settings(parse_app_id="a", parse_rest_key="k"), transport=parse.transport())
    assert {p.object_id for p in enumerate_pdfs(client, with_taxonomy=False)} == {"m1", "m2", "m4"}


def test_ingest_is_idempotent_and_drops_removed(env, parse):
    run, index = env
    first = run()
    assert (first["indexed"], first["skipped"], first["removed"]) == (2, 0, 0)
    assert [f["objectId"] for f in first["failed"]] == ["m4"]
    assert index.document_ids() == {"m1", "m2"}

    second = run()
    assert (second["indexed"], second["skipped"]) == (0, 2)

    # Title change re-indexes just that doc; deactivating a doc removes it.
    parse.classes["Media"][0]["title"] = "Knee Surgical Technique v2"
    parse.classes["Media"][1]["active"] = 0
    third = run()
    assert (third["indexed"], third["skipped"], third["removed"]) == (1, 0, 1)
    assert index.document_ids() == {"m1"}


def test_limited_run_does_not_remove_others(env):
    run, index = env
    run()
    result = run(limit=1)
    assert result["removed"] == 0
    assert index.document_ids() == {"m1", "m2"}


def test_chunk_payload_has_citation_metadata(env):
    run, index = env
    run()
    vec = HashEmbedder().encode(["tibial insert sizes"])[0]
    top = index.search(vec, 1)[0].payload
    assert top["title"] == "Knee Surgical Technique"
    assert top["page"] == 2
    assert top["source_url"].endswith("/m1.pdf")
    assert top["product"] == "Knee System"
