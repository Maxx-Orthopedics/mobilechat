from ingest.chunk import chunk_pages, estimate_tokens
from ingest.extract import PageText


def _para(word: str, n: int) -> str:
    return " ".join(f"{word}{i}." for i in range(n))


def test_short_document_is_one_chunk():
    chunks = chunk_pages([PageText(1, "Intro\n\nThe implant is titanium.")])
    assert len(chunks) == 1
    assert chunks[0].page == 1 and "titanium" in chunks[0].text


def test_chunks_respect_target_and_overlap():
    # ~60 paragraphs of ~50 tokens each across 3 pages
    pages = [PageText(p, "\n\n".join(_para(f"p{p}w{j}x", 20) for j in range(20))) for p in (1, 2, 3)]
    chunks = chunk_pages(pages, target=400, overlap=0.15)
    assert len(chunks) > 3
    for c in chunks:
        assert estimate_tokens(c.text) <= 400 + 60
    # consecutive chunks share their boundary paragraph(s)
    for a, b in zip(chunks, chunks[1:]):
        assert a.text.split("\n\n")[-1] in b.text


def test_primary_page_is_where_most_text_is():
    pages = [PageText(1, "short tail on page one."), PageText(2, _para("big", 60))]
    (chunk,) = chunk_pages(pages, target=900)
    assert chunk.pages == (1, 2)
    assert chunk.page == 2


def test_oversized_paragraph_is_split():
    huge = " ".join(["word"] * 3000)  # one paragraph, no sentence breaks
    chunks = chunk_pages([PageText(1, huge)], target=300)
    assert len(chunks) > 1
    assert all(estimate_tokens(c.text) <= 300 + 50 for c in chunks)
