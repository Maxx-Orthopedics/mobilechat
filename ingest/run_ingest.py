"""Run the full ingestion pipeline. Idempotent: unchanged PDFs are skipped, removed ones dropped.

    python -m ingest.run_ingest --limit 25   # Phase 0 subset (no removals)
    python -m ingest.run_ingest              # full corpus
    python -m ingest.run_ingest --force      # re-extract and re-embed everything
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging

from config import Settings, get_settings
from ingest.chunk import chunk_pages
from ingest.download import PdfStore
from ingest.embed_index import BGEEmbedder, Embedder, VectorIndex, embedding_input, make_qdrant
from ingest.enumerate import MediaPDF, ParseClient, enumerate_pdfs
from ingest.extract import extract_pages

log = logging.getLogger(__name__)

# Bump when extraction/chunking changes so every document is re-indexed.
PIPELINE_VERSION = "1"


def _index_key(pdf: MediaPDF, sha: str) -> str:
    """Changes when the file, its citation metadata, or the pipeline changes."""
    meta = json.dumps([PIPELINE_VERSION, sha, pdf.title, pdf.url, pdf.product, pdf.category, pdf.company])
    return hashlib.sha256(meta.encode()).hexdigest()


def run_ingest(
    settings: Settings | None = None,
    *,
    limit: int | None = None,
    force: bool = False,
    ocr: bool = True,
    parse: ParseClient | None = None,
    store: PdfStore | None = None,
    embedder: Embedder | None = None,
    index: VectorIndex | None = None,
) -> dict:
    settings = settings or get_settings()
    parse = parse or ParseClient(settings)
    store = store or PdfStore(settings.data_dir)
    embedder = embedder or BGEEmbedder(settings.embedding_model)
    index = index or VectorIndex(make_qdrant(settings.qdrant_url), settings.qdrant_collection, embedder.dim)

    pdfs = enumerate_pdfs(parse)
    full_run = limit is None
    if not full_run:
        pdfs = pdfs[:limit]

    indexed = skipped = removed = 0
    failed: list[dict] = []
    for n, pdf in enumerate(pdfs, start=1):
        try:
            path, sha = store.fetch(pdf)
            key = _index_key(pdf, sha)
            if not force and store.is_indexed(pdf.object_id, key):
                skipped += 1
                continue
            chunks = chunk_pages(extract_pages(path, ocr=ocr))
            if not chunks:
                index.delete_document(pdf.object_id)
                failed.append({"objectId": pdf.object_id, "title": pdf.title, "error": "no extractable text"})
                continue
            vectors = embedder.encode([embedding_input(pdf.title, c.text) for c in chunks])
            index.replace_document(pdf, chunks, vectors)
            store.mark_indexed(pdf.object_id, key)
            indexed += 1
            log.info("[%d/%d] indexed %s (%d chunks)", n, len(pdfs), pdf.title, len(chunks))
        except Exception as e:  # noqa: BLE001 - one bad PDF shouldn't stop the run
            log.exception("failed: %s", pdf.title)
            failed.append({"objectId": pdf.object_id, "title": pdf.title, "error": str(e)})
        if n % 10 == 0:
            store.save()

    if full_run:
        # Anything no longer active+visible in Parse gets dropped from the index.
        active = {p.object_id for p in pdfs}
        for object_id in (index.document_ids() | store.ids()) - active:
            index.delete_document(object_id)
            store.forget(object_id)
            removed += 1

    store.save()
    return {"indexed": indexed, "skipped": skipped, "removed": removed, "failed": failed}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, help="only process the first N PDFs (skips removal of others)")
    parser.add_argument("--force", action="store_true", help="re-index even if unchanged")
    parser.add_argument("--no-ocr", action="store_true", help="skip OCR for scanned pages")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = run_ingest(limit=args.limit, force=args.force, ocr=not args.no_ocr)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
