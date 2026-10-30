"""List every active, visible PDF in the Maxx Parse backend.

Run directly for the Phase 0 corpus report (PDF count + how many look scanned):

    python -m ingest.enumerate              # count only
    python -m ingest.enumerate --probe 50   # also download 50 PDFs and check for scans
    python -m ingest.enumerate --probe all
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterator
from dataclasses import asdict, dataclass

import httpx

from config import Settings, get_settings

PAGE_SIZE = 100
# `active` is stored as 1 on some rows and true on others.
PDF_WHERE = {"type": "pdf", "active": {"$in": [1, True]}}


@dataclass(frozen=True)
class MediaPDF:
    object_id: str
    title: str
    url: str
    description: str = ""
    category_id: str | None = None
    category: str | None = None
    product: str | None = None
    company: str | None = None
    updated_at: str | None = None


class ParseClient:
    """Read-only Parse REST client. Server-side only: the base URL is plain HTTP."""

    def __init__(self, settings: Settings, transport: httpx.BaseTransport | None = None):
        if not (settings.parse_app_id and settings.parse_rest_key):
            raise RuntimeError("PARSE_APP_ID and PARSE_REST_KEY must be set (see .env.example)")
        self._client = httpx.Client(
            base_url=settings.parse_base_url.rstrip("/") + "/1/",
            headers={
                "X-Parse-Application-Id": settings.parse_app_id,
                "X-Parse-REST-API-Key": settings.parse_rest_key,
            },
            timeout=30,
            transport=transport,
        )

    def count(self, cls: str, where: dict | None = None) -> int:
        params = {"count": 1, "limit": 0}
        if where:
            params["where"] = json.dumps(where)
        r = self._client.get(f"classes/{cls}", params=params)
        r.raise_for_status()
        return int(r.json().get("count", 0))

    def query(self, cls: str, where: dict | None = None, order: str | None = None) -> Iterator[dict]:
        skip = 0
        while True:
            params: dict = {"limit": PAGE_SIZE, "skip": skip}
            if where:
                params["where"] = json.dumps(where)
            if order:
                params["order"] = order
            r = self._client.get(f"classes/{cls}", params=params)
            r.raise_for_status()
            rows = r.json().get("results", [])
            yield from rows
            if len(rows) < PAGE_SIZE:
                return
            skip += PAGE_SIZE

    def close(self) -> None:
        self._client.close()


def _name(obj: dict | None) -> str | None:
    if not obj:
        return None
    return obj.get("name") or obj.get("title")


def _file_url(value) -> str | None:
    # Parse may return a bare URL or a File object ({"__type": "File", "url": ...}).
    if isinstance(value, dict):
        return value.get("url")
    return value or None


def build_taxonomy(client: ParseClient) -> dict[str, tuple[str | None, str | None, str | None]]:
    """Map category objectId -> (category name, product name, company name)."""
    companies = {c["objectId"]: c for c in client.query("Company")}
    products = {p["objectId"]: p for p in client.query("Products")}
    categories = {c["objectId"]: c for c in client.query("Category")}

    taxonomy = {}
    for cat_id, cat in categories.items():
        # Categories may nest; walk up until the parent is a product.
        parent, seen = cat.get("parent_object_id"), {cat_id}
        while parent in categories and parent not in seen:
            seen.add(parent)
            parent = categories[parent].get("parent_object_id")
        product = products.get(parent)
        company = companies.get(product.get("companyId")) if product else None
        taxonomy[cat_id] = (_name(cat), _name(product), _name(company))
    return taxonomy


def enumerate_pdfs(client: ParseClient, with_taxonomy: bool = True) -> list[MediaPDF]:
    taxonomy = build_taxonomy(client) if with_taxonomy else {}
    pdfs: dict[str, MediaPDF] = {}
    for row in client.query("Media", where=PDF_WHERE, order="createdAt"):
        url = _file_url(row.get("url"))
        if row.get("hidden") is True or not url:
            continue
        category, product, company = taxonomy.get(row.get("categoryId"), (None, None, None))
        pdfs[row["objectId"]] = MediaPDF(
            object_id=row["objectId"],
            title=(row.get("title") or "").strip() or "Untitled document",
            url=url,
            description=row.get("text") or "",
            category_id=row.get("categoryId"),
            category=category,
            product=product,
            company=company,
            updated_at=row.get("updatedAt"),
        )
    return list(pdfs.values())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--probe", default="0", help="number of PDFs to download and check for scans, or 'all'")
    parser.add_argument("--json", action="store_true", help="print the PDF list as JSON")
    args = parser.parse_args(argv)

    settings = get_settings()
    client = ParseClient(settings)
    try:
        server_total = client.count("Media", PDF_WHERE)
        pdfs = enumerate_pdfs(client)
    finally:
        client.close()

    print(f"Active PDFs on server: {server_total}")
    print(f"Visible PDFs (hidden filtered out): {len(pdfs)}")

    if args.json:
        json.dump([asdict(p) for p in pdfs], sys.stdout, indent=2)
        print()

    n = len(pdfs) if args.probe == "all" else int(args.probe)
    if n:
        from ingest.download import PdfStore
        from ingest.extract import scan_profile

        store = PdfStore(settings.data_dir)
        text_docs = partial = scanned = errors = 0
        for pdf in pdfs[:n]:
            try:
                path, _ = store.fetch(pdf)
                pages, ocr_pages = scan_profile(path)
            except Exception as e:  # noqa: BLE001 - report and keep going
                errors += 1
                print(f"  ! {pdf.title}: {e}")
                continue
            if ocr_pages == 0:
                text_docs += 1
            elif ocr_pages == pages:
                scanned += 1
                print(f"  scanned: {pdf.title} ({pages} pages)")
            else:
                partial += 1
                print(f"  partly scanned: {pdf.title} ({ocr_pages}/{pages} pages)")
        store.save()
        print(
            f"Probed {min(n, len(pdfs))}: {text_docs} digital text, {partial} partly scanned, "
            f"{scanned} fully scanned, {errors} errors"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
