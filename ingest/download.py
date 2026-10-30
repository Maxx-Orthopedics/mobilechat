"""Download PDFs into a local cache and track content hashes + index state.

The manifest (`data/manifest.json`) is keyed by Media objectId and records the
source URL, `updatedAt`, file SHA-256, and the key of the version last indexed.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

import httpx

from ingest.enumerate import MediaPDF


class PdfStore:
    def __init__(self, data_dir: Path, client: httpx.Client | None = None):
        self.pdf_dir = Path(data_dir) / "pdfs"
        self.pdf_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = Path(data_dir) / "manifest.json"
        self.manifest: dict[str, dict] = (
            json.loads(self.manifest_path.read_text()) if self.manifest_path.exists() else {}
        )
        self._client = client or httpx.Client(follow_redirects=True, timeout=120)

    def path_for(self, object_id: str) -> Path:
        return self.pdf_dir / f"{object_id}.pdf"

    def fetch(self, pdf: MediaPDF) -> tuple[Path, str]:
        """Return (local path, sha256). Skips the download if the source is unchanged."""
        entry = self.manifest.get(pdf.object_id, {})
        path = self.path_for(pdf.object_id)
        if (
            path.exists()
            and entry.get("sha256")
            and entry.get("url") == pdf.url
            and entry.get("updated_at") == pdf.updated_at
        ):
            return path, entry["sha256"]

        digest = hashlib.sha256()
        fd, tmp = tempfile.mkstemp(dir=self.pdf_dir, suffix=".part")
        try:
            with os.fdopen(fd, "wb") as f, self._client.stream("GET", pdf.url) as r:
                r.raise_for_status()
                first = True
                for block in r.iter_bytes():
                    if first and block:
                        if not block.lstrip()[:5].startswith(b"%PDF"):
                            raise ValueError(f"not a PDF (got {r.headers.get('content-type')})")
                        first = False
                    digest.update(block)
                    f.write(block)
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

        sha = digest.hexdigest()
        entry.update(url=pdf.url, updated_at=pdf.updated_at, sha256=sha)
        self.manifest[pdf.object_id] = entry
        return path, sha

    def is_indexed(self, object_id: str, key: str) -> bool:
        return self.manifest.get(object_id, {}).get("indexed_key") == key

    def mark_indexed(self, object_id: str, key: str) -> None:
        self.manifest.setdefault(object_id, {})["indexed_key"] = key

    def forget(self, object_id: str) -> None:
        self.manifest.pop(object_id, None)
        self.path_for(object_id).unlink(missing_ok=True)

    def ids(self) -> set[str]:
        return set(self.manifest)

    def save(self) -> None:
        tmp = self.manifest_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.manifest, indent=2, sort_keys=True))
        os.replace(tmp, self.manifest_path)
