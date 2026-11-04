"""File-backed metadata store, as an alternative to MongoDB.

``services.search`` needs exactly two things from its metadata store: fetch many
documents by ``meme_id`` in one call, and answer whether it is reachable. Mongo
is the right choice when metadata is being corrected or reloaded independently
of the indexes, which is the point of keeping the two apart (see
``docs/ARCHITECTURE.md``). But it is a whole service to stand up, and for a
read-only deployment — a demo, a container with no database alongside it — the
metadata never changes after the indexes are built.

:class:`JsonlCollection` reads ``memes_final.jsonl`` into memory once and serves
the same narrow interface, so the search path cannot tell the difference. 6,989
documents is a few MB of dicts.

Select it with ``MEME_METADATA_BACKEND=jsonl``; the default stays ``mongo``.
"""

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any


def flatten_record(record: dict[str, Any]) -> dict[str, Any]:
    """Map a ``memes_final.jsonl`` record to the document shape search expects.

    This is deliberately the same mapping that
    ``scripts/pipeline/store_metadata.py`` applies when it upserts into MongoDB.
    The two must agree: the search path reads ``ocr_text``, while the JSONL
    nests it under ``text.ocr``, so a drift here silently costs every result its
    caption instead of raising.
    """
    return {
        "meme_id": record["meme_id"],
        "image_path": record.get("image_path", ""),
        "ocr_text": (record.get("text") or {}).get("ocr", ""),
        "vector_id": record.get("vector_id"),
        "width": record.get("width"),
        "height": record.get("height"),
    }


class JsonlCollection:
    """Read-only stand-in for a MongoDB collection, backed by a JSONL file.

    Implements only what :mod:`services.search` calls: :meth:`find` with an
    ``$in`` filter on ``meme_id``. Unknown ids are skipped rather than raising,
    matching Mongo's behaviour and letting the search path drop hits whose
    metadata is missing.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(
                f"Metadata file not found at {self.path}. "
                "Run scripts/pipeline/embed_memes_fast.py to build it, or set "
                "MEME_METADATA to point at an existing one."
            )
        self._docs: dict[str, dict[str, Any]] = {}
        with open(self.path) as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                doc = flatten_record(json.loads(line))
                self._docs[doc["meme_id"]] = doc

    def __len__(self) -> int:
        return len(self._docs)

    def find(
        self, query: dict[str, Any], projection: dict[str, Any] | None = None
    ) -> Iterator[dict[str, Any]]:
        """Return the documents whose ``meme_id`` appears in the ``$in`` list.

        ``projection`` is accepted for interface compatibility and ignored: the
        documents are already narrow, and the only projection the search path
        passes is ``{"_id": 0}``, which excludes a field this store never has.
        """
        wanted: Iterable[str] = query.get("meme_id", {}).get("$in", [])
        for meme_id in wanted:
            doc = self._docs.get(meme_id)
            if doc is not None:
                yield dict(doc)

    def count_documents(self, _filter: dict[str, Any] | None = None) -> int:
        return len(self._docs)
