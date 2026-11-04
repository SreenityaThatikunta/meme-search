"""Load meme metadata into MongoDB.

Idempotent: re-running upserts on ``meme_id`` instead of inserting duplicates,
and ensures the unique index that the search path queries on.
"""

import json
import sys
from pathlib import Path

from pymongo import ASCENDING, MongoClient, UpdateOne

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import config  # noqa: E402 - needs the project root on sys.path first

BATCH_SIZE = 1000


def main() -> None:
    if not config.METADATA_PATH.exists():
        raise SystemExit(
            f"{config.METADATA_PATH} not found. Run scripts/pipeline/embed_memes_fast.py first."
        )

    client = MongoClient(config.MONGO_URI)
    collection = client[config.MONGO_DB][config.MONGO_COLLECTION]
    collection.create_index([("meme_id", ASCENDING)], unique=True)

    operations = []
    upserted = 0

    def flush() -> int:
        if not operations:
            return 0
        result = collection.bulk_write(operations, ordered=False)
        operations.clear()
        return result.upserted_count + result.modified_count

    with open(config.METADATA_PATH) as handle:
        for line in handle:
            doc = json.loads(line)
            operations.append(
                UpdateOne(
                    {"meme_id": doc["meme_id"]},
                    {
                        "$set": {
                            "image_path": doc["image_path"],
                            "ocr_text": doc["text"]["ocr"],
                            "vector_id": doc["vector_id"],
                            "width": doc.get("width"),
                            "height": doc.get("height"),
                        }
                    },
                    upsert=True,
                )
            )
            if len(operations) >= BATCH_SIZE:
                upserted += flush()

    upserted += flush()
    print(f"Stored metadata for {upserted} memes in {config.MONGO_DB}.{config.MONGO_COLLECTION}")
    print(f"Collection now holds {collection.count_documents({})} documents")


if __name__ == "__main__":
    main()
