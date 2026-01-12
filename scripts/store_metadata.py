import json
from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017")
db = client["meme_search"]
col = db["memes"]

with open("data/memes_final.jsonl") as f:
    for line in f:
        doc = json.loads(line)
        col.insert_one({
            "meme_id": doc["meme_id"],
            "image_path": doc["image_path"],
            "ocr_text": doc["text"]["ocr"],
            "vector_id": doc["vector_id"]
        })

print("Metadata stored")