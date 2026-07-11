"""
build_index.py
---------------
Embeds every chunk in data/processed/chunks.json with a sentence-transformers
model and builds a FAISS index for fast cosine-similarity search.

Run this once after chunk.py (or whenever the source documents / chunking
strategy change). It writes two files into index/:
  - enpower.faiss   -> the FAISS index itself
  - metadata.json   -> chunk_id -> {source, page, text} lookup, in the same
                        order the vectors were added, so a FAISS row id
                        maps directly to metadata[row_id]
"""

import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

CHUNKS_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "chunks.json"
INDEX_DIR = Path(__file__).resolve().parent.parent / "index"
INDEX_PATH = INDEX_DIR / "enpower.faiss"
META_PATH = INDEX_DIR / "metadata.json"


def build_index():
    with open(CHUNKS_PATH, encoding="utf-8") as fh:
        chunks = json.load(fh)

    texts = [c["text"] for c in chunks]
    print(f"[build_index] Embedding {len(texts)} chunks with {EMBED_MODEL_NAME} ...")

    model = SentenceTransformer(EMBED_MODEL_NAME)
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,  # so inner product == cosine similarity
    )
    embeddings = np.asarray(embeddings, dtype="float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product on normalized vecs = cosine sim
    index.add(embeddings)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))

    with open(META_PATH, "w", encoding="utf-8") as fh:
        json.dump(chunks, fh, ensure_ascii=False, indent=2)

    print(f"[build_index] Wrote {index.ntotal} vectors -> {INDEX_PATH}")
    print(f"[build_index] Wrote metadata -> {META_PATH}")


if __name__ == "__main__":
    build_index()
