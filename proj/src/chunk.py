"""
chunk.py
--------
Turns the page/section-level units from ingest.py into smaller, overlapping
chunks suitable for embedding and retrieval.

Chunking strategy (see write-up for the full rationale):
- Chunk by paragraph first, then greedily pack paragraphs into ~CHUNK_SIZE
  character windows. This keeps chunks topically coherent (we don't cut a
  sentence in half) while staying close to a target size.
- A small character overlap (OVERLAP) is kept between consecutive chunks
  from the same page/section so that an answer sitting right on a chunk
  boundary doesn't get orphaned.
- We never merge text across two different source pages/sections into one
  chunk. Losing a little packing efficiency this way is worth it: it keeps
  the "source" citation for every chunk exact instead of "pages 4-5ish".
"""

import json
from pathlib import Path

CHUNK_SIZE = 900       # target characters per chunk (~200-250 tokens)
OVERLAP = 150           # character overlap between consecutive chunks

IN_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "raw_docs.json"
OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "chunks.json"


def chunk_unit(text: str, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    """Greedy paragraph-packing chunker with character overlap."""
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks = []
    current = ""

    for para in paragraphs:
        if not current:
            current = para
        elif len(current) + 1 + len(para) <= chunk_size:
            current += "\n" + para
        else:
            chunks.append(current)
            # start new chunk with overlap tail of the previous one
            tail = current[-overlap:] if overlap < len(current) else current
            current = tail + "\n" + para

    if current:
        chunks.append(current)

    # Safety net: if a single paragraph is itself longer than chunk_size
    # (rare, but the constitution has some dense clauses), hard-split it.
    final = []
    for c in chunks:
        if len(c) <= chunk_size * 1.5:
            final.append(c)
        else:
            for i in range(0, len(c), chunk_size - overlap):
                piece = c[i:i + chunk_size]
                if piece.strip():
                    final.append(piece)
    return final


def build_chunks():
    with open(IN_PATH, encoding="utf-8") as fh:
        units = json.load(fh)

    all_chunks = []
    chunk_id = 0
    for unit in units:
        pieces = chunk_unit(unit["text"])
        for piece in pieces:
            all_chunks.append({
                "chunk_id": chunk_id,
                "source": unit["source"],
                "page": unit["page"],
                "text": piece,
            })
            chunk_id += 1

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(all_chunks, fh, ensure_ascii=False, indent=2)

    print(f"[chunk] {len(units)} unit(s) -> {len(all_chunks)} chunk(s) -> {OUT_PATH}")
    return all_chunks


if __name__ == "__main__":
    build_chunks()
