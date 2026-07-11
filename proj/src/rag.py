"""
rag.py
------
The retrieval + generation core of EnPoWER Insti-Assist.

Pipeline for a single user query:
  1. Embed the query with the same sentence-transformers model used to
     build the index.
  2. Retrieve the top-k most similar chunks from FAISS.
  3. Apply a similarity-score gate: if even the best chunk is too dissimilar
     to the query, we don't bother calling the LLM at all -- we just say
     "I don't know" and show nothing. This is what keeps the assistant
     honest about its own knowledge boundary (bonus: confidence indicator).
  4. Otherwise, build a strict grounded-answering prompt (retrieved chunks
     + explicit "only use this context, say I don't know otherwise"
     instructions) and call the Claude API.
  5. Return the answer together with the exact chunks (source + page) that
     were used, so the UI can render citations.
"""

import json
import os
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import anthropic

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
CLAUDE_MODEL = "claude-sonnet-4-6"

TOP_K = 5
# Cosine similarity gate: below this, we treat the corpus as "not containing
# the answer" rather than trust the LLM to notice weak context on its own.
MIN_SIMILARITY = 0.30

INDEX_DIR = Path(__file__).resolve().parent.parent / "index"
INDEX_PATH = INDEX_DIR / "enpower.faiss"
META_PATH = INDEX_DIR / "metadata.json"

SYSTEM_PROMPT = """You are EnPoWER Insti-Assist, a question-answering assistant for \
EnPoWER -- IIT Bombay's undergraduate research-promotion council (the Engineering \
Oriented Promotion of Work Experience and Research initiative) -- and the \
Students' Gymkhana constitution that governs it.

Rules you must follow exactly:
1. Answer ONLY using the information in the "Retrieved context" block below. \
Do not use outside knowledge, even if you happen to know something about IIT \
Bombay or EnPoWER from training.
2. If the retrieved context does not contain enough information to answer the \
question, reply exactly: "I don't know based on the available EnPoWER documents." \
Do not guess or fill gaps.
3. When you do answer, cite which source document(s) you drew from inline, e.g. \
"(EnPoWER MidTerm Work Report 2025-26.pdf)".
4. Be concise and factual. Do not editorialize.
"""


class EnpowerRAG:
    def __init__(self):
        if not INDEX_PATH.exists():
            raise FileNotFoundError(
                f"No index found at {INDEX_PATH}. Run src/ingest.py, src/chunk.py "
                f"and src/build_index.py first."
            )
        self.index = faiss.read_index(str(INDEX_PATH))
        with open(META_PATH, encoding="utf-8") as fh:
            self.metadata = json.load(fh)
        self.embed_model = SentenceTransformer(EMBED_MODEL_NAME)

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        self.client = anthropic.Anthropic(api_key=api_key) if api_key else None

    def retrieve(self, query: str, top_k: int = TOP_K):
        q_vec = self.embed_model.encode([query], normalize_embeddings=True)
        q_vec = np.asarray(q_vec, dtype="float32")
        scores, ids = self.index.search(q_vec, top_k)
        results = []
        for score, idx in zip(scores[0], ids[0]):
            if idx == -1:
                continue
            chunk = self.metadata[idx]
            results.append({**chunk, "score": float(score)})
        return results

    def build_prompt(self, query: str, chunks):
        context_block = "\n\n".join(
            f"[Source: {c['source']} | {c['page']}]\n{c['text']}"
            for c in chunks
        )
        user_msg = (
            f"Retrieved context:\n{context_block}\n\n"
            f"Question: {query}"
        )
        return user_msg

    def answer(self, query: str, top_k: int = TOP_K):
        chunks = self.retrieve(query, top_k=top_k)

        if not chunks or chunks[0]["score"] < MIN_SIMILARITY:
            return {
                "answer": "I don't know based on the available EnPoWER documents.",
                "grounded": False,
                "sources": [],
                "confidence": chunks[0]["score"] if chunks else 0.0,
            }

        if self.client is None:
            return {
                "answer": "[ANTHROPIC_API_KEY not set -- showing retrieved context "
                          "only, no LLM call made]",
                "grounded": True,
                "sources": chunks,
                "confidence": chunks[0]["score"],
            }

        user_msg = self.build_prompt(query, chunks)
        response = self.client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=700,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
        answer_text = "".join(
            block.text for block in response.content if block.type == "text"
        )

        refused = "i don't know" in answer_text.lower()

        return {
            "answer": answer_text,
            "grounded": not refused,
            "sources": [] if refused else chunks,
            "confidence": chunks[0]["score"],
        }


if __name__ == "__main__":
    rag = EnpowerRAG()
    while True:
        q = input("\nAsk EnPoWER Insti-Assist ('quit' to exit): ")
        if q.strip().lower() in {"quit", "exit"}:
            break
        result = rag.answer(q)
        print("\n--- Answer ---")
        print(result["answer"])
        if result["sources"]:
            print("\n--- Sources ---")
            for s in result["sources"]:
                print(f"  - {s['source']} ({s['page']}), score={s['score']:.2f}")
