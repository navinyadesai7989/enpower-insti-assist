# EnPoWER Insti-Assist

A RAG-powered AI assistant grounded in the real documents of **EnPoWER** — IIT
Bombay's undergraduate research-promotion council (the *Engineering Oriented
Promotion of Work Experience and Research* initiative) — plus the governing
**Students' Gymkhana Constitution**.

Scope chosen: **Council/Club Assistant** (per the assignment's scope options).

The assistant answers questions like *"What did EnPoWER do for SURP in
2025?"* or *"What does the constitution say about council elections?"*
using only retrieved passages from the source documents below, and says
**"I don't know"** when the answer isn't in them — it never falls back on
the underlying LLM's general knowledge.

---

## 1. Data sources (`data/raw/`)

| File | What it is |
|---|---|
| `EndTerm Work Report 2023-24 - EnPoWER.pdf` | Council's end-of-year report, 2023-24 |
| `EnPoWER EndTerm Work Report 2024-25.docx` | Council's end-of-year report, 2024-25 |
| `EnPoWER EndTerm Work Report 2025-26 - Final.pdf` | Council's end-of-year report, 2025-26 |
| `EnPoWER MidTerm Work Report 2025-26.pdf` | Council's mid-year report, 2025-26 |
| `EnPoWER Assignment_Round 2_UGAC.pdf` | Selection assignment for Academic Coordinator applicants |
| `Assignment- DRCs - 2026-27.docx` | Selection assignment for Department Research Coordinator applicants |
| `SAC-Constitution-March-2018.pdf` | IIT Bombay Students' Gymkhana Constitution (81 pages; the umbrella governance document EnPoWER and every other council operates under) |

7 source documents, comfortably over the 5-document minimum.

## 2. Architecture

```
data/raw/*.pdf,*.docx
        │  src/ingest.py        (pdfplumber / python-docx → page/section-tagged text)
        ▼
data/processed/raw_docs.json
        │  src/chunk.py         (paragraph-packing chunker, ~900 chars, 150 char overlap)
        ▼
data/processed/chunks.json
        │  src/build_index.py   (sentence-transformers all-MiniLM-L6-v2 → FAISS IndexFlatIP)
        ▼
index/enpower.faiss + index/metadata.json
        │  src/rag.py           (embed query → top-k retrieval → similarity gate
        │                        → grounded prompt → Claude API → cited answer)
        ▼
app.py  (Streamlit chat UI, shows answer + expandable source chunks)
```

## 3. Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env            # then paste your Anthropic API key into .env
```

The first run of `build_index.py` downloads the small (~90 MB)
`all-MiniLM-L6-v2` embedding model from Hugging Face — this requires normal
internet access.

## 4. Build the index (run once, or whenever data/raw/ changes)

```bash
python src/ingest.py        # -> data/processed/raw_docs.json
python src/chunk.py         # -> data/processed/chunks.json
python src/build_index.py   # -> index/enpower.faiss, index/metadata.json
```

## 5. Run

**Web UI:**
```bash
streamlit run app.py
```

**Command line:**
```bash
python src/rag.py
```

## 6. How grounding / refusal works

1. The query is embedded and matched against the FAISS index (top-`k=5` by
   cosine similarity).
2. If the *best* match scores below `MIN_SIMILARITY = 0.30`, the assistant
   never even calls the LLM — it returns "I don't know..." directly. This
   catches queries that are simply off-topic for the corpus (e.g. "what's
   the mess menu today?").
3. If the similarity gate passes, the retrieved chunks are injected into a
   system prompt that explicitly forbids the model from using outside
   knowledge and instructs it to say "I don't know based on the available
   EnPoWER documents." if the context still doesn't answer the question.
4. Every answer in the UI is shown with a 🟢/🔴 grounded badge, a numeric
   top-similarity confidence score, and an expandable list of the exact
   source document + page/section for every chunk that was used.

## 7. Known limitations

- The `.docx` files are chunked by heading; the DRC assignment doc has no
  heading styles, so it falls back to one large section that the
  paragraph-packing chunker splits further, at the cost of a coarser
  citation label ("Document start").
- The similarity gate threshold (0.30) was picked by inspection, not tuned
  against a labeled eval set — see write-up for what a proper eval would
  look like.
- No OCR: any scanned (image-only) page would extract as empty text. All
  source PDFs here are text-native, so this wasn't an issue in practice.
- No multi-turn memory yet (bonus item, not implemented in this pass).
