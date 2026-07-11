"""
app.py
------
Streamlit web interface for EnPoWER Insti-Assist.

Run with:  streamlit run app.py
Requires:  ANTHROPIC_API_KEY set in the environment (or a .env file, see README).
"""

import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from rag import EnpowerRAG  # noqa: E402

st.set_page_config(page_title="EnPoWER Insti-Assist", page_icon="🔬", layout="centered")

st.title("🔬 EnPoWER Insti-Assist")
st.caption(
    "A RAG-powered assistant grounded in EnPoWER's work reports, selection "
    "assignments, and the IIT Bombay Students' Gymkhana constitution. "
    "It only answers from those documents, and says so when it can't."
)

if not os.environ.get("ANTHROPIC_API_KEY"):
    st.warning(
        "ANTHROPIC_API_KEY is not set. The app will still retrieve and show "
        "matching source passages, but it can't generate an LLM answer until "
        "you set the key (see README).",
        icon="⚠️",
    )


@st.cache_resource(show_spinner="Loading index and embedding model...")
def load_rag():
    return EnpowerRAG()


try:
    rag = load_rag()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.subheader("About")
    st.write(
        "EnPoWER is IIT Bombay's undergraduate research-promotion council. "
        "This assistant is grounded in its work reports (2023-26), selection "
        "assignments, and the governing Students' Gymkhana constitution."
    )
    top_k = st.slider("Chunks retrieved per query (k)", 1, 10, 5)
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander(f"📚 Sources ({len(msg['sources'])})"):
                for s in msg["sources"]:
                    st.markdown(f"**{s['source']}** — {s['page']}  \n"
                                f"*similarity: {s['score']:.2f}*")
                    st.caption(s["text"][:300] + ("..." if len(s["text"]) > 300 else ""))

query = st.chat_input("Ask about EnPoWER, its initiatives, or the Gymkhana constitution...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and generating..."):
            result = rag.answer(query, top_k=top_k)
        st.markdown(result["answer"])

        badge = "🟢 Grounded" if result["grounded"] else "🔴 Not found in documents"
        st.caption(f"{badge} · top similarity: {result['confidence']:.2f}")

        if result["sources"]:
            with st.expander(f"📚 Sources ({len(result['sources'])})"):
                for s in result["sources"]:
                    st.markdown(f"**{s['source']}** — {s['page']}  \n"
                                f"*similarity: {s['score']:.2f}*")
                    st.caption(s["text"][:300] + ("..." if len(s["text"]) > 300 else ""))

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"],
    })
