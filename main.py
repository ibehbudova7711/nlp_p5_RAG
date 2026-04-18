from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

import chromadb
import streamlit as st
from ollama import chat


CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "research_papers_rag")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
DEFAULT_TOP_K = int(os.getenv("TOP_K", "10"))
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "12000"))


st.set_page_config(page_title="Research Paper RAG", page_icon="📚", layout="wide")
st.title("📚 Research Paper RAG")
st.caption("Ask questions over your local Chroma DB using a local Ollama model")


@st.cache_resource
def load_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_collection(COLLECTION_NAME)


collection = load_collection()


def retrieve(question: str, top_k: int) -> Dict[str, Any]:
    results = collection.query(
        query_texts=[question],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    return results


def build_context(results: Dict[str, Any]) -> Tuple[str, List[Dict[str, Any]]]:
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0] if results.get("distances") else []

    sources: List[Dict[str, Any]] = []
    blocks: List[str] = []

    for i, doc in enumerate(docs):
        meta = metas[i] if i < len(metas) else {}
        distance = distances[i] if i < len(distances) else None

        title = meta.get("title", "Unknown title")
        arxiv_id = meta.get("arxiv_id", "Unknown ID")
        chunk_index = meta.get("chunk_index", "N/A")

        block = (
            f"[Source {i+1}]\n"
            f"Title: {title}\n"
            f"arXiv ID: {arxiv_id}\n"
            f"Chunk: {chunk_index}\n"
            f"Text:\n{doc}"
        )
        blocks.append(block)

        sources.append(
            {
                "title": title,
                "arxiv_id": arxiv_id,
                "chunk_index": chunk_index,
                "published": meta.get("published"),
                "pdf_url": meta.get("pdf_url"),
                "token_count": meta.get("token_count"),
                "distance": distance,
                "text": doc,
            }
        )

    context = "\n\n".join(blocks)[:MAX_CONTEXT_CHARS]
    return context, sources


def build_prompt(question: str, context: str) -> str:
    return f"""
You are a research paper question-answering assistant.

Answer the user's question only using the provided context.
If the answer is not supported by the context, say:
"Not found in the retrieved documents."

Rules:
- Be accurate and concise.
- Do not invent facts.
- When possible, mention methods, datasets, years, or findings from the context.
- Synthesize across sources if multiple chunks are relevant.
- Do not mention information that is missing from the context.

Context:
{context}

Question:
{question}
""".strip()


def generate_answer(question: str, context: str) -> str:
    prompt = build_prompt(question, context)

    response = chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ],
    )
    return response["message"]["content"].strip()


with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Top-k retrieved chunks", min_value=1, max_value=10, value=DEFAULT_TOP_K)
    show_sources = st.checkbox("Show sources", value=True)

    st.markdown("### Current config")
    st.code(
        f"CHROMA_DIR={CHROMA_DIR}\n"
        f"COLLECTION_NAME={COLLECTION_NAME}\n"
        f"OLLAMA_MODEL={OLLAMA_MODEL}"
    )

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()


if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_sources" not in st.session_state:
    st.session_state.last_sources = []


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


question = st.chat_input("Ask a question about your research papers")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.status("Retrieving and generating answer..."):
            try:
                results = retrieve(question, top_k=top_k)
                docs = results.get("documents", [[]])[0]

                if not docs:
                    answer = "Not found in the retrieved documents."
                    sources = []
                else:
                    context, sources = build_context(results)
                    answer = generate_answer(question, context)

                st.markdown(answer)
                st.session_state.last_sources = sources

            except Exception as e:
                answer = f"Error: {e}"
                st.error(answer)
                st.session_state.last_sources = []

    st.session_state.messages.append({"role": "assistant", "content": answer})


if show_sources and st.session_state.last_sources:
    st.subheader("Retrieved Sources")

    for i, src in enumerate(st.session_state.last_sources, start=1):
        title = src.get("title") or "Unknown title"
        arxiv_id = src.get("arxiv_id") or "Unknown ID"
        chunk_index = src.get("chunk_index")
        published = src.get("published")
        pdf_url = src.get("pdf_url")
        token_count = src.get("token_count")
        distance = src.get("distance")
        text = src.get("text") or ""

        with st.expander(f"{i}. {title} | {arxiv_id} | chunk {chunk_index}"):
            st.markdown(f"**Published:** {published}")
            st.markdown(f"**Token count:** {token_count}")
            st.markdown(f"**Distance:** {distance}")
            if pdf_url:
                st.markdown(f"**PDF:** {pdf_url}")
            st.markdown("**Chunk text:**")
            st.write(text)