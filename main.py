from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

import chromadb
import streamlit as st
from openai import OpenAI

# CONFIG

CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "research_papers_rag")
DEFAULT_TOP_K = int(os.getenv("TOP_K", "5"))
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "6000"))

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = "gpt-4o-mini"

client = OpenAI(api_key=OPENAI_API_KEY)

# UI

st.set_page_config(page_title="Research Paper RAG", page_icon="📚", layout="wide")
st.title("📚 Research Paper RAG (OpenAI)")
st.caption("Ask questions over your local Chroma DB using OpenAI")

@st.cache_resource
def load_collection():
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    return chroma_client.get_collection(COLLECTION_NAME)

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
Keep the answer simple and focused. 
Avoid mentioning unrelated concepts.
Do not mention components unless explicitly stated in the context.

Rules:

* Be accurate and concise.
* Do not invent facts.
* Mention both What and Why.
* Mention concrete details from the context when possible.
* Combine multiple sources if needed.

Context:
{context}

Question:
{question}
""".strip()

def generate_answer(question: str, context: str) -> str:
    prompt = build_prompt(question, context)

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
     messages=[
         {"role": "user", "content": prompt}
     ],
     temperature=0.2,
    )

    return response.choices[0].message.content.strip()

# SIDEBAR

with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Top-k retrieved chunks", 1, 10, DEFAULT_TOP_K)
    show_sources = st.checkbox("Show sources", True)


st.markdown("### Current config")
st.code(
    f"CHROMA_DIR={CHROMA_DIR}\n"
    f"COLLECTION_NAME={COLLECTION_NAME}\n"
    f"MODEL={OPENAI_MODEL}"
)

if st.button("Clear chat"):
    st.session_state.messages = []
    st.rerun()


# STATE

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_sources" not in st.session_state:
    st.session_state.last_sources = []

# CHAT HISTORY

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# INPUT

question = st.chat_input("Ask a question about your research papers")

if question:
    st.session_state.messages.append({"role": "user", "content": question})


with st.chat_message("user"):
    st.markdown(question)

with st.chat_message("assistant"):
    with st.status("Thinking..."):
        try:
            results = retrieve(question, top_k)
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


# SOURCES

if show_sources and st.session_state.last_sources:
    st.subheader("Retrieved Sources")


for i, src in enumerate(st.session_state.last_sources, start=1):
    with st.expander(f"{i}. {src.get('title')}"):
        st.markdown(f"**arXiv ID:** {src.get('arxiv_id')}")
        st.markdown(f"**Chunk:** {src.get('chunk_index')}")
        st.markdown(f"**Distance:** {src.get('distance')}")
        if src.get("pdf_url"):
            st.markdown(f"**PDF:** {src.get('pdf_url')}")
        st.write(src.get("text"))

