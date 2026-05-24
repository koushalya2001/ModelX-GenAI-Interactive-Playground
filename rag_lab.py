import textwrap
from typing import List, Tuple, Optional

import numpy as np
import streamlit as st
import faiss
from sentence_transformers import SentenceTransformer


@st.cache_resource
def get_embedding_model() -> SentenceTransformer:
    # Small, fast model that works well for demos.[web:18][web:24]
    return SentenceTransformer("all-MiniLM-L6-v2")


def _chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    words = text.split()
    chunks: List[str] = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start = end - overlap if overlap > 0 else end
        if start < 0:
            start = 0
    return [c for c in chunks if c.strip()]


def _build_faiss_index(chunks: List[str]) -> Tuple[faiss.IndexFlatL2, np.ndarray]:
    model = get_embedding_model()
    embeddings = model.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
    dim = embeddings.shape[1]
    # Simple, exact L2 index for teaching.[web:18]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings.astype("float32"))
    return index, embeddings


def _search_faiss(
    index: faiss.IndexFlatL2,
    query: str,
    chunks: List[str],
    k: int,
) -> List[Tuple[int, float, str]]:
    model = get_embedding_model()
    q_emb = model.encode([query], convert_to_numpy=True, show_progress_bar=False)
    distances, indices = index.search(q_emb.astype("float32"), k)
    results: List[Tuple[int, float, str]] = []
    for rank, (idx, dist) in enumerate(zip(indices[0], distances[0]), start=1):
        if idx < 0 or idx >= len(chunks):
            continue
        results.append((idx, float(dist), chunks[idx]))
    return results


def render() -> None:
    st.subheader("📚 RAG & Vector DB Lab (with FAISS demo)")

    st.markdown(
        "This lab now includes a **real FAISS-backed vector index** so you can feel how chunking, "
        "top‑k, and corpus size affect retrieval and prompt footprint.[web:18][web:25]"
    )

    # High-level conceptual controls
    rag_type = st.selectbox(
        "RAG strategy",
        [
            "None (direct LLM)",
            "Naïve: top-k similarity search",
            "Multi-vector / hybrid (sparse + dense, concept only)",
            "Hierarchical / multi-stage (concept only)",
        ],
    )

    vector_db = st.selectbox(
        "Vector DB (actual demo uses FAISS)",
        [
            "None",
            "FAISS (in-process, this demo)",
            "Cloud service (Pinecone / Qdrant / Weaviate, concept only)",
        ],
    )

    chunk_tokens = st.slider(
        "Approx. chunk size (words ≈ tokens here)",
        min_value=64,
        max_value=512,
        value=256,
        step=32,
    )
    chunk_overlap = st.slider(
        "Chunk overlap (words)",
        min_value=0,
        max_value=128,
        value=32,
        step=16,
    )
    top_k = st.slider("Top-k retrieved chunks", min_value=1, max_value=10, value=4)

    st.markdown("### 1️⃣ Define a tiny corpus")

    default_corpus = textwrap.dedent(
        """
        Gemma models are lightweight, open models designed for a wide range of tasks including chat, code, and reasoning.
        Smaller variants like Gemma 2B are ideal for edge and education use-cases where cost and latency matter.

        Retrieval-Augmented Generation (RAG) reduces hallucinations by retrieving relevant context from a knowledge base
        and injecting it into the prompt before calling the LLM.

        Vector databases like FAISS index high-dimensional embeddings so we can efficiently search for semantically
        similar chunks instead of relying on plain keyword matching.[web:18][web:23]

        Chunk size and top-k are critical design parameters: too small and you lose context; too large and you blow up
        prompt length and KV cache usage.
        """
    ).strip()

    corpus_text = st.text_area(
        "Paste or edit your small knowledge base (a few paragraphs).",
        value=default_corpus,
        height=220,
    )

    if "rag_chunks" not in st.session_state:
        st.session_state.rag_chunks = []
    if "rag_index" not in st.session_state:
        st.session_state.rag_index = None

    col_build, col_info = st.columns([1, 2])
    with col_build:
        if st.button("Build / Rebuild FAISS index"):
            chunks = _chunk_text(corpus_text, chunk_size=chunk_tokens, overlap=chunk_overlap)
            if not chunks:
                st.error("No chunks produced – try lowering chunk size or adding more text.")
            else:
                index, _ = _build_faiss_index(chunks)
                st.session_state.rag_chunks = chunks
                st.session_state.rag_index = index
                st.success(f"Indexed {len(chunks)} chunks with FAISS.")

    with col_info:
        if st.session_state.rag_chunks:
            est_tokens = len(st.session_state.rag_chunks) * chunk_tokens
            st.markdown(
                f"- **Chunks in index**: `{len(st.session_state.rag_chunks)}`  \n"
                f"- Approx. tokens covered by corpus (words≈tokens): `{est_tokens}`  \n"
                f"- At `top-k = {top_k}`, a single query can add up to ~"
                f"`{chunk_tokens * top_k}` tokens into your prompt."
            )

    st.markdown("### 2️⃣ Run a query against the FAISS index")

    query = st.text_input(
        "Query (semantic search)",
        value="What is RAG and why use a vector database?",
    )

    if st.button("Retrieve context with FAISS"):
        if not st.session_state.rag_index or not st.session_state.rag_chunks:
            st.error("Build the FAISS index first.")
        else:
            results = _search_faiss(
                st.session_state.rag_index,
                query,
                st.session_state.rag_chunks,
                k=top_k,
            )
            if not results:
                st.warning("No results returned – check your corpus or query.")
            else:
                st.markdown("**Top retrieved chunks:**")
                for rank, (idx, dist, chunk) in enumerate(results, start=1):
                    with st.expander(f"#{rank} – chunk {idx} (distance {dist:.3f})"):
                        st.write(chunk)

                approx_prompt_tokens = sum(len(c.split()) for _, _, c in results)
                st.info(
                    f"Approx. prompt tokens contributed by retrieved context: ~{approx_prompt_tokens} "
                    f"(words used as a rough proxy)."
                )

    st.markdown("### 3️⃣ What this teaches")

    st.write(
        "- You can see how **chunk size** and **top-k** interact: bigger chunks or higher k quickly blow up the "
        "context you must send to the LLM.\n"
        "- FAISS gives near‑instant retrieval in this tiny demo; at larger scale, index choice (Flat vs IVF/HNSW, "
        "CPU vs GPU) becomes a major design decision.[web:18][web:25][web:29]\n"
        "- This lab is intentionally simple and in‑process, mirroring the conceptual RAG diagrams from the main app "
        "while grounding them in a real vector search implementation."
    )