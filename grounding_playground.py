import textwrap
from typing import List, Dict, Any, Optional

import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def _call_model(client, messages: List[Dict[str, str]]) -> Dict[str, Any]:
    if client is None:
        return {
            "ok": False,
            "content": "No active client – configure a backend and API key in the sidebar.",
            "usage": {},
            "cache": {},
        }
    return client.chat(messages=messages, temperature=0.2, max_tokens=512)


def _prompt_grounding_tab(client, model_label: str):
    st.markdown("### 1️⃣ Prompt-based grounding (direct append)")

    st.write(
        "Here we **directly append grounding text** (docs, guidelines) into the system context. "
        "This is the simplest form of grounding: no retrieval, no tools.[web:137][web:145]"
    )

    grounding_text = st.text_area(
        "Grounding text (e.g., an excerpt from Gemma docs, product policy, etc.)",
        height=180,
        value="You are a Gemma-focused assistant. Only answer about Gemma models using official documentation.",
    )
    user_q = st.text_input("User question", "What is Gemma 4 26B A4B good at?")
    if st.button("Ask with prompt grounding", key="ground_prompt_btn"):
        msgs = [
            {"role": "system", "content": grounding_text},
            {"role": "user", "content": user_q},
        ]
        with st.spinner("Calling model..."):
            res = _call_model(client, msgs)
        if not res.get("ok", False):
            st.error(res["content"])
            return
        st.markdown(f"**{model_label}:**")
        st.write(res["content"])
        with st.expander("Raw messages sent"):
            st.json(msgs)


def _tool_grounding_tab(client, model_label: str):
    st.markdown("### 2️⃣ Tool / MCP-style grounding")

    st.write(
        "Here we **pretend** we have a docs/Google-search MCP server.[web:145] "
        "In a real app, the model would call a tool to fetch docs; here you paste tool output "
        "to see how it changes the answer."
    )

    user_q = st.text_input(
        "User question",
        "How big is the context window for Gemma 4 models?",
        key="tool_q",
    )
    tool_output = st.text_area(
        "Simulated MCP tool output (e.g., search results / doc snippet)",
        height=160,
        value=textwrap.dedent(
            """
            [TOOL: docs_search]
            Gemma 4 supports long context windows. Public docs mention up to 128K tokens for server deployments,
            with smaller configurations on edge.[web:16][web:122]
            """
        ).strip(),
    )

    mode = st.radio(
        "How to inject tool output",
        [
            "Append as prior assistant message",
            "Include in system prompt",
        ],
        key="tool_mode",
    )

    if st.button("Ask with tool/MCP-style grounding", key="ground_tool_btn"):
        messages: List[Dict[str, str]] = []

        if mode == "Include in system prompt":
            system = (
                "You can use the following tool output as factual grounding for your answer.\n\n"
                + tool_output
            )
            messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": user_q})
        else:
            messages.append({"role": "user", "content": user_q})
            messages.append(
                {
                    "role": "assistant",
                    "content": "(Tool result injected into context)\n\n" + tool_output,
                }
            )

        with st.spinner("Calling model..."):
            res = _call_model(client, messages)
        if not res.get("ok", False):
            st.error(res["content"])
            return
        st.markdown(f"**{model_label}:**")
        st.write(res["content"])
        with st.expander("Raw messages sent"):
            st.json(messages)


def _rag_faiss_like_tab(client, model_label: str):
    st.markdown("### 3️⃣ RAG-style grounding with in-memory vector store")

    st.write(
        "This simulates a **RAG pipeline** using an in-memory vector index.[web:133][web:136][web:137] "
        "For simplicity we use TF‑IDF + cosine similarity here instead of FAISS; on your own infra you "
        "could replace this with FAISS indexes.[web:132][web:138][web:141]"
    )

    st.caption(
        "Workflow: chunk docs → embed → store in index → at query time, retrieve top‑k chunks → "
        "append them to the prompt + user question."
    )

    default_docs = textwrap.dedent(
        """
        [doc1] Gemma 4 overview:
        Gemma 4 is a multimodal model family with E2B, E4B, 26B A4B, and 31B variants. It targets on-device and server use cases.[web:16][web:122]

        [doc2] Gemma 4 26B A4B:
        26B A4B is a Mixture-of-Experts configuration: 4B parameters active per token, optimized for speed and quality on desktop/small servers.[web:16][web:128]

        [doc3] Gemma 4 31B:
        31B is a dense model for server deployments with strong reasoning and long context support.[web:16][web:124]

        [doc4] Grounding:
        Grounding means anchoring answers to external factual sources, often implemented via RAG or tool calls.[web:140][web:145]
        """
    ).strip()

    docs_text = st.text_area(
        "Documents to index (one corpus; you can paste your own)",
        height=220,
        value=default_docs,
    )

    user_q = st.text_input(
        "User question",
        "Compare Gemma 4 26B A4B and 31B for server deployments.",
        key="rag_q",
    )
    top_k = st.slider("Top‑k chunks to retrieve", 1, 5, 2, key="rag_k")

    if st.button("Ask with RAG-style grounding", key="ground_rag_btn"):
        # Simple splitting by blank lines
        raw_chunks = [c.strip() for c in docs_text.split("\n\n") if c.strip()]
        if not raw_chunks:
            st.error("No chunks found – add some text above.")
            return

        vectorizer = TfidfVectorizer()
        doc_matrix = vectorizer.fit_transform(raw_chunks)
        q_vec = vectorizer.transform([user_q])
        sims = cosine_similarity(q_vec, doc_matrix)[0]
        top_indices = np.argsort(sims)[::-1][:top_k]
        retrieved = [raw_chunks[i] for i in top_indices]

        st.markdown("**Retrieved chunks:**")
        for i, ch in zip(top_indices, retrieved):
            st.markdown(f"- chunk #{i}:")
            st.code(ch)

        # Build grounded prompt
        context_block = "\n\n".join(retrieved)
        system = (
            "You are grounded in the following documentation. Use it as your primary source.\n\n"
            + context_block
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_q},
        ]

        with st.spinner("Calling model..."):
            res = _call_model(client, messages)
        if not res.get("ok", False):
            st.error(res["content"])
            return
        st.markdown(f"**{model_label}:**")
        st.write(res["content"])
        with st.expander("Raw messages sent"):
            st.json(messages)

        st.caption(
            "On your own infra you can swap TF‑IDF for embeddings + FAISS; the flow (retrieve → append → generate) "
            "stays the same.[web:132][web:138][web:141]"
        )


def _hybrid_tab():
    st.markdown("### 4️⃣ Hybrid & other grounding mechanisms (conceptual)")

    st.write(
        "- **Prompt grounding**: cheap and simple; best when docs are small and stable.[web:137][web:140]  \n"
        "- **RAG (vector DB)**: scales to large doc sets; great when you have many pages of docs/KB.[web:133][web:136][web:137]  \n"
        "- **Tool/MCP grounding**: best when data lives in APIs/DBs (orders, configs, live metrics).[web:145]  \n"
        "- **Fine‑tuning**: changes model weights; use when you need style/format behavior that prompting+RAG can’t reach.[web:137][web:140]"
    )

    st.markdown(
        "In production, systems often **combine** these:\n"
        "- System prompt grounding for global rules & safety.\n"
        "- RAG for large unstructured text (docs, policies).\n"
        "- MCP/tools for structured/real‑time data.\n"
        "- Optional fine‑tune for narrow, high‑volume sub‑tasks.[web:137][web:145]"
    )


def render(client: Optional[object], model_label: str) -> None:
    st.subheader("🧷 Grounding playground – compare techniques")

    st.write(
        "Experiment with different grounding mechanisms and see how they change the model's answer while "
        "keeping the same question."
    )

    tabs = st.tabs(
        [
            "Prompt-based grounding",
            "Tool/MCP-style grounding",
            "RAG-style grounding",
            "Hybrid / theory",
        ]
    )

    with tabs[0]:
        _prompt_grounding_tab(client, model_label)
    with tabs[1]:
        _tool_grounding_tab(client, model_label)
    with tabs[2]:
        _rag_faiss_like_tab(client, model_label)
    with tabs[3]:
        _hybrid_tab()
