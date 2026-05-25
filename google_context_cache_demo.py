import time
from typing import Optional

import requests
import streamlit as st


BASE_URL = "https://generativelanguage.googleapis.com/v1beta"


def _create_cached_content(api_key: str, model_id: str, big_context: str) -> dict:
    url = f"{BASE_URL}/cachedContents"
    params = {"key": api_key}
    payload = {
        "model": f"models/{model_id}",
        "contents": [
            {
                "role": "user",
                "parts": [{"text": big_context}],
            }
        ],
    }
    t0 = time.perf_counter()
    resp = requests.post(url, params=params, json=payload, timeout=60)
    t1 = time.perf_counter()
    data = resp.json()
    data["_latency_s"] = t1 - t0
    return {"status_code": resp.status_code, "data": data}


def _call_without_cache(api_key: str, model_id: str, big_context: str, question: str) -> dict:
    url = f"{BASE_URL}/models/{model_id}:generateContent"
    params = {"key": api_key}
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": big_context + "\n\nQuestion: " + question}],
            }
        ],
        "generationConfig": {"temperature": 0.2},
    }
    t0 = time.perf_counter()
    resp = requests.post(url, params=params, json=payload, timeout=60)
    t1 = time.perf_counter()
    data = resp.json()
    data["_latency_s"] = t1 - t0
    return {"status_code": resp.status_code, "data": data}


def _call_with_cache(api_key: str, model_id: str, cached_name: str, question: str) -> dict:
    url = f"{BASE_URL}/models/{model_id}:generateContent"
    params = {"key": api_key}
    payload = {
        "cachedContent": cached_name,
        "contents": [
            {
                "role": "user",
                "parts": [{"text": question}],
            }
        ],
        "generationConfig": {"temperature": 0.2},
    }
    t0 = time.perf_counter()
    resp = requests.post(url, params=params, json=payload, timeout=60)
    t1 = time.perf_counter()
    data = resp.json()
    data["_latency_s"] = t1 - t0
    return {"status_code": resp.status_code, "data": data}


def _extract_text_and_usage(data: dict) -> tuple[str, dict]:
    text = ""
    candidates = data.get("candidates") or []
    if candidates:
        parts = candidates[0].get("content", {}).get("parts", [])
        text = "".join(p.get("text", "") for p in parts)

    usage = data.get("usageMetadata") or {}
    # Typical fields: promptTokenCount, candidatesTokenCount, totalTokenCount, cachedContentTokenCount.[web:159]
    return text, usage


def render(model_id: Optional[str], google_api_key: Optional[str]) -> None:
    st.subheader("🧪 Google AI Studio context caching demo")

    if not google_api_key or not model_id:
        st.info(
            "Select **Google AI Studio** as the active backend in the sidebar and provide an API key and model "
            "to use this demo."
        )
        return

    st.write(
        "This page shows how to:\n"
        "1) Create a cached context with a large, stable prompt.\n"
        "2) Compare a normal call vs a call that references that cached context.\n\n"
        "You should see non-zero `cachedContentTokenCount` only for the cached call.[web:159]"
    )

    big_context = st.text_area(
        "Big, stable context (e.g., docs / long prompt)",
        height=200,
        value=(
            "You are a Gemma 4 expert assistant. Use only the information in the official Gemma 4 docs at "
            "ai.google.dev/gemma when answering questions about capabilities and deployment.[web:122][web:16][web:128]"
        ),
    )

    question = st.text_input(
        "User question for comparison",
        "Explain when to choose Gemma 4 26B A4B vs 31B.",
    )

    if "google_cached_name" not in st.session_state:
        st.session_state.google_cached_name = None

    st.markdown("### Step 1 – Create / refresh cached context")
    if st.button("Create cached context now"):
        with st.spinner("Creating cached context via /cachedContents..."):
            result = _create_cached_content(google_api_key, model_id, big_context)
        if result["status_code"] != 200:
            st.error(f"Error creating cached content: {result['data']}")
        else:
            name = result["data"].get("name")
            st.session_state.google_cached_name = name
            st.success(f"Cached context created: `{name}` (latency {result['data']['_latency_s']:.2f}s)")
            st.caption(
                "This name is what you reference in `cachedContent` for later generateContent calls.[web:159]"
            )

    st.markdown("### Step 2 – Compare calls")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### A) Without cache (normal prompt)")
        if st.button("Call without cache"):
            with st.spinner("Calling generateContent without cachedContent..."):
                result = _call_without_cache(google_api_key, model_id, big_context, question)
            if result["status_code"] != 200:
                st.error(result["data"])
            else:
                text, usage = _extract_text_and_usage(result["data"])
                st.markdown("**Model answer:**")
                st.write(text)
                st.markdown("**Usage metadata:**")
                st.json(usage)
                st.caption(f"Latency: {result['data']['_latency_s']:.2f}s")

    with col2:
        st.markdown("#### B) With cache (using cachedContent)")

        cached_name = st.session_state.google_cached_name
        st.text_input(
            "Cached content name",
            value=cached_name or "",
            key="cached_name_display",
            help="Comes from Step 1; must look like cachedContents/...",
        )

        if st.button("Call with cache"):
            if not cached_name:
                st.error("No cached content created yet – run Step 1 first.")
            else:
                with st.spinner("Calling generateContent with cachedContent..."):
                    result = _call_with_cache(google_api_key, model_id, cached_name, question)
                if result["status_code"] != 200:
                    st.error(result["data"])
                else:
                    text, usage = _extract_text_and_usage(result["data"])
                    st.markdown("**Model answer:**")
                    st.write(text)
                    st.markdown("**Usage metadata:**")
                    st.json(usage)
                    st.caption(f"Latency: {result['data']['_latency_s']:.2f}s")

                    st.info(
                        "Look at fields like `promptTokenCount`, `cachedContentTokenCount`, and `totalTokenCount` "
                        "to see how many tokens came from the cached context.[web:159]"
                    )