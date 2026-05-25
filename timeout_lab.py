import time
from typing import Optional, Dict, Any, List

import streamlit as st
from requests.exceptions import ReadTimeout


def _run_with_timeout(
    client: object,
    messages: List[Dict[str, str]],
    timeout_s: float,
) -> Dict[str, Any]:
    """
    Wrap client.chat(...) and override its timeout if the client supports it.
    For GoogleGemmaClient we assume a `timeout_s` attribute; otherwise we just call chat().
    """
    # Try to set a temporary timeout on the client if the attribute exists
    old_timeout = getattr(client, "timeout_s", None)
    if old_timeout is not None:
        client.timeout_s = timeout_s

    t0 = time.perf_counter()
    try:
        result = client.chat(messages=messages, temperature=0.3, max_tokens=256)
        ok = result.get("ok", True)
    except ReadTimeout:
        t1 = time.perf_counter()
        latency_s = t1 - t0
        result = {
            "ok": False,
            "content": f"ReadTimeout after {timeout_s:.0f} seconds – upstream model or network too slow.",
            "usage": {},
            "cache": {},
            "latency_s": latency_s,
        }
    finally:
        # Restore old timeout if applicable
        if old_timeout is not None:
            client.timeout_s = old_timeout

    # Make sure latency is populated
    if "latency_s" not in result:
        result["latency_s"] = time.perf_counter() - t0

    return result


def render(client: Optional[object], model_label: str) -> None:
    st.subheader("⏱️ API Timeout & Resilience Lab")

    st.write(
        "Use this lab to explore how timeouts affect GenAI apps and what you can do about them.\n\n"
        "When prompts get long, tools add extra context, or the provider is under load, your calls can take a long time. "
        "If they exceed your timeout, you get a `ReadTimeout` instead of a response."
    )

    if client is None:
        st.info("Configure a backend and API key in the sidebar, then return to this lab.")
        return

    col_cfg, col_info = st.columns([2, 1])

    with col_cfg:
        st.markdown("### 1. Configure the experiment")

        user_prompt = st.text_area(
            "Prompt to send to the model",
            height=120,
            value=(
                "Explain, in detail, how context length, grounding, and tool calls affect latency and timeouts "
                "in a Gemma 4 application."
            ),
        )

        timeout_s = st.slider(
            "Client read timeout (seconds)",
            min_value=10,
            max_value=120,
            value=60,
            step=10,
        )

        simulate_long_prompt = st.checkbox(
            "Simulate long prompt by repeating the question 5x",
            value=False,
        )

        retries = st.slider(
            "Retries on timeout (simple retry)",
            min_value=0,
            max_value=3,
            value=1,
        )

        st.caption(
            "In real apps you might use exponential backoff and a circuit breaker instead of simple retries."
        )

    with col_info:
        st.markdown("### 2. What to watch for")
        st.write(
            "- How often the call times out as you **increase prompt size**.\n"
            "- How much retries help or hurt UX.\n"
            "- How latency changes when you trim context or reduce max tokens."
        )

    if st.button("Run timeout experiment"):
        # Build messages
        content = user_prompt
        if simulate_long_prompt:
            content = (user_prompt + "\n\n") * 5

        messages = [{"role": "user", "content": content}]

        attempts = 0
        final_result: Dict[str, Any] = {}
        while attempts <= retries:
            attempts += 1
            with st.spinner(f"Attempt {attempts} / {retries + 1}..."):
                result = _run_with_timeout(client, messages, timeout_s)
            if result.get("ok", False):
                final_result = result
                break
            else:
                st.warning(f"Attempt {attempts} failed: {result['content']}")
                final_result = result

        st.markdown("### 3. Result")

        if final_result.get("ok", False):
            st.success(
                f"Model replied successfully in {final_result.get('latency_s', 0):.2f}s "
                f"after {attempts} attempt(s)."
            )
            st.markdown(f"**{model_label}:**")
            st.write(final_result["content"])
            st.markdown("**Usage (if available):**")
            st.json(final_result.get("usage", {}))
        else:
            st.error(
                f"All {attempts} attempt(s) failed. Last error: {final_result.get('content', 'unknown error')}.\n\n"
                "This is a realistic failure mode in GenAI apps when prompts and models are large or the provider "
                "is under heavy load."
            )

    st.markdown("### 4. Design takeaways")

    st.write(
        "- Timeouts are **normal** in real GenAI apps – models are heavy, prompts are large, and providers can be busy.\n"
        "- You should always:\n"
        "  - Set a **reasonable timeout** per request.\n"
        "  - Implement **retries with backoff** and a clear **user-facing error**.\n"
        "  - Trim context and cap `max_tokens` when you don't need giant answers.\n"
        "  - Log timeouts in metrics so you can see how often they occur and under what conditions."
    )