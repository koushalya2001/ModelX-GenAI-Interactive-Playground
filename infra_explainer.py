import textwrap
from typing import Optional

import streamlit as st

from openrouter_client import OpenRouterClient
from google_ai_client import GoogleGemmaClient  # if present; else ignore type hints


def _openrouter_section():
    st.markdown("### OpenRouter – hosted routing layer")

    st.write(
        "OpenRouter is a hosted router that exposes an **OpenAI-compatible `/chat/completions` API** and forwards your "
        "request to many different upstream providers (Google AI Studio, Fireworks, etc.).[web:12][web:98] "
        "In your app, you already use it for Gemma 4 26B/31B free models."
    )

    code = textwrap.dedent(
        """
        import requests

        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": "google/gemma-4-26b-a4b-it:free",
            "messages": [
                {"role": "user", "content": "Explain KV caching in simple terms"},
            ],
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        print(resp.json())
        """
    ).strip()
    st.code(code, language="python")

    st.caption(
        "Your `OpenRouterClient` wraps exactly this pattern and adds latency measurement, token usage parsing, "
        "and cache header surfacing."
    )


def _google_ai_studio_section():
    st.markdown("### Google AI Studio – managed Gemma/Gemini API")

    st.write(
        "Google AI Studio exposes Gemma and Gemini models via the **Gemini API**, using the `generateContent` endpoint, "
        "with an API key. You do **not** download the model; Google hosts it for you.[web:100][web:104][web:97] "
        "Smaller Gemma 4 E2B/E4B are currently positioned as edge/local models, not as fully hosted AI Studio endpoints."
    )

    code = textwrap.dedent(
        """
        import requests

        API_KEY = "YOUR_GOOGLE_AI_STUDIO_KEY"
        MODEL_ID = "gemma-4-26b-a4b-it"  # example; use exact ID from AI Studio 'Get code'

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_ID}:generateContent"
        params = {"key": API_KEY}
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": "Explain RAG in one paragraph"}],
                }
            ],
            "generationConfig": {"temperature": 0.3},
        }

        resp = requests.post(url, params=params, json=payload, timeout=60)
        data = resp.json()
        print(data)
        """
    ).strip()
    st.code(code, language="python")

    st.caption(
        "Your `GoogleGemmaClient` implements this pattern and normalizes the response into the same format as "
        "`OpenRouterClient.chat(...)`."
    )


def _vllm_section():
    st.markdown("### vLLM / local server – serve Gemma yourself")

    st.write(
        "vLLM lets you host Gemma weights yourself (on a GPU box) and exposes an **OpenAI-style HTTP API**.[web:105][web:110] "
        "This is how you would normally run Gemma 4 E2B/E4B or other small variants in production or on your own infra, "
        "not on Streamlit Community Cloud."
    )

    st.markdown("**Example: run Gemma 4 E4B locally with vLLM**")

    code_serve = textwrap.dedent(
        """
        # Terminal command (on your own GPU machine, not Streamlit Community Cloud)
        vllm serve google/gemma-4-E4B-it \\
            --max-model-len 131072 \\
            --host 0.0.0.0 --port 8000
        """
    ).strip()
    st.code(code_serve, language="bash")

    st.markdown("**Example: call the local vLLM server from Python**")

    code_client = textwrap.dedent(
        """
        import requests

        url = "http://localhost:8000/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": "google/gemma-4-E4B-it",  # served by vLLM
            "messages": [
                {"role": "user", "content": "Summarize this conversation in 3 bullet points."}
            ],
        }

        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        print(resp.json())
        """
    ).strip()
    st.code(code_client, language="python")

    st.caption(
        "This pattern is conceptually the same as OpenRouter, except you point to your own host instead of "
        "https://openrouter.ai."
    )

    st.info(
        "On Streamlit Community Cloud you usually cannot host heavy vLLM servers or download large Gemma weights; "
        "treat vLLM as a pattern for your own infra or a local GPU box."
    )


def _cli_playground(client: Optional[object], model_label: str) -> None:
    st.markdown("### 🖥️ Mini CLI playground (using the active backend)")

    st.write(
        "This simulates a minimal CLI REPL: type a prompt, hit Enter, and see the model response. "
        "Under the hood it uses whichever client is active in the main sidebar (OpenRouter or Google AI Studio)."
    )

    if client is None:
        st.info(
            "No active LLM client. Configure a backend and API key in the sidebar first, then come back "
            "to this page."
        )
        return

    if "cli_history" not in st.session_state:
        st.session_state.cli_history = []

    # Render history
    for i, (role, text) in enumerate(st.session_state.cli_history, start=1):
        prefix = ">>" if role == "user" else model_label
        st.markdown(f"**{prefix}:** {text}")

    user_cmd = st.text_input(
        "CLI input",
        placeholder="ask: explain KV cache vs prompt caching",
        key="cli_input",
    )
    if not user_cmd:
        return

    # Append user
    st.session_state.cli_history.append(("user", user_cmd))

    # Build a simple one-turn chat
    messages = [{"role": "user", "content": user_cmd}]
    with st.spinner("Calling the active backend..."):
        result = client.chat(messages=messages, temperature=0.3, max_tokens=512)
    if not result.get("ok", False):
        st.error(result["content"])
        st.session_state.cli_history.append(("assistant", f"[error] {result['content']}"))
        return

    answer = result["content"]
    st.session_state.cli_history.append(("assistant", answer))

    st.experimental_rerun()


def render(client: Optional[object], model_label: str) -> None:
    """
    Main entrypoint for this page.
    Pass in the currently active client (OpenRouterClient or GoogleGemmaClient)
    and the human-readable model label from app.py.
    """
    st.subheader("🔧 How Gemma models are served")

    tabs = st.tabs(
        [
            "OpenRouter",
            "Google AI Studio",
            "vLLM / local server",
            "CLI playground",
        ]
    )

    with tabs[0]:
        _openrouter_section()
    with tabs[1]:
        _google_ai_studio_section()
    with tabs[2]:
        _vllm_section()
    with tabs[3]:
        _cli_playground(client, model_label)
