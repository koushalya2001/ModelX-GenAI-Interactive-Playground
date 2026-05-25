import streamlit as st


def render() -> None:
    st.subheader("⭐ How to use this interactive learning portal")

    st.markdown("### 1. Start with model & backend selection")

    st.write(
        "- In the **left sidebar**, pick your backend (OpenRouter or Google AI Studio) and a Gemma model.\n"
        "- The header at the top of the main area shows the **active backend and model** for this session."
    )

    st.markdown("### 2. Recommended learning path")

    st.write(
        "1. **Grounding playground** – see how different grounding strategies affect answers.\n"
        "2. **Playground (LLM & Agent)** – contrast chatbot vs agentic patterns and watch tokens/latency.\n"
        "3. **Prompt injection & safety lab** – try attacks and see how safety heuristics work.\n"
        "4. **RAG & Vector DB lab** – reason about chunking, top‑k, and context growth.\n"
        "5. **Metrics scorecard & system design** – inspect your telemetry and architecture diagram.\n"
        "6. **Gemma guide + coach & quiz** – get model‑specific guidance and test your understanding."
    )

    st.markdown("### 3. Where to see technical details")

    st.write(
        "- **UI–Model–Agent flow** explains exactly how user input, session state, agent logic, and model calls "
        "are wired together.\n"
        "- **Infra & Serving 101** shows how the same patterns map to OpenRouter, Google AI Studio, and vLLM/local."
    )

    st.markdown("### 4. Tips for experimenting")

    st.write(
        "- In each playground, run the **same question** under different settings (grounding, context strategy, model) "
        "and compare answers.\n"
        "- Keep an eye on the **metrics scorecard** to understand cost, latency, and cache behaviour.\n"
        "- Use the **Gemma coach** to get feedback on your architecture decisions for your own use case."
    )
