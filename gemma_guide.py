import streamlit as st


def render(model_label: str) -> None:
    st.subheader("📘 Gemma Small-Model Guide")

    st.markdown(
        f"Current selection: **{model_label}**. Use this guide to reason about when to prefer Gemma 2B vs larger variants."
    )

    st.markdown("### Why start with 2B-sized Gemma?")
    st.write(
        "- Small models are easier to run cheaply and are great for teaching KV/cache and context limits.\n"
        "- They still support reasonably long contexts in hosted APIs, but you will see quality drop when prompts get noisy or "
        "when agents chain too many steps together.[web:16][web:8]"
    )

    st.markdown("### Typical use-cases for Gemma 2B:")
    st.write(
        "- Lightweight chatbots with narrow domains.\n"
        "- Edge prototypes or mobile/embedded experiments.\n"
        "- Agent mechanics sandboxes like this app, where you care more about seeing failures and constraints than squeezing "
        "every bit of performance."
    )

    st.markdown("### When to upgrade to 9B / 27B:")
    st.write(
        "- You need multi-step reasoning and tool use reliability.\n"
        "- You are packaging RAG with large context windows and want more robust understanding.\n"
        "- Safety / refusal behaviour on complex injections becomes more important than raw latency."
    )

    st.markdown("### Agentic vs pure chatbot with Gemma")
    st.write(
        "- Chatbot mode: good for straightforward Q&A and explanations.\n"
        "- Agentic mode: lets Gemma orchestrate tools and RAG, but increases state management complexity – context packing, "
        "KV cache usage, and safety checks all become critical.\n"
        "This app lets learners flip between the two with the same model and see practical differences."
    )