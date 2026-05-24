import streamlit as st


def render() -> None:
    st.subheader("📐 Critical Design & Strategy Decisions")

    st.markdown("Use this page as a reading companion while you experiment in the playground.")

    with st.expander("Model choice (Gemma 2B vs 9B vs 27B)"):
        st.write(
            "- **2B**: fast, cheap, great for edge / prototype agents and teaching; weaker reasoning, good for showing "
            "context limits and KV pressure.\n"
            "- **9B**: balanced quality/latency; reasonable for small production chatbots.\n"
            "- **27B**: stronger reasoning and tool use but higher latency and memory footprint; illustrates how "
            "KV cache grows with context length and model width."
        )

    with st.expander("Seed/system prompt choices"):
        st.write(
            "System prompts define the agent's role and safety posture. Good patterns:\n"
            "- Separate **persona**, **task**, **constraints**, and **tool descriptions** into structured sections.\n"
            "- Keep the system prompt stable across turns so providers can exploit **prompt caching**.\n"
            "- For agentic apps, explicitly describe the tool-calling protocol and when to stop planning."
        )

    with st.expander("Multi-turn chat, tools & context strategy"):
        st.write(
            "Key decisions:\n"
            "- How many turns to keep in memory (full transcript vs sliding window vs summary).\n"
            "- When to inject **tool outputs** vs asking the user for confirmation.\n"
            "- Whether to rely on provider **KV cache** to cheaply reuse earlier tokens, or explicitly trim context when "
            "windows get large."
        )

    with st.expander("KV cache & context trimming"):
        st.write(
            "KV cache stores key/value tensors for each token to speed up autoregressive decoding. Larger context windows "
            "and more heads → bigger KV cache and memory traffic.[web:7][web:16][web:13]\n\n"
            "Design levers:\n"
            "- Cap max context length per request, even if the model supports more.\n"
            "- Use conversation summarization to keep recent turns verbatim and older turns compressed.\n"
            "- Consider tiered KV cache or offloading (GPU→CPU→disk) at serving time for heavy workloads, even though this "
            "Streamlit app runs purely as a client."
        )

    with st.expander("Choice of orchestrators (simple loop vs LangGraph/CrewAI/AutoGen)"):
        st.write(
            "- **Simple loop**: Your own Python loop deciding when to call tools; transparent and great for teaching.\n"
            "- **LangGraph/LangChain**: Graph-based workflows, retries, branches; good for complex agents.\n"
            "- **CrewAI/AutoGen**: Multi-agent conversations; powerful but can be harder to debug.\n\n"
            "This app deliberately starts with a simple loop plus toy tools so learners see the control flow before "
            "graduating to heavier frameworks."
        )