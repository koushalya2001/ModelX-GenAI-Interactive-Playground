from typing import Optional, List, Dict, Any

import streamlit as st

from openrouter_client import OpenRouterClient


COACH_SYSTEM_PROMPT = """
You are a Gemma-based GenAI architecture coach.

You are embedded inside an educational Streamlit app that lets beginners experiment with:
- Gemma models via OpenRouter (2B, 9B, 27B)
- Simple chatbot vs agentic assistants with toy tools
- RAG using a small FAISS vector index
- Prompt injection and safety heuristics
- A metrics scorecard with tokens, latency, cache hits, safety flags, and cost estimates

Your goals:
1. Explain tradeoffs in model choice, context strategy, RAG, safety, MCP integration, and monitoring in simple terms.
2. Give concrete, actionable advice tailored to the user's described use case and the app's design.
3. When evaluating designs, balance cost, latency, quality, safety, and maintainability.
4. Be honest when something is unclear and suggest experiments the user can run in the app to learn more.
""".strip()


def _coach_messages() -> List[Dict[str, str]]:
    if "coach_history" not in st.session_state:
        st.session_state.coach_history = []
    return st.session_state.coach_history


def _render_coach_chat(client: OpenRouterClient, model_label: str) -> None:
    st.markdown("### 1️⃣ Ask Gemma about this app or your design")

    st.write(
        "Use this chat to ask anything about model/RAG/safety/orchestration choices while you build. "
        "The same Gemma model you selected in the sidebar is used for all answers."
    )

    history = _coach_messages()
    for msg in history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    user_q = st.chat_input("Ask Gemma a design question (e.g. 'Is my context strategy reasonable?').")
    if not user_q:
        return

    history.append({"role": "user", "content": user_q})
    with st.chat_message("user"):
        st.write(user_q)

    # Build messages: stable system prompt + prior turns
    messages: List[Dict[str, str]] = [{"role": "system", "content": COACH_SYSTEM_PROMPT}] + history

    with st.chat_message("assistant"):
        with st.spinner(f"Gemma ({model_label}) is thinking..."):
            result = client.chat(messages=messages, temperature=0.2)
        if not result.get("ok", False):
            st.error(result["content"])
            return
        answer = result["content"]
        st.write(answer)

    history.append({"role": "assistant", "content": answer})
    st.session_state.coach_history = history


def _summarize_metrics_for_eval() -> str:
    interactions: List[Dict[str, Any]] = st.session_state.interactions
    if not interactions:
        return "No interactions recorded yet."

    total_tokens = sum(i.get("usage", {}).get("total_tokens", 0) for i in interactions)
    avg_tokens = total_tokens / len(interactions)
    avg_latency = sum(i.get("usage", {}).get("latency_s", 0.0) for i in interactions) / len(interactions)
    safety_flags = sum(1 for i in interactions if i.get("safety_flag"))
    cache_hits = sum(1 for i in interactions if i.get("cache", {}).get("status", "").upper() == "HIT")

    return (
        f"- Total turns: {len(interactions)}\n"
        f"- Avg tokens/turn: {avg_tokens:.1f}\n"
        f"- Avg latency: {avg_latency:.2f}s\n"
        f"- Cache hits: {cache_hits}\n"
        f"- Safety flags: {safety_flags}"
    )


SCENARIOS = {
    "E-commerce FAQ assistant":
        "You are building a customer support assistant for an e-commerce site with 1M FAQ entries and order data.",
    "Internal dev helper":
        "You are building an internal code assistant for a small dev team that interacts with Git and CI logs.",
}


def _render_usecase_evaluator(client: OpenRouterClient, model_label: str) -> None:
    st.markdown("### 2️⃣ Use-case evaluator or quiz")

    mode = st.radio(
        "Do you have a real use case?",
        ["Yes, evaluate my use case", "No, give me a scenario & quiz"],
        horizontal=True,
    )

    choices = st.session_state.architecture_choices
    metrics_summary = _summarize_metrics_for_eval()

    if mode == "Yes, evaluate my use case":
        usecase = st.text_area(
            "Describe your real use case",
            placeholder="Example: A chatbot for triaging support tickets for a fintech startup...",
            height=150,
        )
        if st.button("Ask Gemma to evaluate my design"):
            if not usecase.strip():
                st.warning("Please describe your use case first.")
                return

            app_desc = (
                f"Current app settings in the playground:\n"
                f"- App type: {choices['app_type']}\n"
                f"- Framework: {choices['framework']}\n"
                f"- RAG enabled: {choices['use_rag']}\n"
                f"- MCP enabled: {choices['use_mcp']}\n"
                f"- Monitoring: {choices['monitoring']}\n"
                f"- QA/Eval: {choices['qa_tool']}\n\n"
                f"Observed metrics:\n{metrics_summary}\n"
            )

            eval_prompt = (
                "A learner has described their use case and current architecture/design choices.\n\n"
                f"USE CASE:\n{usecase}\n\n"
                f"APP CONFIG + METRICS:\n{app_desc}\n\n"
                "As a Gemma-based architecture coach, do the following:\n"
                "1. Briefly restate the use case in your own words.\n"
                "2. Evaluate whether the chosen model size and settings seem appropriate.\n"
                "3. Highlight 2–3 strong decisions the learner made.\n"
                "4. Highlight 2–3 risky or missing decisions (e.g., RAG, safety, context strategy, MCP, monitoring).\n"
                "5. Suggest concrete adjustments or experiments the learner can run in the app."
            )

            messages = [
                {"role": "system", "content": COACH_SYSTEM_PROMPT},
                {"role": "user", "content": eval_prompt},
            ]
            with st.spinner(f"Gemma ({model_label}) is evaluating your design..."):
                result = client.chat(messages=messages, temperature=0.2)
            if not result.get("ok", False):
                st.error(result["content"])
                return
            st.markdown("**Gemma's evaluation:**")
            st.write(result["content"])

    else:
        scenario_label = st.selectbox("Pick a scenario", list(SCENARIOS.keys()))
        scenario_text = SCENARIOS[scenario_label]
        st.markdown(f"**Scenario:** {scenario_text}")

        st.markdown("Now, pick your high-level design decisions for this scenario.")
        model_choice = st.selectbox(
            "Which Gemma size would you pick?",
            ["2B", "9B", "27B"],
        )
        use_rag = st.checkbox("Use RAG/vector DB", value=True)
        use_mcp = st.checkbox("Use MCP tools (e.g., Git, DB)", value=False)
        focus_cost = st.checkbox("Cost is a hard constraint", value=True)

        if st.button("Submit my design to Gemma for grading"):
            quiz_prompt = (
                "You are grading a beginner's architecture choices for a given scenario.\n\n"
                f"SCENARIO:\n{scenario_text}\n\n"
                "LEARNER'S DESIGN:\n"
                f"- Gemma size: {model_choice}\n"
                f"- RAG enabled: {use_rag}\n"
                f"- MCP enabled: {use_mcp}\n"
                f"- Cost is a hard constraint: {focus_cost}\n\n"
                "Give feedback in this structure:\n"
                "1. Grade from A to D.\n"
                "2. One paragraph: what they did well.\n"
                "3. One paragraph: what you would change (model size, RAG, safety, monitoring, MCP, context strategy).\n"
                "4. Two follow-up questions for the learner to think about."
            )
            messages = [
                {"role": "system", "content": COACH_SYSTEM_PROMPT},
                {"role": "user", "content": quiz_prompt},
            ]
            with st.spinner(f"Gemma ({model_label}) is grading your choices..."):
                result = client.chat(messages=messages, temperature=0.3)
            if not result.get("ok", False):
                st.error(result["content"])
                return
            st.markdown("**Gemma's grade & feedback:**")
            st.write(result["content"])


def render(client: Optional[OpenRouterClient], model_label: str) -> None:
    st.subheader("🧑‍🏫 Gemma Coach & Quiz")

    if client is None:
        st.info("Provide an OpenRouter API key in the sidebar to use Gemma as a coach.")
        return

    _render_coach_chat(client, model_label)
    st.markdown("---")
    _render_usecase_evaluator(client, model_label)