import streamlit as st

from openrouter_client import OpenRouterClient
import playground
import design_decisions
import rag_lab
import safety_lab
import metrics_scorecard
import diagrams
import gemma_guide
import gemma_coach  # NEW


MODEL_OPTIONS = {
    "Gemma 1 2B (small, cheap)": "google/gemma-2b-it",
    "Gemma 2 9B (general)": "google/gemma-2-9b-it",
    "Gemma 2 27B (strong reasoning)": "google/gemma-2-27b-it",
}


def init_session_state():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "interactions" not in st.session_state:
        st.session_state.interactions = []
    if "architecture_choices" not in st.session_state:
        st.session_state.architecture_choices = {
            "app_type": "Chatbot",
            "framework": "Simple loop",
            "use_mcp": False,
            "use_rag": False,
            "monitoring": "None",
            "qa_tool": "None",
        }
    if "context_strategy" not in st.session_state:
        st.session_state.context_strategy = {
            "mode": "Keep all turns",
            "last_n_turns": 8,
        }
    if "cost_per_1k_tokens" not in st.session_state:
        st.session_state.cost_per_1k_tokens = 0.0


def main():
    st.set_page_config(
        page_title="Gemma GenAI & Agentic Playground",
        layout="wide",
    )
    init_session_state()

    st.title("🧠 Gemma GenAI & Agentic Playground")
    st.caption(
        "Explore model choices, agent vs chatbot architectures, RAG designs, KV/cache behaviour, "
        "prompt injections, and evaluation – all on top of Gemma via OpenRouter."
    )

    with st.sidebar:
        st.header("🔐 OpenRouter Setup")
        api_key = st.text_input("OpenRouter API key", type="password")
        model_label = st.selectbox("Model", list(MODEL_OPTIONS.keys()))
        model_id = MODEL_OPTIONS[model_label]

        st.markdown("---")
        st.header("💰 Cost assumptions (optional)")
        st.session_state.cost_per_1k_tokens = st.number_input(
            "Cost per 1K tokens (USD, estimated)",
            min_value=0.0,
            value=0.0,
            step=0.001,
            help="Used only for estimated cost in the scorecard; set to 0 if you don't care.",
        )

        st.markdown("---")
        st.header("🧩 App Architecture")
        app_type = st.selectbox("Application type", ["Chatbot", "Agentic assistant"])
        framework = st.selectbox(
            "Orchestration flavour",
            ["Simple loop", "LangGraph/LangChain (conceptual)", "CrewAI/AutoGen (conceptual)"],
        )
        use_mcp = st.checkbox("Use MCP-style tools (conceptual)", value=True)
        use_rag = st.checkbox("Use RAG/vector DB (conceptual)", value=False)
        monitoring = st.selectbox(
            "Monitoring/LLMOps (conceptual)",
            ["None", "Langfuse", "Arize", "LangWatch"],
        )
        qa_tool = st.selectbox(
            "QA/Eval strategy (conceptual)",
            ["None", "Heuristic safety rules", "LLM-as-judge"],
        )

        st.session_state.architecture_choices = {
            "app_type": app_type,
            "framework": framework,
            "use_mcp": use_mcp,
            "use_rag": use_rag,
            "monitoring": monitoring,
            "qa_tool": qa_tool,
        }

        st.markdown("---")
        st.header("🧠 Context strategy")
        mode = st.selectbox(
            "How to manage chat history",
            ["Keep all turns", "Last N turns", "Summarize when long"],
        )
        last_n_turns = st.slider(
            "N for sliding window / summarization",
            min_value=4,
            max_value=20,
            value=8,
        )
        st.session_state.context_strategy = {
            "mode": mode,
            "last_n_turns": last_n_turns,
        }

        st.markdown("---")
        enable_cache = st.checkbox("Enable response caching demo", value=True)

        page = st.radio(
            "Section",
            [
                "Playground (LLM & Agent)",
                "Design decisions",
                "RAG & Vector DB lab",
                "Prompt injection & safety lab",
                "Metrics scorecard & system design",
                "Gemma 2B guide",
                "Gemma coach & quiz",  # NEW
            ],
        )

    client = None
    if api_key:
        client = OpenRouterClient(
            api_key=api_key,
            model=model_id,
            enable_response_cache=enable_cache,
            app_title="Gemma GenAI & Agentic Playground",
        )

    if page == "Playground (LLM & Agent)":
        playground.render(client, model_label)
    elif page == "Design decisions":
        design_decisions.render()
    elif page == "RAG & Vector DB lab":
        rag_lab.render()
    elif page == "Prompt injection & safety lab":
        safety_lab.render(client)
    elif page == "Metrics scorecard & system design":
        metrics_scorecard.render()
        diagrams.render_diagram_panel(model_label)
    elif page == "Gemma 2B guide":
        gemma_guide.render(model_label)
    elif page == "Gemma coach & quiz":
        gemma_coach.render(client, model_label)


if __name__ == "__main__":
    main()