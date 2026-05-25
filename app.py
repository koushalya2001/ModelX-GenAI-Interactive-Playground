import streamlit as st

from openrouter_client import OpenRouterClient
from google_ai_client import GoogleGemmaClient  # NEW
import playground
import design_decisions
import rag_lab
import safety_lab
import metrics_scorecard
import diagrams
import gemma_guide
import gemma_coach  # NEW
import infra_explainer
import ui_flow_explainer
import google_context_cache_demo
import grounding_playground
import how_to_use
import timeout_lab
# Model options per backend
OPENROUTER_MODEL_OPTIONS = {
    "Gemma 4 26B A4B (free)": "google/gemma-4-26b-a4b-it:free",
    "Gemma 4 31B (free)": "google/gemma-4-31b-it:free",
}

# Replace the IDs below with the exact Google AI Studio model IDs when you wire the client.
GOOGLE_MODEL_OPTIONS = {
    "Gemma 4 26B A4B": "gemma-4-26b-a4b-it",
    "Gemma 4 31B": "gemma-4-31b-it",
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
    if "backend_config" not in st.session_state:
        st.session_state.backend_config = {
            "OpenRouter": {"api_key": "", "active_label": None, "active_model": None},
            "Google AI Studio": {"api_key": "", "active_label": None, "active_model": None},
        }
    if "active_backend" not in st.session_state:
        st.session_state.active_backend = "OpenRouter"


def main():
    st.set_page_config(
        page_title="Gemma GenAI & Agentic Playground",
        layout="wide",
    )
    init_session_state()

    st.title("🧠 Gemma GenAI & Agentic Playground")
    st.caption(
        "Explore model choices, agent vs chatbot architectures, RAG designs, KV/cache behaviour, "
        "prompt injections, and evaluation – all on top of Gemma models."
    )

    with st.sidebar:
        st.header("🧠 Model backends & availability")

        # 1. Static index of models per backend
        with st.expander("OpenRouter models", expanded=True):
            st.write("**Free Gemma 4 models on OpenRouter:**")
            for label, mid in OPENROUTER_MODEL_OPTIONS.items():
                st.write(f"- {label} → `{mid}`")
            st.caption(
                "Free Gemma 4 26B A4B & 31B via OpenRouter (no card needed, subject to OpenRouter free limits)."
            )

        with st.expander("Google AI Studio models"):
            st.write("**Gemma 4 models on Google AI Studio (requires Google key):**")
            for label, mid in GOOGLE_MODEL_OPTIONS.items():
                st.write(f"- {label} → `{mid}`")
            st.caption(
                "Use Google AI Studio for Gemma 4. This app lets you switch between OpenRouter and Google AI Studio as backends."
            )

        st.markdown("---")

        # 2. Choose which backends to configure (can pick both)
        backends_selected = st.multiselect(
            "Backends to configure",
            options=["OpenRouter", "Google AI Studio"],
            default=["OpenRouter"],
            help="Configure one or both backends; then pick which one is active for live calls.",
        )

        st.header("🔐 Backend configuration")

        # OpenRouter backend
        if "OpenRouter" in backends_selected:
            st.subheader("OpenRouter")
            st.session_state.backend_config["OpenRouter"]["api_key"] = st.text_input(
                "OpenRouter API key",
                type="password",
                key="openrouter_api_key",
            )
            openrouter_label = st.selectbox(
                "OpenRouter Gemma model",
                list(OPENROUTER_MODEL_OPTIONS.keys()),
                key="openrouter_model_label",
            )
            st.session_state.backend_config["OpenRouter"]["active_label"] = openrouter_label
            st.session_state.backend_config["OpenRouter"]["active_model"] = OPENROUTER_MODEL_OPTIONS[
                openrouter_label
            ]

        # Google AI Studio backend (UI only for now)
        if "Google AI Studio" in backends_selected:
            st.subheader("Google AI Studio")
            st.session_state.backend_config["Google AI Studio"]["api_key"] = st.text_input(
                "Google AI Studio API key",
                type="password",
                key="google_api_key",
                help="From Google AI Studio; used for Gemma 4 E2B/E4B when wired.",
            )
            google_label = st.selectbox(
                "Google Gemma 4 model",
                list(GOOGLE_MODEL_OPTIONS.keys()),
                key="google_model_label",
            )
            st.session_state.backend_config["Google AI Studio"]["active_label"] = google_label
            st.session_state.backend_config["Google AI Studio"]["active_model"] = GOOGLE_MODEL_OPTIONS[
                google_label
            ]

        st.markdown("---")

        # 3. Single active backend for now (the one we actually call)
        st.session_state.active_backend = st.radio(
            "Active backend for this session",
            options=backends_selected or ["OpenRouter"],
            key="active_backend_radio",
            help="This backend will be used for all live LLM calls in the playground.",
        )

        active_backend = st.session_state.active_backend
        active_cfg = st.session_state.backend_config[active_backend]
        st.caption(
            f"Active backend: **{active_backend}**, model: "
            f"`{active_cfg.get('active_label') or 'not selected'}`"
        )

        st.markdown("---")
        st.header("💰 Cost assumptions (optional)")
        st.session_state.cost_per_1k_tokens = st.number_input(
            "Cost per 1K tokens (USD, estimated)",
            min_value=0.0,
            value=float(st.session_state.cost_per_1k_tokens),
            step=0.001,
            format="%.3f",
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
        if st.button("🚀 Start here: How to use this portal"):
            st.session_state.section_radio = "⭐ How to use this portal (start here)"
        page = st.radio(
            "Section",
            [   "⭐ How to use this portal (start here)", 
                "Playground (LLM & Agent)",                
                "Design decisions",
                "RAG & Vector DB lab",
                "Prompt injection & safety lab",
                "Metrics scorecard & system design",
                "Timeout & resilience lab",   # NEW
                "Grounding playground",
                "Gemma 2B guide",
                "Gemma coach & quiz",  # NEW
                "Infra & Serving 101",  # NEW
                "UI–Model–Agent flow",   # NEW
                "Google context caching demo",  # NEW
            ],
            key="section_radio",  # <-- add this

        )

    # Instantiate client for the active backend (exactly one client/model at a time)
    client = None
    model_label = None

    active_backend = st.session_state.active_backend
    cfg = st.session_state.backend_config[active_backend]

    if active_backend == "OpenRouter":
        api_key = cfg.get("api_key")
        model_id = cfg.get("active_model")
        model_label = cfg.get("active_label")
        if api_key and model_id:
            client = OpenRouterClient(
                api_key=api_key,
                model=model_id,
                enable_response_cache=enable_cache,
                app_title="Gemma GenAI & Agentic Playground",
            )
    elif active_backend == "Google AI Studio":
        api_key = cfg.get("api_key")
        model_id = cfg.get("active_model")
        model_label = cfg.get("active_label")
        if api_key and model_id:
            client = GoogleGemmaClient(
                api_key=api_key,
                model=model_id,
                timeout_s=200.0
            )
    st.markdown(
        f"**Active backend:** `{active_backend}` · "
        f"**Model:** `{model_label or 'not selected'}`"
    )
    # If no client could be created, pages will show an info message when they need it.
    # If no client could be created, pages will show an info message when they need it.

    # Route to pages
    
    if page == "Playground (LLM & Agent)":
        playground.render(client, model_label or "Unknown model")
    elif page == "⭐ How to use this portal (start here)":
        how_to_use.render()
    elif page == "Grounding playground":
        grounding_playground.render(client, model_label or "Unknown model")
    elif page == "Design decisions":
        design_decisions.render()
    elif page == "RAG & Vector DB lab":
        rag_lab.render()
    elif page == "Prompt injection & safety lab":
        safety_lab.render(client)
    elif page == "Metrics scorecard & system design":
        metrics_scorecard.render()
        diagrams.render_diagram_panel(model_label or "Unknown model")
    elif page == "Gemma 2B guide":
        gemma_guide.render(model_label or "Unknown model")
    elif page == "Gemma coach & quiz":
        gemma_coach.render(client, model_label or "Unknown model")
    elif page == "Infra & Serving 101":  # NEW
        infra_explainer.render(client, model_label or "Unknown model")
    elif page == "UI–Model–Agent flow":
        ui_flow_explainer.render(client, model_label or "Unknown model")
    elif page == "Google context caching demo":
    # Only meaningful when Google AI Studio is configured
        google_cfg = st.session_state.backend_config["Google AI Studio"]
        google_api_key = google_cfg.get("api_key")
        google_model_id = google_cfg.get("active_model")
        google_context_cache_demo.render(google_model_id, google_api_key)
    elif page == "Timeout & resilience lab":
        timeout_lab.render(client, model_label or "Unknown model")


if __name__ == "__main__":
    main()
