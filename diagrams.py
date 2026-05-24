import io

import streamlit as st

try:
    from streamlit_mermaid import mermaid
    HAS_MERMAID = True
except Exception:
    HAS_MERMAID = False


def _mermaid_for_choices(model_label: str) -> str:
    choices = st.session_state.architecture_choices
    app_type = choices["app_type"]
    framework = choices["framework"]
    use_mcp = choices["use_mcp"]
    use_rag = choices["use_rag"]
    monitoring = choices["monitoring"]
    qa_tool = choices["qa_tool"]

    # Simple flowchart capturing high-level components.
    lines = [
        "flowchart LR",
        "  U[User] -->|messages| UI[Streamlit UI]",
        "  UI --> OR[OpenRouter]",
        f"  OR --> M[Gemma model: {model_label}]",
    ]

    if use_rag:
        lines.insert(2, "  UI --> RAG[RAG layer + Vector DB]")
        lines.insert(3, "  RAG --> OR")

    if use_mcp:
        lines.append("  M --> MCP[MCP / Tool layer]")
        lines.append("  MCP --> Ext[External systems]")

    if "Agentic" in app_type:
        lines.append("  UI --> CTRL[Agent controller loop]")
        lines.append("  CTRL --> OR")
        if use_mcp:
            lines.append("  CTRL --> MCP")

    if monitoring != "None":
        lines.append(f"  OR --> MON[Monitoring: {monitoring}]")

    if qa_tool != "None":
        lines.append(f"  M --> QA[QA/Eval: {qa_tool}]")

    lines.append("  OR --> UI")

    return "\n".join(lines)


def render_diagram_panel(model_label: str) -> None:
    st.markdown("### 🗺️ System Design Diagram (Mermaid)")

    mermaid_text = _mermaid_for_choices(model_label)

    if HAS_MERMAID:
        mermaid(mermaid_text, height=400)
    else:
        st.info(
            "To render Mermaid diagrams, add `streamlit-mermaid` to requirements.txt. "
            "For now we show the Mermaid source below."
        )
        st.code(mermaid_text, language="mermaid")

    # Downloadable .mmd file
    buf = io.BytesIO(mermaid_text.encode("utf-8"))
    st.download_button(
        label="Download Mermaid diagram (.mmd)",
        data=buf,
        file_name="architecture.mmd",
        mime="text/x-mermaid",
    )