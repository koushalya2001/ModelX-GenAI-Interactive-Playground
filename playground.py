import math
from typing import Optional, List, Dict, Any

import streamlit as st

from openrouter_client import OpenRouterClient


def _append_interaction(
    model_label: str,
    framework: str,
    usage: dict,
    cache: dict,
    app_type: str,
    context_meta: dict,
):
    entry = {
        "model_label": model_label,
        "framework": framework,
        "app_type": app_type,
        "usage": usage,
        "cache": cache,
        "context_strategy": context_meta.get("strategy"),
        "history_tokens_before": context_meta.get("tokens_before"),
        "history_tokens_after": context_meta.get("tokens_after"),
        "rag_used": context_meta.get("rag_used", False),
        "safety_flag": context_meta.get("safety_flag", False),
        "error_type": context_meta.get("error_type"),
        "user_rating": None,
    }
    st.session_state.interactions.append(entry)


def _approx_tokens_from_messages(messages: List[Dict[str, str]]) -> int:
    # Very rough heuristic: 0.75 * word_count
    text = " ".join(m["content"] for m in messages)
    words = text.split()
    return int(math.ceil(len(words) * 0.75))


def _summarize_history_with_gemma(
    client: Any,
    history: List[Dict[str, str]],
    keep_last_n: int,
) -> List[Dict[str, str]]:
    """
    Summarize older turns into a single assistant message using the same Gemma model,
    then keep the summary + last N turns.[web:35][web:41][web:43]
    """
    if len(history) <= keep_last_n:
        return history

    old_part = history[:-keep_last_n]
    recent_part = history[-keep_last_n:]

    summary_prompt = (
        "You are a concise assistant. Summarize the earlier conversation below into a short note "
        "that preserves important facts and decisions, but drops chit-chat.\n\n"
        "EARLIER TURNS:\n"
    )
    for msg in old_part:
        summary_prompt += f"- {msg['role']}: {msg['content']}\n"

    result = client.chat(
        messages=[{"role": "user", "content": summary_prompt}],
        temperature=0.0,
        max_tokens=256,
    )
    if not result.get("ok", False):
        # If summarization fails, fall back to sliding window.
        return history[-keep_last_n:]

    summary_text = result["content"]
    summary_msg = {
        "role": "assistant",
        "content": f"[Summary of earlier turns]\n\n{summary_text}",
    }
    return [summary_msg] + recent_part


def _apply_context_strategy(
    client: Any,
    base_history: List[Dict[str, str]],
    app_type: str,
    has_tool_output: bool,
) -> (List[Dict[str, str]], dict):
    cfg = st.session_state.context_strategy
    mode_config = cfg["mode"]
    last_n = cfg["last_n_turns"]

    tokens_before = _approx_tokens_from_messages(base_history)

    # Adaptive tweak: for agentic + tools, be more aggressive
    if app_type == "Agentic assistant" and has_tool_output:
        # If user said "keep all", treat it as summarize+last N for agentic mode
        if mode_config == "Keep all turns":
            mode_effective = "Summarize older turns"
        else:
            mode_effective = mode_config
        # Shrink last_n a bit so we keep fewer raw turns
        last_n_effective = max(4, last_n // 2)
    else:
        mode_effective = mode_config
        last_n_effective = last_n

    if mode_effective == "Keep all turns":
        trimmed_history = base_history
        strategy_used = "Keep all turns"
    elif mode_effective == "Last N turns":
        trimmed_history = base_history[-last_n_effective:]
        strategy_used = f"Last {last_n_effective} turns (adaptive={has_tool_output})"
    else:
        # Summarize older turns + keep last N (possibly shrunk)
        trimmed_history = _summarize_history_with_gemma(client, base_history, last_n_effective)
        strategy_used = f"Summarize + last {last_n_effective} turns (adaptive={has_tool_output})"

    tokens_after = _approx_tokens_from_messages(trimmed_history)

    meta = {
        "strategy": strategy_used,
        "tokens_before": tokens_before,
        "tokens_after": tokens_after,
        "rag_used": False,
        "safety_flag": False,
        "error_type": None,
    }
    return trimmed_history, meta


def _toy_tool_math(query: str) -> str:
    if any(op in query for op in ["+", "-", "*", "/"]):
        try:
            result = eval(query, {"__builtins__": {}})
            return f"[TOOL math] Result of `{query}` is {result}."
        except Exception:
            return "[TOOL math] Sorry, I could not evaluate that expression safely."
    return "[TOOL math] No arithmetic expression detected."


def render(client: Optional[OpenRouterClient], model_label: str) -> None:
    st.subheader("🎛️ Playground: Chatbot vs Agentic")
    has_tool_output = False
    if client is None:
        st.info("Provide an OpenRouter API key in the sidebar to run live model calls.")
        return

    choices = st.session_state.architecture_choices
    app_type = choices["app_type"]
    framework = choices["framework"]

    st.markdown(
        f"**Current architecture:** `{app_type}` with `{framework}`, "
        f"model **{model_label}**."
    )

    # Show chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    user_prompt = st.chat_input(
        "Ask something, or request an agentic task (e.g. 'plan a 3-step research workflow')."
    )
    if not user_prompt:
        return

    st.session_state.chat_history.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.write(user_prompt)

    messages = list(st.session_state.chat_history)

    # Agentic branch: inject toy tool output
    if app_type == "Agentic assistant":
        tool_choice = st.selectbox(
            "Toy tool to demonstrate tool-calling",
            ["None", "Math evaluator", "Pseudo web search"],
            index=1,
        )
        tool_msg = None
        if tool_choice == "Math evaluator":
            tool_msg = _toy_tool_math(user_prompt)
        elif tool_choice == "Pseudo web search":
            tool_msg = "[TOOL web] Pretend we fetched relevant web pages here and summarized them."

        if tool_msg:
            has_tool_output = True
            tool_assistant_turn = {
                "role": "assistant",
                "content": f"(Tool output injected into context)\n\n{tool_msg}",
            }
            messages.append(tool_assistant_turn)
            st.session_state.chat_history.append(tool_assistant_turn)

    # Apply context strategy before calling the model
    messages_to_send, ctx_meta = _apply_context_strategy(client, messages,app_type, has_tool_output)

    with st.chat_message("assistant"):
        with st.spinner("Querying OpenRouter / Gemma..."):
            max_tokens = 256 if (app_type == "Agentic assistant" and has_tool_output) else 512
            result = client.chat(
                            messages=messages_to_send,
                            max_tokens=max_tokens,
                                 )
        if not result.get("ok", False):
            st.error(result["content"])
            ctx_meta["error_type"] = "openrouter_error"
            _append_interaction(model_label, framework, {}, {}, app_type, ctx_meta)
            return

        content = result["content"]
        st.write(content)

    st.session_state.chat_history.append({"role": "assistant", "content": content})

    usage = result["usage"]
    cache = result["cache"]
    _append_interaction(model_label, framework, usage, cache, app_type, ctx_meta)

    with st.expander("Token & cache metrics for this turn"):
        col1, col2, col3 = st.columns(3)
        col1.metric("Prompt tokens", usage.get("prompt_tokens", 0))
        col2.metric("Completion tokens", usage.get("completion_tokens", 0))
        col3.metric("Cached prompt tokens", usage.get("cached_prompt_tokens", 0))

        col4, col5, col6 = st.columns(3)
        col4.metric("Total tokens", usage.get("total_tokens", 0))
        col5.metric("Latency (s)", f"{usage.get('latency_s', 0):.2f}")
        #col6.metric("Cache status", cache.get("status", "MISS"))

        st.caption(
            f"Context strategy: **{ctx_meta['strategy']}** – approx tokens before: "
            f"{ctx_meta['tokens_before']}, after: {ctx_meta['tokens_after']}."
        )

    # Optional: quick feedback
    with st.expander("How good was this answer?"):
        rating = st.radio(
            "Your rating",
            options=[None, "👍", "👎"],
            format_func=lambda x: "No rating" if x is None else x,
            horizontal=True,
            index=0,
        )
        if rating is not None:
            # Update rating on the last interaction
            if st.session_state.interactions:
                st.session_state.interactions[-1]["user_rating"] = rating
