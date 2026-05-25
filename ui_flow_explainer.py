import textwrap
from typing import Optional

import streamlit as st


def render(client: Optional[object], model_label: str) -> None:
    st.subheader("🧩 How the UI, model, and agent interact")

    st.markdown("### 1. High-level request–response flow")

    st.write(
        "Every time you send a message from the UI, the app follows this pipeline:"
    )

    st.markdown(
        """
        1. **User input**  
           - You type in the chat box (Playground), CLI field (Infra tab), or coach chat.  
           - Streamlit captures this via `st.chat_input` or `st.text_input`.

        2. **Update in-memory conversation state**  
           - The code appends your message to `st.session_state.chat_history` (or a page-specific history like "
           "`coach_history` / `infra_cli_history`).  
           - This is your *current transcript* for that interaction.

        3. **Agent logic builds the prompt**  
           - For simple chatbot mode, the app turns the history into a `messages` list of `{role, content}`.  
           - In agentic mode, toy tools may run first (e.g. math evaluator), and their outputs are added as extra "
           "assistant turns.  
           - The context strategy (`Keep all`, `Last N`, `Summarize when long`) is applied to trim or summarize "
           "older turns before the call.

        4. **Backend client call**  
           - The page calls `client.chat(messages=...)`, where `client` is either:  
             - `OpenRouterClient` (for `google/gemma-4-26b-a4b-it:free`, `google/gemma-4-31b-it:free`), or  
             - `GoogleGemmaClient` (when you use Google AI Studio).  
           - The client sends an HTTP request to the provider, waits for a response, and returns:  
             `content`, `usage` (tokens, latency), `cache` info, and `ok` flag.

        5. **UI + metrics update**  
           - The model's text is appended to the same history list so it shows up in the conversation.  
           - A compact interaction record is pushed into `st.session_state.interactions` with: tokens, latency, "
           "cache status, context strategy, safety flags, and (optionally) your rating.  
           - The metrics scorecard and diagrams read from this shared telemetry.
        """
    )

    st.info(
        "Because every page reads and writes `st.session_state`, the same active backend + model, context "
        "strategy, and metrics are shared across the playground, safety lab, coach, and infra CLI."
    )

    st.markdown("### 2. How the agent decides what to send")

    st.write(
        "Inside the Playground, the agent logic is simple by design so you can see the wiring clearly:"
    )

    code = textwrap.dedent(
        """
        # 1) Append user message to history
        st.session_state.chat_history.append({"role": "user", "content": user_prompt})

        # 2) Optionally inject tool output (agentic mode)
        messages = list(st.session_state.chat_history)
        if app_type == "Agentic assistant":
            tool_msg = run_toy_tool(user_prompt)  # math / pseudo search
            if tool_msg:
                tool_turn = {"role": "assistant", "content": "(Tool output)\\n\\n" + tool_msg}
                messages.append(tool_turn)
                st.session_state.chat_history.append(tool_turn)

        # 3) Apply context strategy (keep all / last N / summarize)
        messages_to_send, ctx_meta = apply_context_strategy(client, messages)

        # 4) Call the active backend client
        result = client.chat(messages=messages_to_send)

        # 5) Append model reply + log metrics
        st.session_state.chat_history.append({"role": "assistant", "content": result["content"]})
        st.session_state.interactions.append(
            {
                "model_label": model_label,
                "usage": result["usage"],
                "cache": result["cache"],
                "context_strategy": ctx_meta["strategy"],
                # ... other telemetry fields ...
            }
        )
        """
    ).strip()
    st.code(code, language="python")

    st.caption(
        "This pattern is reused with minor variations in the coach, safety lab, and infra CLI playground. "
        "The UI just orchestrates session state and delegates model calls to the active client."
    )

    st.markdown("### 3. Where to look in the code")

    st.write(
        "- **`playground.py`** – main chat/agent loop and context strategy.  \n"
        "- **`safety_lab.py`** – runs fixed prompts through the same client and marks `safety_flag` on the "
        "latest interaction.  \n"
        "- **`gemma_coach.py`** – builds a different `messages` list (with a coach system prompt) but still calls "
        "`client.chat` and writes to its own `coach_history`.  \n"
        "- **`metrics_scorecard.py`** – reads `st.session_state.interactions` and never talks to the model directly."
    )