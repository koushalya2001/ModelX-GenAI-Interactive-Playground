import statistics
from collections import Counter
from typing import List, Dict, Any

import streamlit as st


def _safe_mean(values: List[float]) -> float:
    values = [v for v in values if v is not None]
    return statistics.mean(values) if values else 0.0


def _summarize(interactions: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not interactions:
        return {}

    latencies = [
        i.get("usage", {}).get("latency_s", 0.0)
        for i in interactions
        if i.get("usage")
    ]
    total_tokens = [
        i.get("usage", {}).get("total_tokens", 0)
        for i in interactions
        if i.get("usage")
    ]

    # Edge cache HIT rate (OpenRouter only: status HIT/MISS)
    edge_interactions = [
        i for i in interactions
        if i.get("cache", {}).get("status", "").upper() in ("HIT", "MISS")
    ]
    edge_hits = sum(
        1
        for i in edge_interactions
        if i.get("cache", {}).get("status", "").upper() == "HIT"
    )
    edge_hit_rate = (edge_hits / len(edge_interactions)) if edge_interactions else None

    # Context cache token rate (Google AI Studio: cachedContentTokenCount / promptTokenCount)
    context_cache_rates: List[float] = []
    for i in interactions:
        usage = i.get("usage", {}) or {}
        pt = usage.get("prompt_tokens", 0)
        ct = usage.get("cached_prompt_tokens", 0)
        if pt > 0 and ct > 0:
            context_cache_rates.append(ct / pt)
    avg_context_cache_rate = (
        _safe_mean(context_cache_rates) if context_cache_rates else None
    )

    safety_flags = [1 for i in interactions if i.get("safety_flag")]
    context_strategies = [i.get("context_strategy", "Unknown") for i in interactions]

    return {
        "count": len(interactions),
        "avg_latency": _safe_mean(latencies),
        "avg_tokens": _safe_mean(total_tokens),
        "edge_hit_rate": edge_hit_rate,
        "context_cache_rate": avg_context_cache_rate,
        "safety_flag_rate": (sum(safety_flags) / len(interactions)) if interactions else 0.0,
        "context_strategy_counts": Counter(context_strategies),
    }


def render() -> None:
    st.subheader("📊 Metrics Scorecard")

    interactions = st.session_state.interactions
    if not interactions:
        st.info("No interactions yet. Run some prompts in the playground first.")
        return

    summary = _summarize(interactions)
    cost_per_1k = st.session_state.get("cost_per_1k_tokens", 0.0)
    avg_tokens = summary["avg_tokens"]
    est_cost_per_turn = (avg_tokens / 1000.0) * cost_per_1k if cost_per_1k > 0 else 0.0

    # Top summary row
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total turns", summary["count"])
    col2.metric("Avg latency (s)", f"{summary['avg_latency']:.2f}")
    col3.metric("Avg tokens / turn", f"{summary['avg_tokens']:.0f}")

    if summary["edge_hit_rate"] is not None:
        col4.metric("Edge cache HIT rate", f"{summary['edge_hit_rate']*100:.0f}%")
    else:
        col4.metric("Edge cache HIT rate", "N/A")

    if summary["context_cache_rate"] is not None:
        col5.metric(
            "Avg context cache token rate",
            f"{summary['context_cache_rate']*100:.0f}%",
            help="For Google AI Studio: cachedContentTokenCount / promptTokenCount",
        )
    else:
        col5.metric("Avg context cache token rate", "N/A")

    # Second row: safety + cost
    col6, col7, col8, _ = st.columns(4)
    col6.metric("Safety flag rate", f"{summary['safety_flag_rate']*100:.0f}%")
    col7.metric("Est. cost / turn (USD)", f"{est_cost_per_turn:.4f}")
    total_tokens_all = sum(
        i.get("usage", {}).get("total_tokens", 0) for i in interactions
    )
    total_cost = (total_tokens_all / 1000.0) * cost_per_1k if cost_per_1k > 0 else 0.0
    col8.metric("Est. total cost (USD)", f"{total_cost:.4f}")

    st.markdown("---")
    st.markdown("### Context strategies in use")
    cs_counts = summary["context_strategy_counts"]
    for strategy, count in cs_counts.items():
        st.write(f"- **{strategy}**: {count} turns")

    st.markdown("---")
    st.markdown("### Recent interactions (last 10 turns)")

    # Show the last 10 interactions, most recent first
    recent = list(reversed(interactions[-10:]))
    for idx, it in enumerate(recent, start=1):
        usage = it.get("usage", {}) or {}
        cache = it.get("cache", {}) or {}

        total = usage.get("total_tokens", 0)
        prompt = usage.get("prompt_tokens", 0)
        completion = usage.get("completion_tokens", 0)
        latency = usage.get("latency_s", 0.0)
        cached = usage.get("cached_prompt_tokens", 0)

        cache_status = cache.get("status", "N/A")
        if cache_status.upper() in ("HIT", "MISS"):
            cache_str = f"Edge cache: {cache_status}"
        elif prompt > 0 and cached > 0:
            rate = cached / prompt
            cache_str = f"Context cache: {cached}/{prompt} ({rate*100:.0f}%)"
        else:
            cache_str = "No cache info"

        st.markdown(
            f"**Turn #{len(interactions) - (idx - 1)}** – "
            f"Model: `{it.get('model_label')}`, App: `{it.get('app_type')}`, "
            f"Framework: `{it.get('framework')}`"
        )

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Tokens (total)", total)
        c2.metric("Prompt", prompt)
        c3.metric("Completion", completion)
        c4.metric("Latency (s)", f"{latency:.2f}")
        c5.metric("Cache", cache_str)

        st.caption(
            f"Context strategy: **{it.get('context_strategy') or 'n/a'}** · "
            f"history tokens {it.get('history_tokens_before')}/{it.get('history_tokens_after')} · "
            f"RAG: {it.get('rag_used')} · Safety flag: {it.get('safety_flag')} · "
            f"Rating: {it.get('user_rating') or 'none'}"
        )
        st.markdown("---")
