import statistics
from collections import Counter
from typing import List, Dict, Any

import streamlit as st


def _summarize(interactions: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not interactions:
        return {}

    latencies = [i.get("usage", {}).get("latency_s", 0.0) for i in interactions]
    total_tokens = [i.get("usage", {}).get("total_tokens", 0) for i in interactions]
    cache_hits = [
        1 for i in interactions if (i.get("cache", {}).get("status", "").upper() == "HIT")
    ]
    safety_flags = [1 for i in interactions if i.get("safety_flag")]
    context_strategies = [i.get("context_strategy", "Unknown") for i in interactions]

    return {
        "count": len(interactions),
        "avg_latency": statistics.mean(latencies),
        "avg_tokens": statistics.mean(total_tokens),
        "cache_hit_rate": (sum(cache_hits) / len(interactions)) if interactions else 0.0,
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

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total turns", summary["count"])
    col2.metric("Avg latency (s)", f"{summary['avg_latency']:.2f}")
    col3.metric("Avg tokens / turn", f"{summary['avg_tokens']:.0f}")
    col4.metric("Edge cache HIT rate", f"{summary['cache_hit_rate']*100:.0f}%")

    col5, col6, col7, _ = st.columns(4)
    col5.metric("Safety flag rate", f"{summary['safety_flag_rate']*100:.0f}%")
    col6.metric("Est. cost / turn (USD)", f"{est_cost_per_turn:.4f}")
    total_tokens_all = sum(i.get("usage", {}).get("total_tokens", 0) for i in interactions)
    total_cost = (total_tokens_all / 1000.0) * cost_per_1k if cost_per_1k > 0 else 0.0
    col7.metric("Est. total cost (USD)", f"{total_cost:.4f}")

    st.markdown("---")
    st.markdown("### Context strategies in use")
    cs_counts = summary["context_strategy_counts"]
    for strategy, count in cs_counts.items():
        st.write(f"- **{strategy}**: {count} turns")

    st.markdown("---")
    st.markdown("### Recent interactions")

    for i, it in enumerate(reversed(interactions[-10:]), start=1):
        usage = it.get("usage", {})
        cache = it.get("cache", {})
        st.markdown(
            f"**#{i}** – Model: `{it.get('model_label')}`, App: `{it.get('app_type')}`, "
            f"Framework: `{it.get('framework')}`  \n"
            f"Tokens: {usage.get('total_tokens', 0)}, latency: {usage.get('latency_s', 0):.2f}s, "
            f"context strategy: {it.get('context_strategy')}, safety: {it.get('safety_flag')}, "
            f"user rating: {it.get('user_rating') or 'none'}"
        )
