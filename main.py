"""Streamlit interface for CUDA Documentation Copilot."""

from __future__ import annotations

import json
from datetime import datetime

import streamlit as st

from components.s5_rag_llm import RAGModel
from components.settings import load_settings
from components.ui_helpers import (
    SUGGESTED_QUESTIONS,
    citation_cards,
    response_metrics,
    safe_error_message,
)


@st.cache_resource(show_spinner=False)
def get_rag_model() -> RAGModel:
    return RAGModel()


def _render_sources(result: dict) -> None:
    cards = citation_cards(result)
    if not cards:
        st.info("No supporting source was retrieved for this question.")
        return

    st.subheader("Sources")
    for card in cards:
        with st.container(border=True):
            st.text(card["title"])
            if card["section"]:
                st.text(card["section"])
            st.text(card["excerpt"])
            st.caption(f"Retrieval score: {card['score']}")
            st.link_button("Open NVIDIA source", card["url"])


def _render_metrics(result: dict) -> None:
    metrics = response_metrics(result)
    for column, (label, value) in zip(st.columns(len(metrics)), metrics.items()):
        column.metric(label, value)


def main() -> None:
    st.set_page_config(page_title="CUDA Documentation Copilot", page_icon="⚡")
    settings = load_settings(require_gemini=False)

    st.title("CUDA Documentation Copilot")
    st.caption(
        f"Mode: **{settings.rag_mode}** · "
        + (
            f"Generation: **{settings.gemini_model}**"
            if settings.gemini_api_key
            else "Generation: **offline extractive fallback**"
        )
    )
    st.write("Ask a CUDA question and inspect the sources behind the answer.")

    st.write("**Try an example:**")
    for column, question in zip(st.columns(len(SUGGESTED_QUESTIONS)), SUGGESTED_QUESTIONS):
        if column.button(question, use_container_width=True):
            st.session_state["query_input"] = question

    query = st.text_area(
        "Question",
        key="query_input",
        height=100,
        max_chars=settings.query_max_chars,
        placeholder="How do CUDA streams allow work to overlap?",
    )

    if not st.button("Generate answer", type="primary"):
        return
    if not query.strip():
        st.warning("Please enter a CUDA question.")
        return

    with st.spinner("Retrieving cited CUDA sources and preparing an answer…"):
        try:
            result = get_rag_model().process_query(query)
        except Exception as exc:
            st.error(safe_error_message(exc))
            return

    st.subheader("Answer")
    if result.get("insufficient_context"):
        st.info(result["answer"])
    else:
        st.write(result["answer"])

    _render_metrics(result)
    _render_sources(result)

    with st.expander("Retrieval trace", expanded=False):
        st.json(result.get("retrieval_metadata", {}))

    st.download_button(
        label="Download answer and sources (JSON)",
        data=json.dumps(result, indent=2),
        file_name=f"cuda_answer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json",
    )


if __name__ == "__main__":
    main()
