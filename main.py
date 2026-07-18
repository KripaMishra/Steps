import json
from datetime import datetime

import streamlit as st

from components.s5_rag_llm import RAGModel


@st.cache_resource
def get_rag_model() -> RAGModel:
    return RAGModel()


def main():
    st.title("RAG Application")
    query = st.text_area("Enter your query:", height=100)

    if st.button("Generate Answer"):
        if not query.strip():
            st.warning("Please enter a query.")
            return

        with st.spinner("Generating answer..."):
            try:
                rag_model = get_rag_model()
                result = rag_model.process_query(query)
                rag_model.save_query_results(result)
            except Exception as exc:
                st.error(str(exc))
                return

        if result.get("error"):
            st.error(result["error"])
            return

        st.subheader("Results:")
        st.write(f"**Original Query:** {result.get('original_query', query)}")
        st.write(f"**Timestamp:** {result.get('timestamp', '')}")

        st.subheader("Context:")
        st.json(result.get("context", {}))

        st.subheader("Answer:")
        st.write(result.get("answer", "No answer was generated."))

        st.download_button(
            label="Download Results as JSON",
            data=json.dumps(result, indent=2),
            file_name=f"query_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
        )


if __name__ == "__main__":
    main()
