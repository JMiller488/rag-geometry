"""
app.py
------
Streamlit interface for the RAG-Geometry project.
Lets the user ask a question, see the retrieved chunks,
and view the LLM-generated answer.
"""

import streamlit as st

from retrieval import retrieve
from llm import generate_answer

st.set_page_config(page_title="RAG Geometry", layout="wide")

st.title("RAG Geometry")
st.caption(
    "An exploration of retrieval-augmented generation, "
    "with a focus on the geometry of embedding spaces. "
    "Corpus: endurance training and triathlon research."
)

# --- Sidebar controls ---
st.sidebar.header("Retrieval settings")
top_k = st.sidebar.slider("Number of chunks to retrieve", 1, 10, 5)

# --- Main interface ---
question = st.text_input(
    "Ask a question about endurance training, periodisation, or nutrition:",
    placeholder="e.g. What is zone 2 training?",
)

if question:
    with st.spinner("Retrieving relevant chunks..."):
        chunks = retrieve(question, top_k=top_k)

    # --- Show retrieved chunks ---
    st.header("Retrieved chunks")
    st.caption(
        f"The {top_k} chunks closest to your question by cosine similarity, "
        f"out of all 705 chunks in the corpus."
    )

    for i, chunk in enumerate(chunks, start=1):
        with st.expander(
            f"Chunk {i} — score {chunk['score']:.3f} — {chunk['source']}"
        ):
            st.markdown(f"**Source:** `{chunk['source']}`")
            st.markdown(f"**Chunk index:** {chunk['chunk_index']}")
            st.markdown(f"**Cosine similarity:** {chunk['score']:.4f}")
            st.markdown("---")
            st.write(chunk["text"])

    # --- Generate answer ---
    st.header("Answer")
    with st.spinner("Generating answer with Ollama..."):
        answer = generate_answer(question, chunks)
    st.write(answer)

else:
    st.info("Enter a question above to get started.")