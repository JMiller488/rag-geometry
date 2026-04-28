"""
app.py
------
Streamlit interface for the RAG-Geometry project.
Two tabs:
 - Ask: the RAG pipeline (retrieval + answer)
 - Geometry: visualisation of the embedding space
"""

import streamlit as st
import plotly.express as px

from retrieval import retrieve
from llm import generate_answer
from pinecone_client import fetch_all_vectors
from visualisation import fit_pca, project, build_corpus_dataframe

st.set_page_config(page_title="RAG Geometry", layout="wide")

st.title("RAG Geometry")
st.caption(
    "An exploration of retrieval-augmented generation, "
    "with a focus on the geometry of embedding spaces. "
    "Corpus: endurance training and triathlon research."
)


# --- Cache heavy operations so they only run once per session ---
@st.cache_resource
def load_corpus_and_pca():
    """Fetch all vectors from Pinecone and fit PCA once per session."""
    vectors = fetch_all_vectors()
    embeddings = [v["values"] for v in vectors]
    pca = fit_pca(embeddings)
    coords_2d = project(pca, embeddings)
    df = build_corpus_dataframe(vectors, coords_2d)
    variance_explained = sum(pca.explained_variance_ratio_)
    return df, pca, variance_explained


# --- Tabs ---
tab_ask, tab_geometry = st.tabs(["Ask", "Geometry"])

# --- Ask tab ---
with tab_ask:
    st.sidebar.header("Retrieval settings")
    top_k = st.sidebar.slider("Number of chunks to retrieve", 1, 10, 5)

    question = st.text_input(
        "Ask a question about endurance training, periodisation, or nutrition:",
        placeholder="e.g. What is zone 2 training?",
    )

    if question:
        with st.spinner("Retrieving relevant chunks..."):
            chunks = retrieve(question, top_k=top_k)

        st.header("Retrieved chunks")
        for i, chunk in enumerate(chunks, start=1):
            with st.expander(
                f"Chunk {i} — score {chunk['score']:.3f} — {chunk['source']}"
            ):
                st.markdown(f"**Source:** `{chunk['source']}`")
                st.markdown(f"**Chunk index:** {chunk['chunk_index']}")
                st.markdown(f"**Cosine similarity:** {chunk['score']:.4f}")
                st.markdown("---")
                st.write(chunk["text"])

        st.header("Answer")
        with st.spinner("Generating answer with Ollama..."):
            answer = generate_answer(question, chunks)
        st.write(answer)
    else:
        st.info("Enter a question above to get started.")

# --- Geometry tab ---
with tab_geometry:
    st.header("The embedding space")
    st.caption(
        "Each point is one chunk of the corpus, projected from 384 dimensions "
        "to 2 using Principal Component Analysis. Hover to see details."
    )

    df, pca, variance_explained = load_corpus_and_pca()

    st.metric(
        "Variance preserved by 2D projection",
        f"{variance_explained:.1%}",
        help=(
            "The fraction of the original 384-dimensional geometric variance "
            "captured by the two principal components. Typical range for "
            "text embeddings is 15–30%. Distances in this plot are "
            "qualitative, not quantitative."
        ),
    )

    fig = px.scatter(
    df,
    x="x",
    y="y",
    color="theme",
    hover_data=["source", "chunk_index", "text", "stance"],
    title="Corpus projected to 2D, coloured by theme",
    )
    fig.update_traces(marker={"size": 6, "opacity": 0.6})
    fig.update_layout(height=600)

    st.plotly_chart(fig, use_container_width=True)