"""
retrieval.py
------------
Retrieves relevant chunks from Pinecone given a query.
Embeds the query using the same model as ingestion to ensure
both live in the same vector space.
"""

from sentence_transformers import SentenceTransformer

from pinecone_client import query

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_TOP_K = 5


def embed_query(question: str, model: SentenceTransformer) -> list[float]:
    """Embed a question string into a vector."""
    embedding = model.encode(question, convert_to_numpy=True)
    return embedding.tolist()


def retrieve(question: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
    """Retrieve the top_k most relevant chunks for a question.

    Args:
        question: The user's question as a string.
        top_k: Number of chunks to return.

    Returns:
        List of dicts, each with 'text', 'source', 'chunk_index', and 'score'.
    """
    model = SentenceTransformer(EMBEDDING_MODEL)
    query_vector = embed_query(question, model)
    results = query(query_vector, top_k=top_k)

    return [
        {
            "text": match["metadata"]["text"],
            "source": match["metadata"]["source"],
            "chunk_index": match["metadata"]["chunk_index"],
            "score": match["score"],
        }
        for match in results["matches"]
    ]


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python retrieval.py 'your question here'")
        sys.exit(1)

    question = " ".join(sys.argv[1:])
    print(f"Question: {question}\n")

    results = retrieve(question)
    for i, result in enumerate(results, start=1):
        print(f"--- Result {i} (score: {result['score']:.4f}) ---")
        print(f"Source: {result['source']} (chunk {result['chunk_index']})")
        print(f"Text: {result['text'][:300]}...")
        print()