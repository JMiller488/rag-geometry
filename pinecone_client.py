"""
pinecone_client.py
------------------
Handles connection to Pinecone and basic vector operations
(upsert and query). Reads credentials from environment variables.
"""

import os
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")


def get_index():
    """Connect to Pinecone and return a reference to the index."""
    pc = Pinecone(api_key=API_KEY)
    return pc.Index(INDEX_NAME)


def upsert_vectors(vectors: list[dict]) -> None:
    """Insert or update vectors in the index.

    Args:
        vectors: List of dicts, each with keys 'id', 'values', and 'metadata'.
    """
    index = get_index()
    index.upsert(vectors=vectors)


def query(vector: list[float], top_k: int = 5) -> dict:
    """Query the index with a vector and return the top_k nearest matches.

    Args:
        vector: Query vector as a list of floats.
        top_k: Number of nearest neighbours to return.

    Returns:
        Dict containing matches with their IDs, scores, and metadata.
    """
    index = get_index()
    return index.query(vector=vector, top_k=top_k, include_metadata=True)


if __name__ == "__main__":
    index = get_index()
    stats = index.describe_index_stats()
    print("Connected to Pinecone successfully.")
    print(f"Index stats: {stats}")