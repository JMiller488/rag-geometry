"""
ingest.py
---------
Pipeline for getting documents into Pinecone:
extract text from PDF, chunk it, embed each chunk, and upsert
to the index with metadata.
"""

import uuid
from pathlib import Path

import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from pinecone_client import upsert_vectors

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def extract_text(pdf_path: str) -> str:
    """Extract all text from a PDF file."""
    doc = fitz.open(pdf_path)
    return "\n".join(page.get_text() for page in doc)


def chunk_text(text: str) -> list[str]:
    """Split text into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_text(text)


def embed_chunks(chunks: list[str], model: SentenceTransformer) -> list[list[float]]:
    """Embed a list of chunks into vectors."""
    embeddings = model.encode(chunks, convert_to_numpy=True)
    return embeddings.tolist()


def build_vectors(
    chunks: list[str], embeddings: list[list[float]], source: str
) -> list[dict]:
    """Build Pinecone-ready vector records with metadata."""
    return [
        {
            "id": str(uuid.uuid4()),
            "values": embedding,
            "metadata": {
                "text": chunk,
                "source": source,
                "chunk_index": i,
            },
        }
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
    ]


def ingest_pdf(pdf_path: str) -> int:
    """End-to-end ingestion: PDF → Pinecone.

    Returns:
        Number of chunks ingested.
    """
    source = Path(pdf_path).name
    print(f"Ingesting {source}...")

    text = extract_text(pdf_path)
    chunks = chunk_text(text)
    print(f"  → {len(chunks)} chunks created")

    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = embed_chunks(chunks, model)
    print(f"  → {len(embeddings)} embeddings generated")

    vectors = build_vectors(chunks, embeddings, source)
    upsert_vectors(vectors)
    print(f"  → upserted to Pinecone")

    return len(chunks)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python ingest.py <path_to_pdf>")
        sys.exit(1)

    n = ingest_pdf(sys.argv[1])
    print(f"\nDone. Ingested {n} chunks.")