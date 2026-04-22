"""
ingest.py
---------
Pipeline for getting documents into Pinecone:
extract text from PDF, chunk it, embed each chunk, and upsert
to the index with metadata.

Supports single-file ingestion and batch ingestion via a manifest.
"""

import json
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
    chunks: list[str],
    embeddings: list[list[float]],
    source: str,
    extra_metadata: dict | None = None,
) -> list[dict]:
    """Build Pinecone-ready vector records with metadata.

    Uses deterministic IDs (source + chunk index) so re-ingestion
    overwrites existing vectors rather than creating duplicates.
    """
    extra_metadata = extra_metadata or {}
    return [
        {
            "id": f"{source}_chunk_{i}",
            "values": embedding,
            "metadata": {
                "text": chunk,
                "source": source,
                "chunk_index": i,
                **extra_metadata,
            },
        }
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
    ]


def ingest_pdf(
    pdf_path: str,
    model: SentenceTransformer | None = None,
    extra_metadata: dict | None = None,
) -> int:
    """End-to-end ingestion: PDF → Pinecone.

    Args:
        pdf_path: Path to the PDF file.
        model: A preloaded SentenceTransformer (optional). If not provided,
               one will be loaded. Passing one in avoids reloading for batches.
        extra_metadata: Additional metadata fields to attach to each chunk.

    Returns:
        Number of chunks ingested.
    """
    source = Path(pdf_path).name
    print(f"Ingesting {source}...")

    text = extract_text(pdf_path)
    chunks = chunk_text(text)
    print(f"  → {len(chunks)} chunks created")

    if model is None:
        model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = embed_chunks(chunks, model)
    print(f"  → {len(embeddings)} embeddings generated")

    vectors = build_vectors(chunks, embeddings, source, extra_metadata)
    upsert_vectors(vectors)
    print(f"  → upserted to Pinecone")

    return len(chunks)


def batch_ingest(manifest_path: str, data_dir: str) -> int:
    """Ingest every PDF listed in a manifest file.

    The manifest is a JSON list of objects, each with a 'filename' key
    and arbitrary additional metadata fields (e.g. theme, stance, year).

    Args:
        manifest_path: Path to the manifest JSON file.
        data_dir: Directory containing the PDF files.

    Returns:
        Total number of chunks ingested across all files.
    """
    with open(manifest_path) as f:
        manifest = json.load(f)

    model = SentenceTransformer(EMBEDDING_MODEL)
    total_chunks = 0

    for entry in manifest:
        filename = entry["filename"]
        extra_metadata = {k: v for k, v in entry.items() if k != "filename"}
        pdf_path = Path(data_dir) / filename

        if not pdf_path.exists():
            print(f"  ⚠ Skipping {filename} — file not found")
            continue

        n = ingest_pdf(str(pdf_path), model=model, extra_metadata=extra_metadata)
        total_chunks += n

    return total_chunks


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 1:
        # No args → batch ingest using default paths
        n = batch_ingest("data/manifest.json", "data")
        print(f"\nDone. Ingested {n} chunks across the corpus.")
    elif len(sys.argv) == 2:
        # One arg → single file ingest
        n = ingest_pdf(sys.argv[1])
        print(f"\nDone. Ingested {n} chunks.")
    else:
        print("Usage: python ingest.py [path_to_pdf]")
        sys.exit(1)