# RAG Geometry

A retrieval-augmented generation (RAG) system built from first principles, with a particular focus on the **geometry of embedding spaces**. The project ingests a corpus of endurance training and triathlon research papers (interest areas of mine), indexes them in a Pinecone vector database, and exposes both a question-answering interface and an interactive 2-dimensional geometric interpretation of the embedded chunks and query.

## Motivation
I wanted to understand RAG from first principles. As a mathematician (with a particular interest in geometry), I wanted to get understand the whole RAG process from a geometrical perspective (embedding, cosine similarity, PCA projection etc.). The end result lets you actually *see* the retrieval happening in a PCA generated 2-dimensional space. My hope is that having a thorough understanding of RAG will allow me to better build and understand Agentic systems.

## What it does

- **Ingests PDFs** into 500-character overlapping chunks, embeds each chunk using `all-MiniLM-L6-v2` (384 dimensions), and upserts them into Pinecone with metadata (source, theme, stance, year, chunk index).
- **Retrieves** relevant chunks for any natural-language question by embedding the query into the same vector space and searching by cosine similarity.
- **Generates** grounded answers using a local Ollama / Mistral instance, with the retrieved chunks as context.
- **Visualises** the entire 384-dimensional corpus as a 2D PCA projection coloured by theme. Queries and retrieved chunks overlay on the same plot.

## The corpus

Six papers across three themes:

| Theme | Papers |
| --- | --- |
| Zone 2 training | Valenzuela et al. (pro), Storoschuk & Gibala (skeptic) |
| Periodisation | Hydren (polarised training review), Clemente-Suárez et al. (reverse vs traditional) |
| Endurance nutrition | 2025 carbohydrate review, GSSI carbohydrate guidelines |

The corpus deliberately contains internal disagreement (Zone 2 pro/skeptic) so that retrieval can be tested on contested questions.

## Mathematical framing

The system is fundamentally an exercise in geometry. Each chunk is a point in $\mathbb{R}^{384}$, placed there by a sentence transformer trained so that semantic similarity corresponds to geometric proximity. A user query is mapped into the same space by the same model, and retrieval simply - and elegantly - reduces to finding the nearest neighbours by cosine similarity.

The visualisation tab applies PCA to the 705-chunk corpus, projecting it into $\mathbb{R}^2$ via the two principal components. The 2D projection preserves approximately **20% of the total geometric variance** - typical for such text embeddings, which place semantic information across many dimensions rather than concentrating it on a few.

## Architecture

```
rag-geometry/
├── pinecone_client.py    # Vector database interface (upsert, query, fetch_all)
├── ingest.py             # PDF → chunks → embeddings → Pinecone
├── retrieval.py          # Query-time retrieval with cosine similarity
├── llm.py                # Answer generation via local Ollama
├── visualisation.py      # PCA projection of the embedding space
├── app.py                # Streamlit UI (Ask + Geometry tabs)
├── data/
│   └── manifest.json     # Per-document metadata for batch ingestion
├── requirements.txt
└── README.md
```

Each module owns one responsibility. For example swapping Pinecone for a different vector database, or swapping the LLM for any provider, requires changing only the corresponding file — the rest of the system is unaffected.

## Running it

### Prerequisites

- Python 3.12
- A Pinecone account (free tier is sufficient)
- Ollama installed locally with the Mistral model pulled (`ollama pull mistral`)

### Setup

```bash
git clone https://github.com/JMiller488/rag-geometry.git
cd rag-geometry
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file at the project root:

```
PINECONE_API_KEY=your_key_here
PINECONE_INDEX_NAME=rag-geometry
```

Configure your Pinecone index manually with **dimension 384** and **cosine similarity**. The free serverless tier on AWS `us-east-1` works fine.

### Ingestion

Place your PDFs in `data/` and update `data/manifest.json` to match. Then:

```bash
python ingest.py
```

This batch-ingests every document in the manifest with full metadata.

### Running the app

Make sure Ollama is running, then:

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

## Implementation notes

- **Deterministic IDs.** Each chunk's Pinecone ID is `{filename}_chunk_{index}`, which makes re-ingestion idempotent — running the pipeline twice updates existing chunks rather than creating duplicates.
- **Manifest-driven ingestion.** Per-document metadata lives in a JSON manifest rather than being hardcoded, so adding a new paper to the corpus requires no code changes.
- **PCA fitted once, applied repeatedly.** The PCA model is fitted on the full corpus once per session (cached via Streamlit), then used to project query vectors into the same 2D space. This separation of `fit` and `transform` is what makes the query-overlay visualisation geometrically coherent.
- **Single-document ingestion preserved.** `python ingest.py path/to/file.pdf` still works for ad-hoc ingestion, alongside the batch flow.

## Known limitations

- The embedding model uses mean-pooling without IDF weighting, so common tokens can dominate similarity for short strings.
- The PCA projection captures ~20% of total variance; fine-grained local structure in the 2D plot is unreliable.
- Retrieval has no concept of "I don't know" — the system always returns the top-k closest chunks, even for off-topic queries. The LLM prompt is designed to mitigate this, but it's a structural property of the retrieval layer.
- The corpus is small (~705 chunks). Ingestion of an order-of-magnitude larger corpus would benefit from paginated retrieval in `fetch_all_vectors` and possibly a different visualisation approach.

## Tech stack

- **Embeddings:** `sentence-transformers` (`all-MiniLM-L6-v2`)
- **Vector database:** Pinecone (serverless)
- **LLM:** Ollama with Mistral (local)
- **Chunking:** LangChain `RecursiveCharacterTextSplitter`
- **PDF extraction:** PyMuPDF
- **Dimensionality reduction:** scikit-learn PCA
- **Plotting:** Plotly Express
- **UI:** Streamlit