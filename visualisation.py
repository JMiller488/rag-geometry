"""
visualisation.py
----------------
Utilities for visualising the geometry of the embedding space.
Uses PCA to project 384-dimensional chunk embeddings into 2D
for plotting, while reporting how much variance is preserved.
"""

import numpy as np
from sklearn.decomposition import PCA


def fit_pca(vectors: list[list[float]], n_components: int = 2) -> PCA:
    """Fit a PCA model on a list of high-dimensional vectors.

    Args:
        vectors: List of embedding vectors.
        n_components: Target dimensionality (2 for plotting).

    Returns:
        A fitted sklearn PCA object, which can later be used to
        transform new vectors (like a query) into the same 2D space.
    """
    X = np.array(vectors)
    pca = PCA(n_components=n_components)
    pca.fit(X)
    return pca


def project(pca: PCA, vectors: list[list[float]]) -> np.ndarray:
    """Project vectors into the PCA space.

    Args:
        pca: A PCA object already fitted on the corpus.
        vectors: One or more vectors to project.

    Returns:
        A (n, n_components) numpy array of projected coordinates.
    """
    X = np.array(vectors)
    return pca.transform(X)

def build_corpus_dataframe(
    vectors: list[dict], coords_2d: np.ndarray
):
    """Combine projected coordinates with metadata into a single DataFrame.

    Useful for plotting — each row is one chunk, with its 2D position
    and associated metadata available in one place.

    Args:
        vectors: The list returned from fetch_all_vectors().
        coords_2d: The (n, 2) array returned from project().

    Returns:
        A pandas DataFrame with columns: x, y, source, theme, stance,
        year, chunk_index, text.
    """
    import pandas as pd

    return pd.DataFrame({
        "x": coords_2d[:, 0],
        "y": coords_2d[:, 1],
        "source": [v["metadata"]["source"] for v in vectors],
        "theme": [v["metadata"].get("theme", "") for v in vectors],
        "stance": [v["metadata"].get("stance", "") for v in vectors],
        "year": [v["metadata"].get("year", "") for v in vectors],
        "chunk_index": [v["metadata"]["chunk_index"] for v in vectors],
        "text": [v["metadata"]["text"][:200] for v in vectors],  # truncate for hover
    })