import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


def semantic_search(
    query: str,
    model: SentenceTransformer,
    embeddings: np.ndarray,
    data: pd.DataFrame,
    top_k: int = 3,
) -> pd.DataFrame:
    """
    Perform semantic similarity search over precomputed embeddings.
    """

    # Encode query
    query_embedding = model.encode([query], convert_to_numpy=True)

    # Compute cosine similarity
    similarities = cosine_similarity(query_embedding,embeddings)[0]

    # Attach similarity scores
    results = data.copy()
    results["score"] = similarities

    # Sort and return top-k results
    return results.sort_values(
        by="score",
        ascending=False
    ).head(top_k)[["domain", "text", "score"]]


