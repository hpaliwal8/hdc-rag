from typing import List

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover
    SentenceTransformer = None


class Embedder:
    def __init__(self, model_name: str, batch_size: int = 32):
        if SentenceTransformer is None:
            raise ImportError("sentence-transformers is not installed")
        self.model = SentenceTransformer(model_name)
        self.batch_size = batch_size

    def encode(self, texts: List[str]) -> np.ndarray:
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return np.asarray(vectors, dtype="float32")
