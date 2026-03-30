from typing import Iterable, List

import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self, model_name: str, batch_size: int = 32):
        self.model_name = model_name
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name)

    def _prefix(self, text: str, kind: str) -> str:
        if "e5" not in self.model_name.lower():
            return text
        prefix = "query: " if kind == "query" else "passage: "
        return prefix + text

    def _sanitize_and_normalize(self, embeddings: np.ndarray, label: str) -> np.ndarray:
        embeddings = np.asarray(embeddings, dtype="float32")

        non_finite_mask = ~np.isfinite(embeddings)
        if np.any(non_finite_mask):
            bad_values = int(non_finite_mask.sum())
            bad_rows = int(np.any(non_finite_mask, axis=1).sum())
            print(
                f"[embedding] Replacing {bad_values} non-finite values "
                f"across {bad_rows} {label} embedding(s)."
            )
            embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=0.0, neginf=0.0)

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        zero_norm_mask = np.isclose(norms.squeeze(-1), 0.0)
        if np.any(zero_norm_mask):
            zero_count = int(zero_norm_mask.sum())
            print(f"[embedding] Found {zero_count} zero-norm {label} embedding(s).")
            embeddings[zero_norm_mask] = 0.0
            norms[zero_norm_mask] = 1.0

        embeddings = embeddings / norms
        return embeddings.astype("float32")

    def encode(self, texts: Iterable[str], kind: str = "passage") -> np.ndarray:
        prepared: List[str] = [self._prefix(str(text), kind) for text in texts]
        embeddings = self.model.encode(
            prepared,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=(kind == "passage"),
        )
        return self._sanitize_and_normalize(embeddings, label=kind)

    def encode_passages(self, texts: Iterable[str]) -> np.ndarray:
        return self.encode(texts, kind="passage")

    def encode_queries(self, texts: Iterable[str]) -> np.ndarray:
        embeddings = self.encode(texts, kind="query")
        if not np.isfinite(embeddings).all():
            raise ValueError("Query embeddings contain non-finite values after sanitization.")
        return embeddings
