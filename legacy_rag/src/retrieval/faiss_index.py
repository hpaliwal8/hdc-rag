import os
from typing import Tuple

import faiss
import numpy as np

from src.utils.io import ensure_dir


def build_index(vectors: np.ndarray) -> faiss.Index:
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    return index


def save_index(index: faiss.Index, index_dir: str) -> None:
    ensure_dir(index_dir)
    path = os.path.join(index_dir, "faiss.index")
    faiss.write_index(index, path)


def load_index(index_dir: str) -> faiss.Index:
    path = os.path.join(index_dir, "faiss.index")
    return faiss.read_index(path)


def search(index: faiss.Index, query_vectors: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
    scores, ids = index.search(query_vectors, top_k)
    return scores, ids
