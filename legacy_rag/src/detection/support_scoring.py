from typing import List, Dict, Any

import numpy as np

from src.retrieval.embeddings import Embedder


def max_cosine_similarity(answer_vec: np.ndarray, passage_vecs: np.ndarray) -> float:
    if passage_vecs.size == 0:
        return 0.0
    scores = passage_vecs @ answer_vec.T
    return float(scores.max())


def score_support(answer: str, passages: List[Dict[str, Any]], embedder: Embedder) -> float:
    if not passages:
        return 0.0
    answer_vec = embedder.encode_queries([answer])[0]
    passage_texts = [p["text"] for p in passages]
    passage_vecs = embedder.encode_passages(passage_texts)
    return max_cosine_similarity(answer_vec, passage_vecs)


def label_support(score: float, supported_threshold: float, uncertain_threshold: float) -> str:
    if score >= supported_threshold:
        return "supported"
    if score >= uncertain_threshold:
        return "uncertain"
    return "unsupported"
