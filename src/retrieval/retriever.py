from dataclasses import dataclass
from typing import List, Dict, Any

from src.retrieval.embeddings import Embedder
from src.retrieval.faiss_index import search


@dataclass
class Passage:
    pid: str
    text: str
    meta: Dict[str, Any]
    score: float


class Retriever:
    def __init__(self, embedder: Embedder, index, passages: List[Dict[str, Any]], top_k: int):
        self.embedder = embedder
        self.index = index
        self.passages = passages
        self.top_k = top_k

    def retrieve(self, question: str) -> List[Passage]:
        qvec = self.embedder.encode([question])
        scores, ids = search(self.index, qvec, self.top_k)
        results: List[Passage] = []
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0:
                continue
            p = self.passages[int(idx)]
            results.append(
                Passage(
                    pid=p["pid"],
                    text=p["text"],
                    meta=p.get("meta", {}),
                    score=float(score),
                )
            )
        return results


def load_passages(path: str) -> List[Dict[str, Any]]:
    from src.utils.io import read_jsonl

    return list(read_jsonl(path))


def passage_texts(passages: List[Dict[str, Any]]) -> List[str]:
    return [p["text"] for p in passages]
