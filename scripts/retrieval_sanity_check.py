#!/usr/bin/env python
import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config.load import load_config
from src.retrieval.embeddings import Embedder
from src.retrieval.faiss_index import load_index
from src.retrieval.retriever import Retriever, load_passages
from src.utils.io import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--top_k", type=int, default=3)
    args = parser.parse_args()

    config = load_config(args.config)
    passages_path = config["paths"]["corpus_passages"]
    index_dir = config["paths"]["index_dir"]
    questions_path = config["paths"]["questions_path"]

    embedder = Embedder(
        model_name=config["embedding"]["model_name"],
        batch_size=config["embedding"]["batch_size"],
    )
    index = load_index(index_dir)
    passages = load_passages(passages_path)

    retriever = Retriever(
        embedder,
        index,
        passages,
        top_k=args.top_k,
        bm25_k=config["retrieval"].get("bm25_k", 20),
        alpha=config["retrieval"].get("alpha", 0.85),
    )
    questions = list(read_jsonl(questions_path))[: args.limit]

    for q in questions:
        print("=" * 80)
        print(f"Q: {q['question']}")
        results = retriever.retrieve(q["question"])
        for i, p in enumerate(results, start=1):
            title = p.meta.get("title", "")
            print(
                f"\n[{i}] score={p.score:.4f} dense={p.dense_score:.4f} "
                f"bm25={p.bm25_score:.4f} bonus={p.title_bonus:.4f} "
                f"pid={p.pid} title={title}"
            )
            print(p.text[:500].replace("\n", " "))


if __name__ == "__main__":
    main()
