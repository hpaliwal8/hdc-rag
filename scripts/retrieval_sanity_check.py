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

    retriever = Retriever(embedder, index, passages, top_k=args.top_k)
    questions = list(read_jsonl(questions_path))[: args.limit]

    for q in questions:
        print("=" * 80)
        print(f"Q: {q['question']}")
        results = retriever.retrieve(q["question"])
        for i, p in enumerate(results, start=1):
            print(f"\n[{i}] score={p.score:.4f} pid={p.pid}")
            print(p.text[:500].replace("\n", " "))


if __name__ == "__main__":
    main()
