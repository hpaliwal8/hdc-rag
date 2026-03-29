#!/usr/bin/env python
import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config.load import load_config
from src.retrieval.embeddings import Embedder
from src.retrieval.faiss_index import build_index, save_index
from src.retrieval.retriever import load_passages, passage_texts
from src.utils.io import ensure_dir
from src.utils.logging import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)

    passages_path = config["paths"]["corpus_passages"]
    index_dir = config["paths"]["index_dir"]

    embedder = Embedder(
        model_name=config["embedding"]["model_name"],
        batch_size=config["embedding"]["batch_size"],
    )

    passages = load_passages(passages_path)
    texts = passage_texts(passages)
    vectors = embedder.encode(texts)

    index = build_index(vectors)
    ensure_dir(index_dir)
    save_index(index, index_dir)

    print(f"Built index with {len(passages)} passages -> {index_dir}")


if __name__ == "__main__":
    main()
