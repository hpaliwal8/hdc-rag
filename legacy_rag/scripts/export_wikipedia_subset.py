#!/usr/bin/env python
import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config.load import load_config
from src.data.wiki_corpus import (
    export_documents_to_corpus,
    load_evidence_documents,
    load_wikipedia_subset,
)
from src.utils.logging import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--pool_size", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--query", default=None)
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)
    corpus_dir = config["paths"]["corpus_dir"]

    if args.query:
        docs = load_evidence_documents(
            query=args.query,
            api_limit=min(args.limit, 10),
            fallback_limit=args.limit,
            fallback_pool_size=args.pool_size,
        )
    else:
        docs = load_wikipedia_subset(
            limit=args.limit,
            pool_size=args.pool_size,
            seed=args.seed,
            query=None,
        )

    export_documents_to_corpus(docs, corpus_dir)
    print(f"Exported {len(docs)} Wikipedia articles to {corpus_dir}")


if __name__ == "__main__":
    main()
