#!/usr/bin/env python
import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from typing import List, Dict

from src.config.load import load_config
from src.data.wiki_corpus import iter_text_files
from src.retrieval.chunking import chunk_text_words
from src.utils.io import ensure_dir, write_jsonl
from src.utils.logging import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)
    corpus_dir = config["paths"]["corpus_dir"]
    out_path = config["paths"]["corpus_passages"]

    chunk_words = config["chunking"]["chunk_words"]
    overlap_words = config["chunking"]["overlap_words"]

    rows: List[Dict] = []
    for doc in iter_text_files(corpus_dir):
        text = doc["text"].strip()
        if doc.get("title"):
            text = f"Title: {doc['title']}\n\n{text}"
        chunks = chunk_text_words(text, chunk_words, overlap_words)
        for i, chunk in enumerate(chunks):
            rows.append(
                {
                    "pid": f"{doc['doc_id']}::{i}",
                    "text": chunk,
                    "meta": {
                        "doc_id": doc["doc_id"],
                        "title": doc.get("title", ""),
                        "source_path": doc["source_path"],
                        "chunk": i,
                    },
                }
            )

    ensure_dir("/".join(out_path.split("/")[:-1]))
    write_jsonl(out_path, rows)
    print(f"Wrote {len(rows)} passages to {out_path}")


if __name__ == "__main__":
    main()
