#!/usr/bin/env python
import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from typing import List, Dict, Any

from src.config.load import load_config
from src.retrieval.embeddings import Embedder
from src.retrieval.faiss_index import load_index
from src.retrieval.retriever import Retriever, load_passages
from src.baseline.generation import generate_baseline
from src.detection.support_scoring import score_support, label_support
from src.correction.generation import correct_answer
from src.utils.io import read_jsonl, write_jsonl, ensure_dir
from src.utils.logging import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)

    passages_path = config["paths"]["corpus_passages"]
    index_dir = config["paths"]["index_dir"]
    questions_path = config["paths"]["questions_path"]
    outputs_dir = config["paths"]["outputs_dir"]

    supported_threshold = config["support"]["supported_threshold"]
    uncertain_threshold = config["support"]["uncertain_threshold"]

    embedder = Embedder(
        model_name=config["embedding"]["model_name"],
        batch_size=config["embedding"]["batch_size"],
    )
    index = load_index(index_dir)
    passages = load_passages(passages_path)

    retriever = Retriever(embedder, index, passages, top_k=config["retrieval"]["top_k"])

    questions = list(read_jsonl(questions_path))
    if args.limit and args.limit > 0:
        questions = questions[: args.limit]

    baseline_rows: List[Dict[str, Any]] = []
    corrected_rows: List[Dict[str, Any]] = []

    for q in questions:
        question = q["question"]
        qid = q.get("id")
        category = q.get("category")
        source = q.get("source")
        reference_answer = q.get("reference_answer")

        retrieved = retriever.retrieve(question)
        passages_payload = [
            {"pid": p.pid, "text": p.text, "score": p.score, "meta": p.meta}
            for p in retrieved
        ]

        baseline = generate_baseline(question, config)
        base_score = score_support(baseline, passages_payload, embedder)
        base_label = label_support(
            base_score,
            supported_threshold,
            uncertain_threshold,
        )

        if base_score < supported_threshold:
            corrected = correct_answer(question, baseline, passages_payload, config)
        else:
            corrected = baseline

        corr_score = score_support(corrected, passages_payload, embedder)
        corr_label = label_support(
            corr_score,
            supported_threshold,
            uncertain_threshold,
        )

        baseline_rows.append(
            {
                "id": qid,
                "question": question,
                "category": category,
                "source": source,
                "reference_answer": reference_answer,
                "answer": baseline,
                "baseline_answer": baseline,
                "support_score": base_score,
                "support_label": base_label,
                "passages": passages_payload,
                "retrieved_passages": passages_payload,
            }
        )

        corrected_rows.append(
            {
                "id": qid,
                "question": question,
                "category": category,
                "source": source,
                "reference_answer": reference_answer,
                "answer": corrected,
                "baseline_answer": baseline,
                "corrected_answer": corrected,
                "support_score": corr_score,
                "support_label": corr_label,
                "passages": passages_payload,
                "retrieved_passages": passages_payload,
            }
        )

    ensure_dir(outputs_dir)
    baseline_out = f"{outputs_dir}/baseline.jsonl"
    corrected_out = f"{outputs_dir}/corrected.jsonl"
    write_jsonl(baseline_out, baseline_rows)
    write_jsonl(corrected_out, corrected_rows)

    print(f"Wrote {len(baseline_rows)} baseline rows -> {baseline_out}")
    print(f"Wrote {len(corrected_rows)} corrected rows -> {corrected_out}")


if __name__ == "__main__":
    main()
