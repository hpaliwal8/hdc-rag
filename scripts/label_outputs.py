#!/usr/bin/env python
import argparse
import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.labeling.evidence import extract_evidence
from src.labeling.nli import classify
from src.labeling.heuristics import classify_hallucination_type
from src.utils.io import read_jsonl, ensure_dir
from src.utils.logging import setup_logging


def load_hotpotqa_index(path: str) -> dict:
    index = {}
    for row in read_jsonl(path):
        index[row["id"]] = row
    return index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Experiment output JSONL file.")
    parser.add_argument("--hotpotqa", required=True, help="Processed HotpotQA JSONL file.")
    parser.add_argument("--output", required=True, help="Labeled output JSONL file.")
    args = parser.parse_args()

    setup_logging()

    hotpotqa_index = load_hotpotqa_index(args.hotpotqa)
    ensure_dir(os.path.dirname(args.output))

    with open(args.output, "w", encoding="utf-8") as out_f:
        for record in read_jsonl(args.input):
            row_id = record.get("id")
            answer = record.get("answer", "")
            dataset = record.get("dataset", "")

            if dataset != "hotpotqa" or row_id not in hotpotqa_index:
                out_f.write(json.dumps(record) + "\n")
                out_f.flush()
                continue

            hotpotqa_row = hotpotqa_index[row_id]
            supporting_facts = hotpotqa_row.get("supporting_facts", [])
            context = hotpotqa_row.get("context", {})
            question_type = hotpotqa_row.get("type", "")

            evidence = extract_evidence(supporting_facts, context)

            nli_result = classify(evidence, answer)

            hallucination_type = classify_hallucination_type(
                nli_result=nli_result,
                answer=answer,
                evidence=evidence,
                question_type=question_type,
            )

            is_hallucinated = hallucination_type not in ("supported", "abstained")

            labeled = {
                **record,
                "evidence": evidence,
                "nli_label": nli_result["nli_label"],
                "confidence": nli_result["confidence"],
                "prob_entailment": nli_result["prob_entailment"],
                "prob_neutral": nli_result["prob_neutral"],
                "prob_contradiction": nli_result["prob_contradiction"],
                "hallucination_type": hallucination_type,
                "is_hallucinated": is_hallucinated,
            }

            out_f.write(json.dumps(labeled) + "\n")
            out_f.flush()

    print(f"Labeled outputs written to {args.output}")


if __name__ == "__main__":
    main()
