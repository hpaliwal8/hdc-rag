#!/usr/bin/env python
import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.metrics.eval import compute_summary
from src.utils.io import read_jsonl
from src.utils.logging import setup_logging


def flag_overconfident_abstention(records: list) -> list:
    groups = defaultdict(dict)
    for r in records:
        key = (r.get("model_id"), r.get("id"))
        groups[key][r.get("prompt_id")] = r

    flagged = []
    for r in records:
        key = (r.get("model_id"), r.get("id"))
        prompt_variants = groups[key]

        plain = prompt_variants.get("plain")
        abstain = prompt_variants.get("abstain")

        r = dict(r)
        if (
            plain and abstain
            and r.get("prompt_id") == "plain"
            and r.get("is_hallucinated")
            and abstain.get("hallucination_type") == "abstained"
        ):
            r["overconfident_abstention_failure"] = True
        else:
            r["overconfident_abstention_failure"] = False

        flagged.append(r)

    return flagged


def print_summary(rows: list) -> None:
    header = f"{'Model':<25} {'Prompt':<12} {'Total':>6} {'EM':>6} {'F1':>6} {'Hall%':>7} {'Abst%':>7}"
    print(header)
    print("-" * len(header))

    for row in rows:
        print(
            f"{row['model_id']:<25} "
            f"{row['prompt_id']:<12} "
            f"{row['total']:>6} "
            f"{row['exact_match']:>6.3f} "
            f"{row['token_f1']:>6.3f} "
            f"{row['hallucination_rate']:>7.3f} "
            f"{row['abstention_rate']:>7.3f}"
        )

    print()
    print("Hallucination type breakdown:")
    for row in rows:
        print(f"  {row['model_id']} / {row['prompt_id']}: {row['type_counts']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Labeled output JSONL file.")
    parser.add_argument("--output", default=None, help="Optional path to write metrics JSON.")
    args = parser.parse_args()

    setup_logging()

    records = list(read_jsonl(args.input))
    records = flag_overconfident_abstention(records)

    overconfident_count = sum(1 for r in records if r.get("overconfident_abstention_failure"))
    print(f"Loaded {len(records)} records. "
          f"Overconfident abstention failures: {overconfident_count}\n")

    summary = compute_summary(records)
    print_summary(summary)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "total_records": len(records),
                       "overconfident_abstention_failures": overconfident_count}, f, indent=2)
        print(f"\nMetrics written to {args.output}")


if __name__ == "__main__":
    main()
