#!/usr/bin/env python
import argparse
import json
import os
import random
import sys
from collections import defaultdict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.io import read_jsonl, ensure_dir
from src.utils.logging import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--per_type", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    setup_logging()
    ensure_dir(os.path.dirname(args.output))
    rng = random.Random(args.seed)

    by_type: dict = defaultdict(list)
    for path in args.inputs:
        for r in read_jsonl(path):
            t = r.get("hallucination_type")
            if t:
                by_type[t].append(r)

    sampled: dict = {}
    for t, rows in by_type.items():
        rng.shuffle(rows)
        sampled[t] = rows[: args.per_type]

    with open(args.output, "w", encoding="utf-8") as f:
        for t, rows in sampled.items():
            f.write(f"\n{'=' * 80}\n")
            f.write(f"TYPE: {t}  (n={len(rows)})\n")
            f.write(f"{'=' * 80}\n")
            for i, r in enumerate(rows, 1):
                f.write(f"\n--- Example {i} ---\n")
                f.write(f"Model:    {r.get('model_id')}\n")
                f.write(f"Prompt:   {r.get('prompt_id')}\n")
                f.write(f"Dataset:  {r.get('dataset')}\n")
                level = r.get("level")
                if level:
                    f.write(f"Level:    {level}\n")
                f.write(f"\nQ: {r.get('question', '')}\n")
                f.write(f"\nReference: {r.get('reference_answer', '')}\n")
                evidence = r.get("evidence")
                if evidence:
                    f.write(f"\nEvidence: {evidence[:500]}\n")
                f.write(f"\nA: {r.get('answer', '')[:500]}\n")
                conf = r.get("confidence")
                if conf is not None:
                    f.write(f"\nNLI: {r.get('nli_label')} (conf={conf:.3f})\n")
                f.write("\n")

    counts = {t: len(rows) for t, rows in sampled.items()}
    print(f"Examples written to {args.output}")
    print(f"Counts per type: {counts}")


if __name__ == "__main__":
    main()
