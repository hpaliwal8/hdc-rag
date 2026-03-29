#!/usr/bin/env python
import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.io import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True)
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    rows = list(read_jsonl(args.path))[: args.limit]
    for i, row in enumerate(rows, start=1):
        print("=" * 80)
        print(f"[{i}] ID: {row.get('id')}")
        print(f"Q: {row.get('question')}")
        if row.get("reference_answer"):
            print(f"Ref: {row.get('reference_answer')}")
        print(f"Ans: {row.get('answer')}")


if __name__ == "__main__":
    main()
