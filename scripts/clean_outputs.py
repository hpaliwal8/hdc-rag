#!/usr/bin/env python
import argparse
import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.io import read_jsonl, ensure_dir
from src.utils.logging import setup_logging

REASONING_PREFIX = "If evidence is insufficient, say so."
MARKER = "Answer:"
ROLE_PREFIXES = ("assistant\n", "assistant ")


def clean_answer(answer: str, prompt_id: str) -> str:
    idx = answer.find(MARKER)
    if idx == -1:
        return _strip_role_prefix(answer.strip())

    cleaned = answer[idx + len(MARKER):].strip()

    if prompt_id == "reasoning" and cleaned.startswith(REASONING_PREFIX):
        cleaned = cleaned[len(REASONING_PREFIX):].strip()

    return _strip_role_prefix(cleaned)


def _strip_role_prefix(text: str) -> str:
    lower = text.lower()
    for prefix in ROLE_PREFIXES:
        if lower.startswith(prefix):
            return text[len(prefix):].strip()
    return text


def clean_file(input_path: str, output_path: str) -> None:
    cleaned = 0
    unchanged = 0

    with open(output_path, "w", encoding="utf-8") as out_f:
        for record in read_jsonl(input_path):
            original = record.get("answer", "")
            prompt_id = record.get("prompt_id", "")
            record["answer"] = clean_answer(original, prompt_id)

            if record["answer"] != original:
                cleaned += 1
            else:
                unchanged += 1

            out_f.write(json.dumps(record) + "\n")
            out_f.flush()

    print(f"{os.path.basename(input_path)} → cleaned: {cleaned}, unchanged: {unchanged}")


def spot_check(output_path: str, n: int = 3) -> None:
    print(f"\nSpot check — {os.path.basename(output_path)}:")
    for i, record in enumerate(read_jsonl(output_path)):
        if i >= n:
            break
        print(f"  Q: {record['question']}")
        print(f"  A: {record['answer']}")
        print(f"  prompt: {record['prompt_id']}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", nargs="+", required=True, help="Input JSONL file(s).")
    parser.add_argument("--output_dir", required=True, help="Directory to write cleaned files.")
    args = parser.parse_args()

    setup_logging()
    ensure_dir(args.output_dir)

    for input_path in args.input:
        filename = os.path.basename(input_path)
        output_path = os.path.join(args.output_dir, filename)
        clean_file(input_path, output_path)
        spot_check(output_path)


if __name__ == "__main__":
    main()
