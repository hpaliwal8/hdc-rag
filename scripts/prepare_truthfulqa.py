#!/usr/bin/env python
import argparse
import os
import random
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config.load import load_config
from src.data.truthfulqa_loader import load_truthfulqa
from src.utils.io import ensure_dir, write_jsonl
from src.utils.logging import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--limit", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)

    rows = load_truthfulqa(split=args.split)
    random.Random(args.seed).shuffle(rows)
    rows = rows[: args.limit]

    out_path = config["paths"]["questions_path"]
    ensure_dir("/".join(out_path.split("/")[:-1]))
    write_jsonl(out_path, rows)
    print(f"Wrote {len(rows)} questions to {out_path}")


if __name__ == "__main__":
    main()
