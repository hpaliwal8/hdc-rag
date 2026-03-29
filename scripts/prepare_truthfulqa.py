#!/usr/bin/env python
import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config.load import load_config
from src.data.truthfulqa_loader import load_truthfulqa_subset, build_project_schema
from src.utils.io import ensure_dir
from src.utils.logging import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--subset_size", type=int, default=120)
    parser.add_argument("--no_stratify", action="store_true")
    parser.add_argument("--dataset_name", default="domenicrosati/TruthfulQA")
    parser.add_argument("--split", default="train")
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)

    df = load_truthfulqa_subset(
        subset_size=args.subset_size,
        seed=args.seed,
        stratify_by_category=not args.no_stratify,
        dataset_name=args.dataset_name,
        split=args.split,
    )
    project_df = build_project_schema(df)

    out_path = config["paths"]["questions_path"]
    ensure_dir("/".join(out_path.split("/")[:-1]))
    project_df.to_json(out_path, orient="records", lines=True)
    print(f"Wrote {len(project_df)} questions to {out_path}")


if __name__ == "__main__":
    main()
