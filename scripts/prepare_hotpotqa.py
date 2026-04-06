#!/usr/bin/env python
import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config.load import load_config
from src.data.hotpotqa_loader import load_hotpotqa_subset, build_project_schema
from src.utils.io import ensure_dir
from src.utils.logging import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--subset_size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset_name", default="hotpot_qa")
    parser.add_argument("--config_name", default="distractor")
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)

    df = load_hotpotqa_subset(
        subset_size=args.subset_size,
        seed=args.seed,
        dataset_name=args.dataset_name,
        config_name=args.config_name,
        split=args.split,
    )
    project_df = build_project_schema(df)

    paths = config.get("paths", {})
    out_path = paths.get("hotpotqa_path")
    if not out_path:
        raise KeyError("Missing paths.hotpotqa_path in config")

    ensure_dir("/".join(out_path.split("/")[:-1]))
    project_df.to_json(out_path, orient="records", lines=True)
    print(f"Wrote {len(project_df)} HotpotQA samples to {out_path}")


if __name__ == "__main__":
    main()
