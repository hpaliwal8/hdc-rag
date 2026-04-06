#!/usr/bin/env python
import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config.load import load_config
from src.eval.run_eval import run_eval


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    outputs_dir = config["paths"]["outputs_dir"]

    baseline_path = f"{outputs_dir}/baseline.jsonl"
    corrected_path = f"{outputs_dir}/corrected.jsonl"

    results = run_eval(baseline_path, corrected_path)
    for k, v in results.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()
