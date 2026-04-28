#!/usr/bin/env python
import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from src.utils.io import read_jsonl, ensure_dir
from src.utils.logging import setup_logging


sns.set_theme(style="whitegrid", context="notebook")
PROMPT_ORDER = ["plain", "abstain", "reasoning"]
MODEL_ORDER = [
    "phi-4-mini-instruct",
    "mistral-7b-instruct",
    "qwen2.5-7b-instruct",
    "llama-3.1-8b-instruct",
]


def load_records(paths: list) -> pd.DataFrame:
    rows = []
    for p in paths:
        for r in read_jsonl(p):
            rows.append(r)
    return pd.DataFrame(rows)


def plot_hallucination_rate(df: pd.DataFrame, out_dir: str, dataset: str) -> None:
    rate = (
        df.groupby(["model_id", "prompt_id"])["is_hallucinated"]
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=rate,
        x="model_id",
        y="is_hallucinated",
        hue="prompt_id",
        hue_order=PROMPT_ORDER,
        order=MODEL_ORDER,
        ax=ax,
    )
    ax.set_title(f"Hallucination Rate by Model × Prompt ({dataset})")
    ax.set_ylabel("Hallucination Rate")
    ax.set_xlabel("Model")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=20)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"hallucination_rate_{dataset}.png"), dpi=150)
    plt.close()


def plot_type_distribution(df: pd.DataFrame, out_dir: str, dataset: str) -> None:
    halls = df[df["is_hallucinated"]].copy()
    if halls.empty:
        return

    counts = (
        halls.groupby(["model_id", "hallucination_type"])
        .size()
        .reset_index(name="count")
    )
    pivot = counts.pivot(
        index="model_id", columns="hallucination_type", values="count"
    ).fillna(0)
    pivot = pivot.reindex(MODEL_ORDER)

    fig, ax = plt.subplots(figsize=(11, 6))
    pivot.plot(kind="bar", stacked=True, ax=ax, colormap="tab20")
    ax.set_title(f"Hallucination Type Distribution per Model ({dataset})")
    ax.set_ylabel("Count")
    ax.set_xlabel("Model")
    ax.tick_params(axis="x", rotation=20)
    ax.legend(title="Type", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"type_distribution_{dataset}.png"), dpi=150)
    plt.close()


def plot_prompt_sensitivity(df: pd.DataFrame, out_dir: str, dataset: str) -> None:
    rate = (
        df.groupby(["model_id", "prompt_id"])["is_hallucinated"]
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(
        data=rate,
        x="prompt_id",
        y="is_hallucinated",
        hue="model_id",
        hue_order=MODEL_ORDER,
        style="model_id",
        markers=True,
        dashes=False,
        markersize=10,
        ax=ax,
    )
    ax.set_title(f"Prompt Sensitivity per Model ({dataset})")
    ax.set_ylabel("Hallucination Rate")
    ax.set_xlabel("Prompt Variant")
    ax.set_ylim(0, 1)
    ax.set_xticks(range(len(PROMPT_ORDER)))
    ax.set_xticklabels(PROMPT_ORDER)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"prompt_sensitivity_{dataset}.png"), dpi=150)
    plt.close()


def plot_difficulty_breakdown(df: pd.DataFrame, out_dir: str) -> None:
    if "level" not in df.columns:
        return
    df = df[df["level"].notna() & (df["level"] != "")]
    if df.empty:
        return

    rate = (
        df.groupby(["model_id", "level"])["is_hallucinated"]
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=rate,
        x="model_id",
        y="is_hallucinated",
        hue="level",
        hue_order=["easy", "medium", "hard"],
        order=MODEL_ORDER,
        ax=ax,
    )
    ax.set_title("Hallucination Rate by Difficulty Level (HotpotQA)")
    ax.set_ylabel("Hallucination Rate")
    ax.set_xlabel("Model")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=20)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "difficulty_breakdown_hotpotqa.png"), dpi=150)
    plt.close()


def plot_cross_dataset(hotpotqa_df: pd.DataFrame, truthfulqa_df: pd.DataFrame, out_dir: str) -> None:
    h = (
        hotpotqa_df.groupby("model_id")["is_hallucinated"]
        .mean()
        .reset_index()
        .assign(dataset="HotpotQA")
    )
    t = (
        truthfulqa_df.groupby("model_id")["is_hallucinated"]
        .mean()
        .reset_index()
        .assign(dataset="TruthfulQA")
    )
    combined = pd.concat([h, t], ignore_index=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=combined,
        x="model_id",
        y="is_hallucinated",
        hue="dataset",
        order=MODEL_ORDER,
        ax=ax,
    )
    ax.set_title("Cross-Dataset Hallucination Rate")
    ax.set_ylabel("Hallucination Rate")
    ax.set_xlabel("Model")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=20)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "cross_dataset_hallucination.png"), dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hotpotqa_inputs", nargs="+", default=[])
    parser.add_argument("--truthfulqa_inputs", nargs="+", default=[])
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    setup_logging()
    ensure_dir(args.out_dir)

    hotpotqa_df = load_records(args.hotpotqa_inputs) if args.hotpotqa_inputs else pd.DataFrame()
    truthfulqa_df = load_records(args.truthfulqa_inputs) if args.truthfulqa_inputs else pd.DataFrame()

    if not hotpotqa_df.empty:
        plot_hallucination_rate(hotpotqa_df, args.out_dir, "hotpotqa")
        plot_type_distribution(hotpotqa_df, args.out_dir, "hotpotqa")
        plot_prompt_sensitivity(hotpotqa_df, args.out_dir, "hotpotqa")
        plot_difficulty_breakdown(hotpotqa_df, args.out_dir)

    if not truthfulqa_df.empty:
        plot_hallucination_rate(truthfulqa_df, args.out_dir, "truthfulqa")
        plot_type_distribution(truthfulqa_df, args.out_dir, "truthfulqa")
        plot_prompt_sensitivity(truthfulqa_df, args.out_dir, "truthfulqa")

    if not hotpotqa_df.empty and not truthfulqa_df.empty:
        plot_cross_dataset(hotpotqa_df, truthfulqa_df, args.out_dir)

    print(f"Plots written to {args.out_dir}")


if __name__ == "__main__":
    main()
