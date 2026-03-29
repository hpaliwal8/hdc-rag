from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional
import random

import pandas as pd
from datasets import load_dataset


@dataclass
class TruthfulQASample:
    sample_id: str
    question: str
    category: Optional[str]
    source: Optional[str]
    best_answer: Optional[str]
    correct_answers: List[str]
    incorrect_answers: List[str]


def _to_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        # Handles semicolon-separated text if needed
        parts = [x.strip() for x in value.split(";")]
        return [x for x in parts if x]
    return [str(value).strip()]


def _normalize_row(row: Dict[str, Any], idx: int) -> TruthfulQASample:
    # Different HF mirrors may expose slightly different column names.
    question = row.get("question") or row.get("Question") or ""
    category = row.get("category") or row.get("Category")
    source = row.get("source") or row.get("Source")

    best_answer = (
        row.get("best_answer")
        or row.get("Best Answer")
        or row.get("best")
        or row.get("answer")
    )

    correct_answers = _to_list(
        row.get("correct_answers")
        or row.get("Correct Answers")
        or row.get("correct")
    )

    incorrect_answers = _to_list(
        row.get("incorrect_answers")
        or row.get("Incorrect Answers")
        or row.get("incorrect")
    )

    sample_id = str(row.get("id") or row.get("Id") or f"truthfulqa_{idx:04d}")

    return TruthfulQASample(
        sample_id=sample_id,
        question=str(question).strip(),
        category=str(category).strip() if category is not None else None,
        source=str(source).strip() if source is not None else None,
        best_answer=str(best_answer).strip() if best_answer is not None else None,
        correct_answers=correct_answers,
        incorrect_answers=incorrect_answers,
    )


def load_truthfulqa_subset(
    subset_size: int = 120,
    seed: int = 42,
    stratify_by_category: bool = True,
    dataset_name: str = "domenicrosati/TruthfulQA",
    split: str = "train",
) -> pd.DataFrame:
    """
    Load a reproducible subset of TruthfulQA.

    Args:
        subset_size: Number of examples to sample.
        seed: Random seed for reproducibility.
        stratify_by_category: Try to preserve category diversity.
        dataset_name: Hugging Face dataset repo.
        split: Preferred split/config. "generation" is common for open-ended QA.

    Returns:
        pd.DataFrame with normalized columns.
    """
    try:
        ds = load_dataset(dataset_name, split=split)
    except ValueError as e:
        # Fallback for datasets that only expose a single split (often "train").
        if 'Unknown split' in str(e) and split != "train":
            ds = load_dataset(dataset_name, split="train")
        else:
            raise
    rows = [dict(x) for x in ds]
    samples = [_normalize_row(row, idx=i) for i, row in enumerate(rows)]

    if subset_size >= len(samples):
        return pd.DataFrame([asdict(s) for s in samples])

    rng = random.Random(seed)

    if not stratify_by_category:
        chosen = rng.sample(samples, subset_size)
        return pd.DataFrame([asdict(s) for s in chosen])

    # Stratified-ish sampling by category
    grouped: Dict[str, List[TruthfulQASample]] = {}
    for sample in samples:
        key = sample.category or "unknown"
        grouped.setdefault(key, []).append(sample)

    categories = list(grouped.keys())
    rng.shuffle(categories)

    chosen: List[TruthfulQASample] = []
    per_cat = max(1, subset_size // max(1, len(categories)))

    # First pass: balanced pick
    for cat in categories:
        group = grouped[cat][:]
        rng.shuffle(group)
        chosen.extend(group[:per_cat])

    # Second pass: fill remaining
    if len(chosen) < subset_size:
        already = {s.sample_id for s in chosen}
        remaining = [s for s in samples if s.sample_id not in already]
        rng.shuffle(remaining)
        chosen.extend(remaining[: subset_size - len(chosen)])

    # Trim in case first pass overshot
    chosen = chosen[:subset_size]

    df = pd.DataFrame([asdict(s) for s in chosen])
    df = df.sort_values("sample_id").reset_index(drop=True)
    return df


def build_project_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert normalized TruthfulQA rows into your shared project schema.
    """
    project_df = pd.DataFrame(
        {
            "id": df["sample_id"],
            "question": df["question"],
            "category": df["category"],
            "reference_answer": df["best_answer"],
            "source": df["source"],
            "baseline_answer": None,
            "retrieved_passages": None,
            "support_score": None,
            "support_label": None,
            "corrected_answer": None,
            "final_label": None,
        }
    )
    return project_df


if __name__ == "__main__":
    df = load_truthfulqa_subset(
        subset_size=120,
        seed=42,
        stratify_by_category=True,
    )

    project_df = build_project_schema(df)

    print(project_df.head(3).to_dict(orient="records"))
    project_df.to_json("truthfulqa_subset_120.jsonl", orient="records", lines=True)
    project_df.to_csv("truthfulqa_subset_120.csv", index=False)

    print(f"Saved {len(project_df)} samples.")
