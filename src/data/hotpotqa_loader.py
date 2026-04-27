from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Tuple
import random

import pandas as pd
from datasets import load_dataset


@dataclass
class HotpotQASample:
    sample_id: str
    question: str
    answer: str
    supporting_facts: List[Tuple[str, int]]
    context: Any
    level: str


def _normalize_supporting_facts(raw: Any) -> List[Tuple[str, int]]:
    if not raw:
        return []

    if isinstance(raw, dict):
        titles = raw.get("title") or raw.get("titles") or []
        sent_ids = raw.get("sent_id") or raw.get("sent_ids") or []
        pairs: List[Tuple[str, int]] = []
        for title, sent_id in zip(titles, sent_ids):
            try:
                pairs.append((str(title), int(sent_id)))
            except Exception:
                continue
        return pairs

    pairs: List[Tuple[str, int]] = []
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                title = item.get("title") or item.get("doc_title") or ""
                sent_id = item.get("sent_id")
                if title and sent_id is not None:
                    try:
                        pairs.append((str(title), int(sent_id)))
                    except Exception:
                        continue
                continue
            if isinstance(item, (list, tuple)):
                if len(item) >= 2:
                    try:
                        pairs.append((str(item[0]), int(item[1])))
                    except Exception:
                        continue
                continue
    return pairs


def _normalize_row(row: Dict[str, Any]) -> HotpotQASample:
    sample_id = str(row.get("id") or row.get("_id") or row.get("qid"))
    question = str(row.get("question", "")).strip()
    answer = str(row.get("answer", "")).strip()
    supporting_facts = _normalize_supporting_facts(row.get("supporting_facts"))
    context = row.get("context")
    level = str(row.get("level", "")).strip()

    return HotpotQASample(
        sample_id=sample_id,
        question=question,
        answer=answer,
        supporting_facts=supporting_facts,
        context=context,
        level=level,
    )


def load_hotpotqa_subset(
    subset_size: int = 500,
    seed: int = 42,
    dataset_name: str = "hotpot_qa",
    config_name: str = "distractor",
    split: str = "validation",
) -> pd.DataFrame:
    """
    Load a reproducible subset of HotpotQA with evidence extracted from supporting facts.
    """
    try:
        ds = load_dataset(dataset_name, config_name, split=split)
    except ValueError as e:
        if 'Unknown split' in str(e) and split != "train":
            ds = load_dataset(dataset_name, config_name, split="train")
        else:
            raise

    rows = [dict(x) for x in ds]
    samples = [_normalize_row(row) for row in rows]

    if subset_size >= len(samples):
        return pd.DataFrame([asdict(s) for s in samples])

    rng = random.Random(seed)
    chosen = rng.sample(samples, subset_size)
    return pd.DataFrame([asdict(s) for s in chosen])


def build_project_schema(df: pd.DataFrame) -> pd.DataFrame:
    project_df = pd.DataFrame(
        {
            "id": df["sample_id"],
            "dataset": "hotpotqa",
            "question": df["question"],
            "reference_answer": df["answer"],
            "supporting_facts": df["supporting_facts"],
            "context": df["context"] if "context" in df.columns else None,
            "level": df["level"] if "level" in df.columns else None,
        }
    )
    return project_df


if __name__ == "__main__":
    df = load_hotpotqa_subset(subset_size=50, seed=42)
    project_df = build_project_schema(df)
    print(project_df.head(3).to_dict(orient="records"))
