from typing import List, Dict, Any

from datasets import load_dataset


def load_truthfulqa(split: str = "validation") -> List[Dict[str, Any]]:
    dataset = load_dataset("truthful_qa", "generation", split=split)
    rows: List[Dict[str, Any]] = []
    for row in dataset:
        rows.append(
            {
                "id": row.get("id"),
                "question": row.get("question"),
                "reference_answer": row.get("best_answer"),
            }
        )
    return rows
