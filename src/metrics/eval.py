import re
from collections import Counter, defaultdict
from typing import Any, Dict, List


def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    return " ".join(text.split())


def exact_match(prediction: str, reference: str) -> int:
    return int(_normalize(prediction) == _normalize(reference))


def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = _normalize(prediction).split()
    ref_tokens = _normalize(reference).split()

    if not pred_tokens or not ref_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_common = sum(common.values())

    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_summary(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        key = (r.get("model_id", ""), r.get("prompt_id", ""))
        groups[key].append(r)

    rows = []
    for (model_id, prompt_id), group in sorted(groups.items()):
        total = len(group)
        hallucinated = [r for r in group if r.get("is_hallucinated")]
        abstained = [r for r in group if r.get("hallucination_type") == "abstained"]

        em_scores = [
            exact_match(r.get("answer", ""), r.get("reference_answer", ""))
            for r in group
        ]
        f1_scores = [
            token_f1(r.get("answer", ""), r.get("reference_answer", ""))
            for r in group
        ]

        type_counts: Dict[str, int] = Counter(
            r.get("hallucination_type", "unknown") for r in hallucinated
        )

        rows.append({
            "model_id": model_id,
            "prompt_id": prompt_id,
            "total": total,
            "exact_match": round(sum(em_scores) / total, 4),
            "token_f1": round(sum(f1_scores) / total, 4),
            "hallucination_rate": round(len(hallucinated) / total, 4),
            "abstention_rate": round(len(abstained) / total, 4),
            "type_counts": dict(type_counts),
        })

    return rows
