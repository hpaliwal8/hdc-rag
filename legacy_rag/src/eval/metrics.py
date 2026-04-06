from typing import List, Dict


def hallucination_rate(rows: List[Dict]) -> float:
    if not rows:
        return 0.0
    unsupported = sum(1 for r in rows if r.get("support_label") == "unsupported")
    return unsupported / len(rows)


def hallucination_reduction(baseline_rate: float, corrected_rate: float) -> float:
    if baseline_rate == 0:
        return 0.0
    return (baseline_rate - corrected_rate) / baseline_rate
