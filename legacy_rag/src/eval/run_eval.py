from typing import List, Dict

from src.eval.metrics import hallucination_rate, hallucination_reduction
from src.utils.io import read_jsonl


def load_rows(path: str) -> List[Dict]:
    return list(read_jsonl(path))


def run_eval(baseline_path: str, corrected_path: str) -> Dict[str, float]:
    baseline_rows = load_rows(baseline_path)
    corrected_rows = load_rows(corrected_path)

    base_rate = hallucination_rate(baseline_rows)
    corr_rate = hallucination_rate(corrected_rows)
    reduction = hallucination_reduction(base_rate, corr_rate)

    return {
        "baseline_hallucination_rate": base_rate,
        "corrected_hallucination_rate": corr_rate,
        "hallucination_reduction": reduction,
    }
