from typing import Any, Dict, List, Optional


def build_output_record(
    sample_id: Optional[str],
    question: str,
    reference_answer: Optional[str],
    baseline_answer: str,
    retrieved_passages: List[Dict[str, Any]],
    answer: str,
    corrected_answer: Optional[str] = None,
    support_score: Optional[float] = None,
    support_label: Optional[str] = None,
    category: Optional[str] = None,
    source: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "id": sample_id,
        "question": question,
        "category": category,
        "source": source,
        "reference_answer": reference_answer,
        "answer": answer,
        "baseline_answer": baseline_answer,
        "retrieved_passages": retrieved_passages,
        "passages": retrieved_passages,
        "support_score": support_score,
        "support_label": support_label,
        "corrected_answer": corrected_answer,
    }
