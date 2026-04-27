import re
from typing import Dict, Any


ABSTAIN_PHRASES = [
    "i don't know",
    "i do not know",
    "i'm not sure",
    "i am not sure",
    "insufficient evidence",
    "cannot determine",
    "cannot be determined",
    "not enough information",
    "unclear",
]

NUMBER_PATTERN = re.compile(r"\b\d{1,4}\b")


def _is_abstention(answer: str) -> bool:
    answer_lower = answer.lower()
    return any(phrase in answer_lower for phrase in ABSTAIN_PHRASES)


def _entity_missing_from_evidence(answer: str, evidence: str) -> bool:
    evidence_lower = evidence.lower()
    words = answer.split()
    capitalized = [w.strip(".,;:?!\"'") for w in words if w and w[0].isupper()]
    if not capitalized:
        return False
    missing = [w for w in capitalized if w.lower() not in evidence_lower]
    return len(missing) > len(capitalized) // 2


def _attribute_mismatch(answer: str, evidence: str) -> bool:
    answer_numbers = set(NUMBER_PATTERN.findall(answer))
    evidence_numbers = set(NUMBER_PATTERN.findall(evidence))
    if not answer_numbers:
        return False
    return bool(answer_numbers - evidence_numbers)


def classify_hallucination_type(
    nli_result: Dict[str, Any],
    answer: str,
    evidence: str,
    question_type: str = "",
) -> str:
    if nli_result["nli_label"] == "entailment":
        return "supported"

    if _is_abstention(answer):
        return "abstained"

    if nli_result["nli_label"] == "contradiction":
        return "contradiction_to_evidence"

    if _attribute_mismatch(answer, evidence):
        return "attribute_error"

    if _entity_missing_from_evidence(answer, evidence):
        return "entity_error"

    if question_type in ("bridge", "comparison"):
        return "multi_hop_reasoning_error"

    return "unsupported_inference"
