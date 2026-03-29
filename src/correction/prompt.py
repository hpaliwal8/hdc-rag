from typing import List, Dict, Any


def build_correction_prompt(question: str, baseline: str, passages: List[Dict[str, Any]]) -> str:
    evidence = "\n\n".join([p["text"] for p in passages])
    prompt = (
        "You are given a question, a baseline answer, and evidence passages.\n"
        "Rewrite the answer using ONLY the evidence.\n"
        "If the evidence is insufficient, say you don't know.\n\n"
        f"Question: {question}\n\n"
        f"Baseline Answer: {baseline}\n\n"
        f"Evidence:\n{evidence}\n\n"
        "Corrected Answer:"
    )
    return prompt
