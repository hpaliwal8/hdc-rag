from typing import List, Dict, Any


def build_correction_prompt(question: str, baseline: str, passages: List[Dict[str, Any]]) -> str:
    evidence_blocks = []
    for i, passage in enumerate(passages, start=1):
        meta = passage.get("meta", {})
        title = meta.get("title") or meta.get("doc_id") or "Untitled"
        score = passage.get("score", 0.0)
        evidence_blocks.append(
            f"[Evidence {i}] Title: {title}\n"
            f"Score: {score:.4f}\n"
            f"Text: {passage['text']}"
        )
    evidence = "\n\n".join(evidence_blocks)
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
