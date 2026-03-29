from typing import Dict, Any, List

from src.correction.prompt import build_correction_prompt


def correct_answer(
    question: str,
    baseline: str,
    passages: List[Dict[str, Any]],
    config: Dict[str, Any],
) -> str:
    mode = config.get("correction", {}).get("mode", "dummy")
    if mode == "dummy":
        return "I don't know."
    if mode == "custom":
        prompt = build_correction_prompt(question, baseline, passages)
        raise NotImplementedError("Implement your correction LLM call here. Prompt built.")
    raise ValueError(f"Unknown correction mode: {mode}")
