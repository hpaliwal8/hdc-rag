from typing import Dict, Any, List

from src.correction.prompt import build_correction_prompt
from src.utils.llm import resolve_llm_model, resolve_llm_provider


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
        provider = resolve_llm_provider(config)
        model = resolve_llm_model(config)
        raise NotImplementedError(
            f"Implement your correction LLM call here. provider={provider} model={model}"
        )
    raise ValueError(f"Unknown correction mode: {mode}")
