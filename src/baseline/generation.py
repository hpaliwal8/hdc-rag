from typing import Dict, Any

from src.utils.llm import resolve_llm_model, resolve_llm_provider


def generate_baseline(question: str, config: Dict[str, Any]) -> str:
    mode = config.get("baseline", {}).get("mode", "dummy")
    if mode == "dummy":
        return "I don't know."
    if mode == "custom":
        provider = resolve_llm_provider(config)
        model = resolve_llm_model(config)
        raise NotImplementedError(
            f"Implement your baseline LLM call here. provider={provider} model={model}"
        )
    raise ValueError(f"Unknown baseline mode: {mode}")
