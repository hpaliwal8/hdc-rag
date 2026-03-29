from typing import Dict, Any


def generate_baseline(question: str, config: Dict[str, Any]) -> str:
    mode = config.get("baseline", {}).get("mode", "dummy")
    if mode == "dummy":
        return "I don't know."
    if mode == "custom":
        raise NotImplementedError("Implement your baseline LLM call here.")
    raise ValueError(f"Unknown baseline mode: {mode}")
