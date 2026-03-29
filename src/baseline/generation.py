from typing import Dict, Any

from src.utils.llm import resolve_llm_model, resolve_llm_provider
from src.utils.ollama import ollama_generate


def generate_baseline(question: str, config: Dict[str, Any]) -> str:
    mode = config.get("baseline", {}).get("mode", "dummy")
    if mode == "dummy":
        return "I don't know."
    if mode == "custom":
        provider = resolve_llm_provider(config)
        model = resolve_llm_model(config)
        if provider == "ollama":
            llm_cfg = config.get("llm", {})
            endpoint = llm_cfg.get("endpoint", "http://localhost:11434")
            options = {
                "temperature": llm_cfg.get("temperature", 0.2),
                "top_p": llm_cfg.get("top_p", 0.9),
                "num_predict": llm_cfg.get("max_tokens", 256),
            }
            prompt_cfg = llm_cfg.get("prompt", {})
            system = prompt_cfg.get(
                "baseline_system",
                "You are a precise, factual assistant. Answer concisely and do not speculate.",
            )
            style = prompt_cfg.get(
                "baseline_style",
                "Answer the question concisely and factually. If you are unsure, say you don't know.",
            )
            prompt = (
                f"{system}\n\n"
                f"{style}\n\n"
                f"Question: {question}\nAnswer:"
            )
            return ollama_generate(prompt, model=model, endpoint=endpoint, options=options)
        raise NotImplementedError(
            f"Implement your baseline LLM call here. provider={provider} model={model}"
        )
    raise ValueError(f"Unknown baseline mode: {mode}")
