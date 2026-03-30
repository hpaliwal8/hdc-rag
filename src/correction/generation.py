from typing import Dict, Any, List

from src.correction.prompt import build_correction_prompt
from src.utils.llm import resolve_llm_model, resolve_llm_provider
from src.utils.ollama import ollama_generate


def correct_answer(
    question: str,
    baseline: str,
    passages: List[Dict[str, Any]],
    config: Dict[str, Any],
) -> str:
    mode = config.get("correction", {}).get("mode", "dummy")
    if mode == "dummy":
        if passages:
            top_meta = passages[0].get("meta", {})
            title = top_meta.get("title") or top_meta.get("doc_id")
            if question.lower().startswith("who") and title:
                return f"According to the retrieved evidence, {title} is the most relevant answer."
        return "I don't know."
    if mode == "custom":
        prompt = build_correction_prompt(question, baseline, passages)
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
                "correction_system",
                "You are a grounded assistant. Use only the provided evidence.",
            )
            style = prompt_cfg.get(
                "correction_style",
                "Rewrite the answer using ONLY the evidence. If insufficient, say you don't know.",
            )
            wrapped_prompt = f"{system}\n\n{style}\n\n{prompt}"
            return ollama_generate(
                wrapped_prompt, model=model, endpoint=endpoint, options=options
            )
        raise NotImplementedError(
            f"Implement your correction LLM call here. provider={provider} model={model}"
        )
    raise ValueError(f"Unknown correction mode: {mode}")
