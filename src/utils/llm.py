import os
from typing import Dict, Any


def resolve_llm_model(config: Dict[str, Any]) -> str:
    llm_cfg = config.get("llm", {})
    use_dev = llm_cfg.get("use_dev_model", False)
    env_override = os.getenv("HDC_RAG_USE_DEV_MODEL")
    if env_override is not None:
        use_dev = env_override.strip() in {"1", "true", "TRUE", "yes", "YES"}
    if use_dev:
        return llm_cfg.get("dev_model", llm_cfg.get("model", ""))
    return llm_cfg.get("model", "")


def resolve_llm_provider(config: Dict[str, Any]) -> str:
    return config.get("llm", {}).get("provider", "")
