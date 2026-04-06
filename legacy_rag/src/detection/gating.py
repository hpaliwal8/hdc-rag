from typing import Dict, Any


def should_correct(score: float, config: Dict[str, Any]) -> bool:
    threshold = config.get("support", {}).get("supported_threshold", 0.75)
    return score < threshold
