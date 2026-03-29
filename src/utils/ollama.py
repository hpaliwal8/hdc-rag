import json
import urllib.request
from typing import Dict, Any, Optional


def ollama_generate(
    prompt: str,
    model: str,
    endpoint: str,
    options: Optional[Dict[str, Any]] = None,
) -> str:
    url = endpoint.rstrip("/") + "/api/generate"
    payload: Dict[str, Any] = {"model": model, "prompt": prompt, "stream": False}
    if options:
        payload["options"] = options

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        raise RuntimeError(f"Ollama request failed: {e}") from e

    response = data.get("response")
    if response is None:
        raise RuntimeError(f"Ollama response missing 'response': {data}")
    return str(response).strip()
