#!/usr/bin/env python
import argparse
import json
import os
import sys
import urllib.request

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config.load import load_config
from src.utils.llm import resolve_llm_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    llm_cfg = config.get("llm", {})
    endpoint = llm_cfg.get("endpoint", "http://localhost:11434")
    model = resolve_llm_model(config)

    url = endpoint.rstrip("/") + "/api/tags"
    try:
        with urllib.request.urlopen(url) as resp:
            tags = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        print(f"Failed to reach Ollama at {endpoint}: {e}")
        sys.exit(1)

    models = [m.get("name") for m in tags.get("models", [])]
    print(f"Ollama reachable at {endpoint}")
    print(f"Available models: {', '.join(models) if models else 'none'}")
    print(f"Configured model: {model}")

    if model and model not in models:
        print("Warning: configured model not found. Pull it with:")
        print(f"  ollama pull {model}")


if __name__ == "__main__":
    main()
