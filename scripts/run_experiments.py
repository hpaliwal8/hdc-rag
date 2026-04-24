#!/usr/bin/env python
import argparse
import os
import sys
from typing import Dict, Any, List

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config.load import load_config
from src.utils.io import read_jsonl, ensure_dir
from src.utils.logging import setup_logging
from src.utils.ollama import ollama_generate


def build_prompt(system: str, template: str, question: str) -> str:
    return f"{system}\n\n{template.format(question=question)}"


_HF_CACHE = {}


def _load_hf_model(model_name: str, hf_cfg: Dict[str, Any]):
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "HF provider requires transformers + torch. Install in Colab:\n"
            "pip install transformers accelerate\n"
            "pip install torch"
        ) from e

    key = (model_name, tuple(sorted(hf_cfg.items())))
    if key in _HF_CACHE:
        return _HF_CACHE[key]

    device_map = hf_cfg.get("device_map", "auto")
    trust_remote_code = bool(hf_cfg.get("trust_remote_code", False))
    torch_dtype_cfg = hf_cfg.get("torch_dtype", "auto")
    if torch_dtype_cfg == "float16":
        torch_dtype = torch.float16
    elif torch_dtype_cfg == "bfloat16":
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = None

    quant_config = None
    if hf_cfg.get("load_in_4bit") or hf_cfg.get("load_in_8bit"):
        try:
            from transformers import BitsAndBytesConfig
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "Quantized loading requires bitsandbytes. Install:\n"
                "pip install bitsandbytes"
            ) from e
        quant_config = BitsAndBytesConfig(
            load_in_4bit=bool(hf_cfg.get("load_in_4bit")),
            load_in_8bit=bool(hf_cfg.get("load_in_8bit")),
        )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=trust_remote_code
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
        quantization_config=quant_config,
    )

    _HF_CACHE[key] = (tokenizer, model)
    return tokenizer, model


def hf_generate(
    system: str,
    template: str,
    question: str,
    model_name: str,
    hf_cfg: Dict[str, Any],
    options: Dict[str, Any],
) -> str:
    import torch

    tokenizer, model = _load_hf_model(model_name, hf_cfg)
    use_chat = bool(hf_cfg.get("use_chat_template", True))
    user_text = template.format(question=question)

    if use_chat and hasattr(tokenizer, "apply_chat_template"):
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_text},
        ]
        input_text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
    else:
        input_text = f"{system}\n\n{user_text}"

    inputs = tokenizer(input_text, return_tensors="pt")
    if hasattr(model, "device"):
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    max_new_tokens = int(options.get("num_predict", 256))
    temperature = float(options.get("temperature", 0.2))
    top_p = float(options.get("top_p", 0.9))

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature,
            top_p=top_p,
        )
    text = tokenizer.decode(output[0], skip_special_tokens=True)
    return text.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dataset", default=None, help="Optional dataset id to run.")
    parser.add_argument("--output", default=None, help="Override output path.")
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)

    exp_cfg = config.get("experiments", {})
    datasets = exp_cfg.get("datasets", [])
    models = exp_cfg.get("models", [])
    prompts = exp_cfg.get("prompts", [])

    if args.dataset:
        datasets = [d for d in datasets if d.get("id") == args.dataset]
        if not datasets:
            raise ValueError(f"Unknown dataset id: {args.dataset}")

    outputs_dir = config["paths"]["outputs_dir"]
    ensure_dir(outputs_dir)
    out_path = args.output or os.path.join(outputs_dir, "experiments.jsonl")

    llm_cfg = config.get("llm", {})
    endpoint = llm_cfg.get("endpoint", "http://localhost:11434")
    options = {
        "temperature": llm_cfg.get("temperature", 0.2),
        "top_p": llm_cfg.get("top_p", 0.9),
        "num_predict": llm_cfg.get("max_tokens", 256),
    }
    hf_cfg = exp_cfg.get("hf", {})

    with open(out_path, "w", encoding="utf-8") as f:
        for d in datasets:
            dataset_id = d.get("id")
            dataset_path = d.get("path")
            if not dataset_path:
                continue

            rows = list(read_jsonl(dataset_path))
            if args.limit and args.limit > 0:
                rows = rows[: args.limit]

            for model in models:
                model_id = model.get("id")
                provider = model.get("provider", "ollama")
                model_name = model.get("model")

                for prompt in prompts:
                    prompt_id = prompt.get("id")
                    system = prompt.get("system", "")
                    template = prompt.get("template", "{question}")

                    for row in rows:
                        question = row.get("question", "")
                        if provider == "ollama":
                            prompt_text = build_prompt(system, template, question)
                            answer = ollama_generate(
                                prompt_text,
                                model=model_name,
                                endpoint=endpoint,
                                options=options,
                            )
                        elif provider == "hf":
                            answer = hf_generate(
                                system,
                                template,
                                question,
                                model_name=model_name,
                                hf_cfg=hf_cfg,
                                options=options,
                            )
                        else:
                            raise NotImplementedError(
                                f"Provider not supported yet: {provider}"
                            )

                        record: Dict[str, Any] = {
                            "id": row.get("id"),
                            "dataset": dataset_id,
                            "question": question,
                            "reference_answer": row.get("reference_answer"),
                            "model_id": model_id,
                            "prompt_id": prompt_id,
                            "answer": answer,
                        }

                        f.write(json_dumps(record) + "\n")
                        f.flush()

    print(f"Wrote experiment outputs to {out_path}")


def json_dumps(obj: Dict[str, Any]) -> str:
    import json

    return json.dumps(obj, ensure_ascii=True)


if __name__ == "__main__":
    main()
