# HDC-RAG (Research Pivot)

Comparative hallucination analysis across multiple models using HotpotQA (primary) and TruthfulQA (stress test).
See `PROJECT_PLAN.md`, `WEEKLY_PLAN.md`, and `TODO.md` for the research design.

## Quick Start

1. Create a virtualenv and install deps:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Download TruthfulQA (stress-test set) and create a questions file:

```bash
python scripts/prepare_truthfulqa.py --config config/default.yaml
```

Optional flags include `--subset_size`, `--dataset_name`, `--split`, and `--no_stratify`.

## Local LLM Health Check

Verify Ollama is running and the configured model is available:

```bash
python scripts/ollama_check.py --config config/default.yaml
```

## Project Layout

- `src/data/`: dataset loaders
- `src/baseline/`: baseline answer generator
- `src/utils/`: LLM utilities, IO helpers
- `scripts/`: CLI entrypoints
- `config/`: YAML config

## Notes

- Baseline generation is wired for Ollama (`llm.provider: ollama`).
- LLM model selection is centralized in `config/default.yaml` under `llm`.
- To switch to the dev model without editing config, set `HDC_RAG_USE_DEV_MODEL=1`.
- For local inference, this project supports Ollama via `llm.provider: ollama`.

## Next Steps

- Add HotpotQA loader + sampler.
- Add multi-model + prompt-variant experiment runner.
- Add NLI-based labeling and analysis scripts.

Legacy RAG code is archived under `legacy_rag/`.
