# HDC-RAG Boilerplate

A minimal, practical template for a 4-week RAG + hallucination-detection project.

## Quick Start

1. Create a virtualenv and install deps:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Put your corpus files (plain `.txt`) under `data/raw/corpus/`.

3. Build passages + index:

```bash
python scripts/build_corpus.py --config config/default.yaml
python scripts/build_index.py --config config/default.yaml
```

4. Download TruthfulQA (or your dataset) and create a questions file:

```bash
python scripts/prepare_truthfulqa.py --config config/default.yaml
```

5. Run baseline + pipeline:

```bash
python scripts/run_pipeline.py --config config/default.yaml
```

6. Evaluate:

```bash
python scripts/run_eval.py --config config/default.yaml
```

## Project Layout

- `src/data/`: dataset and corpus loaders
- `src/retrieval/`: chunking, embeddings, FAISS index, retriever
- `src/baseline/`: baseline answer generator
- `src/detection/`: support scoring + gating
- `src/correction/`: correction prompt + generator
- `src/eval/`: metrics + evaluation runner
- `scripts/`: CLI entrypoints
- `config/`: YAML config

## Notes

- Baseline + correction are stubs. Fill in your preferred LLM calls.
- Keep the corpus small early (5K–15K passages) to iterate fast.
- Use the retrieval sanity check before moving to detection.
