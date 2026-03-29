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

Optional flags include `--subset_size`, `--dataset_name`, `--split`, and `--no_stratify`.

5. Run baseline + pipeline:

```bash
python scripts/run_pipeline.py --config config/default.yaml
```

To run baseline only (skip retrieval/correction):

```bash
python scripts/run_pipeline.py --config config/default.yaml --skip_retrieval --limit 10
```

6. Evaluate:

```bash
python scripts/run_eval.py --config config/default.yaml
```

## Local LLM Health Check

Verify Ollama is running and the configured model is available:

```bash
python scripts/ollama_check.py --config config/default.yaml
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
- LLM model selection is centralized in `config/default.yaml` under `llm`.
- To switch to the dev model without editing config, set `HDC_RAG_USE_DEV_MODEL=1`.
- For local inference, this project supports Ollama via `llm.provider: ollama`.

## Day-Wise To-Do (From Plan)

- [x] Day 0 — Repo boilerplate scaffolded (modules + scripts in place).

- [x] Day 1 (Person A) — Finalize corpus choice (Wikipedia vs dataset evidence).
- [ ] Day 1 (Person A) — Decide chunking (150–300 words, 30-word overlap).
- [x] Day 1 (Person A) — Setup retrieval module (scaffolded).
- [x] Day 1 (Person B) — Finalize dataset (TruthfulQA).
- [x] Day 1 (Person B) — Define hallucination rule (unsupported OR contradicted).
- [x] Day 1 (Person B) — Setup baseline + evaluation modules (scaffolded).
- [ ] Day 1 (Shared) — Finalize schema, baseline model, pilot size (20–30).
- [ ] Day 1 (Shared) — Lock main metric: Hallucination Reduction (%).

- [ ] Day 2 (Person A) — Load corpus.
- [ ] Day 2 (Person A) — Clean + chunk text.
- [x] Day 2 (Person B) — Load dataset.
- [x] Day 2 (Person B) — Normalize fields (id, question, reference_answer).
- [x] Day 2 (Person B) — Write baseline prompt.

- [ ] Day 3 (Person A) — Choose embedding model (bge-small / e5-base).
- [ ] Day 3 (Person A) — Embed small subset.
- [ ] Day 3 (Person A) — Build FAISS index.
- [x] Day 3 (Person B) — Implement `generate_baseline(question)`.
- [x] Day 3 (Person B) — Run baseline on 5–10 samples.
- [ ] Day 3 (Shared) — Inspect baseline answers.

- [ ] Day 4 (Person A) — Implement `retrieve(question, k=5)`.
- [ ] Day 4 (Person A) — Run retrieval on 10–15 questions.
- [ ] Day 4 (Person A) — Retrieval sanity check (top-3 relevant?).
- [ ] Day 4 (Person B) — Logging pipeline.
- [ ] Day 4 (Person B) — Store outputs in schema.

- [ ] Day 5 (Person A) — Finalize retriever.
- [x] Day 5 (Person B) — Implement support scoring (max cosine similarity).
- [x] Day 5 (Person B) — Define thresholds (>=0.75 supported, 0.5–0.75 uncertain, <0.5 unsupported).
- [ ] Day 5 (Shared) — Validate on 5–10 examples.

- [ ] Day 6 (Person A) — Design correction prompt.
- [x] Day 6 (Person B) — Implement gating logic.

- [ ] Day 7 (Person A) — Implement `correct_answer(question, baseline, evidence)`.
- [ ] Day 7 (Person B) — Run detection on same samples.
- [ ] Day 7 (Shared) — Compare baseline vs corrected grounding.

- [ ] Day 8 (Person A) — Refine correction prompt.
- [x] Day 8 (Person B) — Build evaluation script (support rate, hallucination rate).

- [ ] Day 9 (Person A) — Run correction pipeline (20–30 samples).
- [ ] Day 9 (Person B) — Compute baseline and corrected hallucination rates.
- [ ] Day 9 (Shared) — Inspect full pipeline table.

- [ ] Day 10 (Person A) — Lock chunk size, k value, embedding model.
- [ ] Day 10 (Person B) — Lock thresholds, scoring method.

- [ ] Day 11 (Person A) — Run correction on full set (100–200).
- [ ] Day 11 (Person B) — Ensure baseline outputs complete.

- [ ] Day 12 (Person A) — Ablation: top-3 vs top-5 retrieval.
- [ ] Day 12 (Person B) — Compute baseline stats.

- [ ] Day 13 (Person A) — Generate final corrected answers.
- [ ] Day 13 (Person B) — Compute Hallucination Reduction (%).

- [ ] Day 14 (Person A) — Analyze failures (retrieval vs prompt).
- [ ] Day 14 (Person B) — Compute Precision / Recall / F1 (if labels exist).

- [ ] Day 15 (Person A) — Finalize configs.
- [ ] Day 15 (Person B) — Prepare tables, plots, examples.

- [ ] Day 16 (Person A) — Write retrieval + correction methods.
- [ ] Day 16 (Person B) — Write baseline + detection + metrics methods.

- [ ] Day 17 (Person A) — Prepare qualitative cases.
- [ ] Day 17 (Person B) — Write results section.

- [ ] Day 18 (Person A + B) — Error analysis table (type, example, cause).

- [ ] Day 19 (Shared) — Slides + demo (2–3 strong examples).

- [ ] Day 20 (Shared) — Final polish (clean code, verify demo, final report).

## Finalized Architecture

```bash
[TruthfulQA Question]
        ↓
Baseline LLM Answer
        ↓
[Wikipedia Retrieval]
        ↓
Top-k Evidence Passages
        ↓
Support Scoring
        ↓
Correction (if needed)
        ↓
Final Answer
```

## Flow Of Operation

1. Prepare TruthfulQA subset into `data/processed/questions.jsonl`.
2. Place Wikipedia `.txt` articles in `data/raw/corpus/`.
3. Build passages (`scripts/build_corpus.py`) and FAISS index (`scripts/build_index.py`).
4. Run baseline-only or full pipeline:
   - Baseline-only: `scripts/run_pipeline.py --skip_retrieval`
   - Full: `scripts/run_pipeline.py`
5. Inspect outputs in `data/outputs/` (use `scripts/preview_outputs.py`).
6. Evaluate hallucination reduction with `scripts/run_eval.py`.
## Next Steps

1. Decide corpus source and chunking settings, then run `scripts/build_corpus.py` and `scripts/build_index.py`.
2. Generate the TruthfulQA subset with `scripts/prepare_truthfulqa.py` and confirm the schema output.
3. Implement baseline and correction LLM calls, then run `scripts/retrieval_sanity_check.py` before full pipeline runs.

## Knowledge Graph
![Knowledge Graph](https://github.com/hpaliwal8/hdc-rag/blob/main/knowledge-graph.png "KG")
