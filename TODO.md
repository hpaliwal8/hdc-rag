# Research Re-Alignment Notes (2026-04-05)

## Dataset Plan

- Primary dataset: **HotpotQA**
- Secondary stress-test dataset: **TruthfulQA**

**Rationale:** HotpotQA is Wikipedia-based, multi-hop, and includes supporting facts, which makes evidence-backed hallucination typing much easier. TruthfulQA is still valuable, but more as a stress test for misconception-style falsehoods than for a rich hallucination taxonomy.

## Model Plan (4 Models)

- **Phi-4-mini-instruct** (3.8B, 128K context)
- **Mistral-7B-Instruct-v0.3** (7B)
- **Qwen2.5-7B-Instruct** (7B, 32K context)
- **Llama-3.1-8B-Instruct** (8B, 128K context)

**Why these:** Strong enough to be interesting, diverse enough to compare, and realistic on Colab Pro T4/A100 for inference.

## Project Direction

This is no longer a RAG-correction project. It is a **comparative hallucination analysis study**:
- Same QA questions, same evaluation pipeline, four models.
- Analyze what kinds of hallucinations they produce, when they hallucinate, and how prompt wording changes behavior.

## Dataset Sizes (Recommended)

- **HotpotQA:** 400–600 questions (dev split)
- **TruthfulQA:** 100 questions (secondary stress test)

## Hardware Guidance

- **T4 (16 GB):** stay in the 4B–8B range, use 4-bit quantized inference.
- **A100:** more headroom, but 13B+ is not required for this study.
- This recommendation is based on model sizes vs T4 memory limits.

## Research Questions (Locked)

1. What types of hallucinations does each model produce?
2. Under what conditions do hallucinations increase or decrease?
3. Do specific prompt cues reduce hallucination?
4. Which model gives the best tradeoff between accuracy, abstention, and hallucination rate?

## Hallucination Taxonomy

- **Entity error:** wrong person/place/thing
- **Attribute error:** right entity, wrong property/date/role
- **Multi-hop reasoning error:** facts plausible individually, chain is wrong
- **Unsupported inference:** answer goes beyond evidence
- **Contradiction to evidence:** answer conflicts with supporting facts
- **Overconfident abstention failure:** model should abstain but invents an answer

## Taxonomy Classifier (NLI + Heuristics)

- Use **DeBERTa‑MNLI** cross‑encoder as NLI judge.
- Premise: HotpotQA supporting facts (concatenated evidence).
- Hypothesis: model answer.
- NLI label → supported vs hallucinated:
  - Entailment → supported
  - Contradiction → hallucinated (type = contradiction)
  - Neutral → hallucinated (type determined by heuristics below)

**Heuristics for hallucination type (after NLI):**
- Abstention check: if answer says “I don’t know” / “insufficient evidence” → abstained.
- If answer entity not in evidence → entity error / unsupported inference.
- If entity present but attribute mismatch (date/number/location) → attribute error.
- If HotpotQA question is bridge/comparison and evidence has parts but answer neutral → multi‑hop reasoning error.
- Otherwise → unsupported inference.
- If hallucinated and no abstention → overconfident abstention failure flag.

## Prompt Experiment (3 Variants)

- **Plain QA prompt**
- **Abstention prompt:** “If unsure, say ‘I don’t know’.”
- **Reasoning + abstention:** “Use step-by-step reasoning; if evidence is insufficient, say so.”

Key phrases to test: “if unsure, say I don’t know” and “do not guess.”

## Comparative Evaluation (Report)

- Exact Match / token-level F1 (HotpotQA)
- Hallucination rate
- Abstention rate
- Hallucination type distribution
- Prompt sensitivity (hallucination rate shift by prompt variant)
- Per-model confusion table (supported / unsupported / contradictory)

## Execution Plan (High-Level)

- **Environment:** Colab Pro for inference, local VS Code for preprocessing + analysis.
- **Study Matrix:** 4 models × 3 prompts × HotpotQA/TruthfulQA subsets.
- **Outcome:** Comparative analysis + evidence-grounded hallucination taxonomy.
