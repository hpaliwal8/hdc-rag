# Comparative Hallucination Analysis Study

## Objective

Design and execute a comparative study of hallucinations across four LLMs on two QA datasets (HotpotQA primary, TruthfulQA secondary). The study will classify hallucination types, identify prompt‑sensitivity patterns, and report cross‑model comparisons.

## Research Questions

1. What types of hallucinations does each model produce?
2. Under what conditions do hallucinations increase or decrease?
3. Do specific prompt cues reduce hallucination?
4. Which model gives the best tradeoff between accuracy, abstention, and hallucination rate?

## Datasets

- **Primary:** HotpotQA (dev split, 400–600 questions)
- **Secondary:** TruthfulQA (100 questions, stress test)

## Models (4)

- Phi-4-mini-instruct (3.8B)
- Mistral-7B-Instruct-v0.3 (7B)
- Qwen2.5-7B-Instruct (7B)
- Llama-3.1-8B-Instruct (8B)

## Hallucination Taxonomy

### `supported`
The model's answer is entailed by the evidence. Not a hallucination.

### `abstained`
The model correctly declined to answer (e.g. "I don't know", "I'm not sure", "insufficient evidence"). Not a hallucination — this is the desired behavior under uncertainty.

### `contradiction_to_evidence`
The model's answer directly conflicts with the supporting facts. Detected by NLI (contradiction label). Example: evidence says "founded in 1755", model says "founded in 1800".

### `entity_error`
The model names a person, place, or thing that does not appear in the evidence at all. Detected by checking if named entities in the answer are present in the evidence string.

### `attribute_error`
The correct entity is mentioned but a property is wrong — typically a date, number, or location. Detected by finding entities that appear in both answer and evidence, but with mismatched numeric/date values.

### `multi_hop_reasoning_error`
The model gets individual facts right but chains them incorrectly across hops. Specific to HotpotQA `bridge` and `comparison` questions where multiple evidence passages must be combined. Detected when question type is multi-hop and evidence is partially matched but NLI is neutral.

### `unsupported_inference`
Catch-all for neutral NLI cases where no specific heuristic fires. The answer goes beyond the evidence without directly contradicting it. If this label dominates (>40% of hallucinations), heuristics need improvement via manual spot-checking.

### `overconfident_abstention_failure`
Cross-cutting flag: the model hallucinated an answer when it should have abstained. Applied on top of any hallucination type when the model gives a confident wrong answer on a question where abstention would have been appropriate.

## Labeling Methodology (NLI + Heuristics)

Use a **DeBERTa‑MNLI** cross‑encoder as an NLI judge.

- **Premise:** HotpotQA supporting facts (concatenated evidence).
- **Hypothesis:** model answer.

**Decision flow:**
- Entailment → supported (not hallucinated).
- Contradiction → hallucinated (type = contradiction).
- Neutral → hallucinated (type assigned by heuristics below).

**Heuristic typing (for neutral/unsupported):**
- Abstention check: if answer says “I don’t know” / “insufficient evidence” → abstained.
- If answer entity not in evidence → entity error / unsupported inference.
- If entity present but attribute mismatch (date/number/location) → attribute error.
- If HotpotQA question is bridge/comparison and evidence has parts but answer neutral → multi‑hop reasoning error.
- Otherwise → unsupported inference.
- If hallucinated and no abstention → overconfident abstention failure flag.

## Prompt Variants (3)

1. Plain QA prompt  
2. Abstention prompt: “If unsure, say ‘I don’t know’.”  
3. Reasoning + abstention: “Use step-by-step reasoning; if evidence is insufficient, say so.”  

## Evaluation Metrics

- Exact Match / token-level F1 (HotpotQA)
- Hallucination rate
- Abstention rate
- Hallucination type distribution
- Prompt sensitivity (hallucination shift by prompt variant)
- Per-model confusion table (supported / unsupported / contradictory)

## Environment

- **Inference:** Google Colab Pro (T4 preferred, A100 optional)
- **Local:** VS Code for preprocessing, analysis, and plotting

## Outputs

- Per-model result tables
- Hallucination type distribution charts
- Prompt‑sensitivity comparisons
- Error analysis examples
