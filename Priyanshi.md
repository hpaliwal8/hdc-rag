Option 1 — Difficulty × Hallucination

HotpotQA labels every question as easy, medium, or hard. You now have that level field in your labeled outputs.

The premise is simple: do models hallucinate more on harder questions?

You'd compute hallucination rate broken down by difficulty level, for each model. The interesting question isn't just "yes harder = more hallucinations" — that's obvious. The interesting question is how much more, and whether it's consistent across models.

For example, imagine you found:

Phi-4-mini: easy=60% hall, hard=95% hall — huge gap
Llama: easy=80% hall, hard=88% hall — small gap
That would mean Phi-4-mini is much more sensitive to question difficulty than Llama. That's a meaningful finding about model robustness that you can't see from aggregate rates alone.

Option 2 — Prompt × Model Interaction

You already have hallucination rates per model × prompt. The premise here is to go beyond reporting the numbers and make a claim about why models respond differently to prompts.

Your data already shows something striking:

Qwen: reasoning prompt reduces hallucination by 17.6pp
Llama: reasoning prompt increases hallucination by 4.2pp
The research question is: why does the same prompt instruction help one model and hurt another? This points to fundamental differences in how models follow instructions — Qwen internalizes "reason step by step" and uses it to slow down and chain evidence. Llama becomes more verbose and introduces more errors.

This is more than a metric — it's a behavioral observation about model personalities under instruction.

Both are computable from data you already have. No new inference needed.


The most research-valuable feature you could add is a hallucination early warning signal — a lightweight classifier that predicts, from the model's output alone (no gold evidence needed), whether a given answer is likely hallucinated.

The premise:

Right now your pipeline requires gold evidence (HotpotQA supporting facts) to label hallucinations via NLI. In the real world you don't have gold evidence — you just have a model's answer. The research question becomes:

Can you detect hallucinations without knowing the ground truth?

How you'd build it:

Use your labeled outputs as training signal. You have ~6000 records labeled as supported or hallucinated. Extract features from the model answer alone:

Answer length
Hedging language presence ("I think", "probably", "around")
Named entity count
NLI confidence score (you already have this)
Sentence count
Presence of numbers/dates
Train a simple logistic regression or decision tree classifier on these features to predict is_hallucinated. Then evaluate how well it generalizes across models and datasets.

Why it's valuable:

Generalizes beyond HotpotQA — works on any QA dataset without supporting facts
Directly applicable in production systems
Connects to a live research problem (hallucination detection without references)
Extends naturally to a paper: "Lightweight hallucination detection from model outputs alone"
The bigger vision:

This is the seed of a retrieval-free hallucination detector. If it works even modestly (AUC > 0.70), that's a publishable finding. If it doesn't work well, that's also a finding — it means surface features alone aren't sufficient and you need evidence grounding.

Either outcome is defensible in a research paper.

However, if you wanted to train anything more sophisticated — a fine-tuned BERT classifier for example — 6000 rows is borderline. You'd want at least 10,000-20,000 for a neural classifier to generalize reliably, especially across models and datasets.

The bigger concern isn't quantity, it's balance.

Check your class distribution:


hallucinated = sum(1 for r in records if r["is_hallucinated"])
supported = len(records) - hallucinated
print(f"Hallucinated: {hallucinated}, Supported: {supported}")
From the metrics we already saw, hallucination rates are 86-93% on plain prompts. That means your dataset is heavily imbalanced — maybe 5000 hallucinated vs 1000 supported. A naive classifier would just predict "hallucinated" for everything and get 88% accuracy without learning anything.

You'd need to either:

Undersample the hallucinated class
Use only plain prompt records and balance manually
Use class weights in the classifier
So the honest answer is: 6000 rows is enough for a simple classifier, but only if you handle the class imbalance first.