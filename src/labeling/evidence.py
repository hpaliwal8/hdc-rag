from typing import Any, Dict, List, Tuple


def extract_evidence(
    supporting_facts: List[Tuple[str, int]],
    context: Dict[str, Any],
) -> str:
    """
    Extract and concatenate the gold supporting sentences from HotpotQA context.

    supporting_facts: list of (title, sentence_index) pairs
    context: dict with keys 'title' (list of str) and 'sentences' (list of list of str)
    """
    titles = context.get("title", [])
    sentences = context.get("sentences", [])

    title_to_sentences = {
        title: sents for title, sents in zip(titles, sentences)
    }

    evidence_parts = []
    for title, sent_idx in supporting_facts:
        doc_sentences = title_to_sentences.get(title, [])
        if sent_idx < len(doc_sentences):
            evidence_parts.append(doc_sentences[sent_idx].strip())

    return " ".join(evidence_parts)
