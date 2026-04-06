from typing import List


def clean_text(text: str) -> str:
    return text.replace("\n", " ").strip()


def _normalize_document(doc):
    if isinstance(doc, str):
        return {"title": "", "text": clean_text(doc), "source": "unknown"}
    return {
        "title": clean_text(doc.get("title", "")),
        "text": clean_text(doc.get("text", "")),
        "source": doc.get("source", "unknown"),
    }


def chunk_text_words(text: str, chunk_words: int, overlap_words: int) -> List[str]:
    words = clean_text(text).split()
    if not words:
        return []
    chunks: List[str] = []
    step = max(1, chunk_words - overlap_words)
    for start in range(0, len(words), step):
        end = start + chunk_words
        chunk = words[start:end]
        if not chunk:
            break
        chunks.append(" ".join(chunk))
        if end >= len(words):
            break
    return chunks


def chunk_documents(documents, chunk_words: int, overlap_words: int):
    chunks = []
    for i, raw_doc in enumerate(documents):
        doc = _normalize_document(raw_doc)
        text = doc["text"]
        if not text:
            continue
        text_for_chunking = text
        if doc["title"]:
            text_for_chunking = f"Title: {doc['title']}\n\n{text}"
        split_chunks = chunk_text_words(text_for_chunking, chunk_words, overlap_words)
        for chunk_idx, chunk in enumerate(split_chunks):
            chunks.append(
                {
                    "text": chunk,
                    "doc_id": i,
                    "chunk_id": chunk_idx,
                    "char_len": len(chunk),
                    "word_len": len(chunk.split()),
                    "title": doc["title"],
                    "source": doc.get("source", "unknown"),
                }
            )
    return chunks
