from typing import List


def chunk_text_words(text: str, chunk_words: int, overlap_words: int) -> List[str]:
    words = text.split()
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
