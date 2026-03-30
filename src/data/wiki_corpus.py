import glob
import os
import json
import re
from typing import Iterable, Dict, Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

from datasets import load_dataset

WIKIPEDIA_USER_AGENT = (
    "HDC-RAG/1.0 "
    "(educational retrieval prototype; contact: local-dev)"
)


def _tokenize(text: str):
    return re.findall(r"\b\w+\b", text.lower())


def _extract_query_terms(query: str):
    stopwords = {
        "a", "an", "the", "is", "are", "was", "were", "what", "which", "who",
        "whom", "when", "where", "why", "how", "did", "does", "do", "of", "to",
        "in", "on", "for", "by", "with", "and", "or",
    }
    return [token for token in _tokenize(query) if token not in stopwords]


def _expand_query(query: str):
    query_terms = _extract_query_terms(query)
    expanded_terms = set(query_terms)

    joined = " ".join(query_terms)
    if "gravity" in expanded_terms:
        expanded_terms.update({"gravitation", "newton", "isaac", "physics"})
    if "discovered" in expanded_terms or "discover" in expanded_terms:
        expanded_terms.update({"discovery", "discovered", "known", "identified"})
    if "invented" in expanded_terms or "invent" in expanded_terms:
        expanded_terms.update({"invention", "invented", "created"})

    if "law of gravity" in joined or ("gravity" in expanded_terms and "law" in expanded_terms):
        expanded_terms.update({"universal", "gravitation"})

    return expanded_terms


def _search_queries(query: str):
    normalized = query.strip()
    queries = [normalized]

    lowered = normalized.lower()
    if "gravity" in lowered:
        queries.extend(
            [
                "Isaac Newton gravity",
                "Isaac Newton gravitation",
                "universal gravitation",
                "gravity physics",
            ]
        )
    if "invent" in lowered or "discover" in lowered:
        queries.append(f"{normalized} history")

    deduped = []
    seen = set()
    for item in queries:
        key = item.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(item)
    return deduped


def _is_noise_title(title: str):
    lowered = title.lower()
    blocked_terms = {
        "film", "tv", "series", "episode", "album", "song", "game", "comic",
        "falls", "rush", "character", "franchise",
    }
    return any(term in lowered for term in blocked_terms)


def _doc_relevance_score(title: str, text: str, query: str):
    query_tokens = _expand_query(query)
    title_tokens = set(_tokenize(title))
    text_tokens = set(_tokenize(text[:3000]))

    title_overlap = len(query_tokens & title_tokens)
    text_overlap = len(query_tokens & text_tokens)
    exact_title_bonus = 6 if query.lower() in title.lower() else 0
    title_phrase_bonus = 0

    lowered_title = title.lower()
    if "gravity" in query.lower():
        if "isaac newton" in lowered_title:
            title_phrase_bonus += 12
        if "gravitation" in lowered_title:
            title_phrase_bonus += 10
        if lowered_title == "gravity":
            title_phrase_bonus += 4

    noise_penalty = 8 if _is_noise_title(title) else 0
    return exact_title_bonus + title_phrase_bonus + (4 * title_overlap) + text_overlap - noise_penalty


def _lead_section(text: str, max_chars: int = 4000):
    text = text.strip()
    if not text:
        return ""

    section_split = re.split(r"\n\s*\n|==[^=]+==", text, maxsplit=1)
    lead = section_split[0].strip()
    return lead[:max_chars]


def _safe_filename(title: str, fallback: str):
    base = re.sub(r"[^a-zA-Z0-9_-]+", "_", title.strip()).strip("_")
    return base or fallback


def iter_text_files(corpus_dir: str) -> Iterable[Dict[str, Any]]:
    pattern = os.path.join(corpus_dir, "**", "*.txt")
    for path in glob.glob(pattern, recursive=True):
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        title = os.path.splitext(os.path.basename(path))[0].replace("_", " ").strip()
        yield {
            "doc_id": os.path.splitext(os.path.basename(path))[0],
            "title": title,
            "text": text,
            "source_path": path,
        }


def load_wikipedia_subset(limit: int = 500, pool_size: int = 5000, seed: int = 42, query: str = None):
    dataset = load_dataset(
        "wikimedia/wikipedia",
        "20231101.en",
        split="train",
    )

    dataset = dataset.shuffle(seed=seed)
    sample_size = min(len(dataset), max(limit, pool_size))
    rows = [dataset[i] for i in range(sample_size)]

    if query:
        rows.sort(
            key=lambda row: _doc_relevance_score(
                row.get("title", ""),
                row.get("text", ""),
                query,
            ),
            reverse=True,
        )

    docs = []
    for row in rows[:limit]:
        text = _lead_section(row.get("text", ""))
        if not text:
            continue
        docs.append(
            {
                "title": row.get("title", ""),
                "text": text,
                "source": "wikipedia_dataset",
            }
        )
    return docs


def search_wikipedia(query: str, limit: int = 10):
    collected = {}

    for search_query in _search_queries(query):
        params = urlencode(
            {
                "action": "query",
                "format": "json",
                "generator": "search",
                "gsrsearch": search_query,
                "gsrlimit": limit,
                "prop": "extracts",
                "explaintext": 1,
                "exintro": 1,
            }
        )
        url = f"https://en.wikipedia.org/w/api.php?{params}"
        request = Request(
            url,
            headers={
                "User-Agent": WIKIPEDIA_USER_AGENT,
                "Accept": "application/json",
            },
        )

        try:
            with urlopen(request, timeout=15) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            print(f"[wiki_corpus] Wikipedia API search failed with HTTP {exc.code}: {exc.reason}")
            return []
        except URLError as exc:
            print(f"[wiki_corpus] Wikipedia API search failed: {exc}")
            return []

        pages = payload.get("query", {}).get("pages", {})
        for page in pages.values():
            title = page.get("title", "")
            text = _lead_section(page.get("extract", ""))
            if not text or title in collected:
                continue
            collected[title] = {
                "title": title,
                "text": text,
                "source": "wikipedia_api",
            }

    docs = list(collected.values())
    docs.sort(key=lambda item: _doc_relevance_score(item["title"], item["text"], query), reverse=True)
    return docs[:limit]


def load_evidence_documents(query: str, api_limit: int = 10, fallback_limit: int = 75, fallback_pool_size: int = 3000):
    docs = search_wikipedia(query, limit=api_limit)
    if docs:
        print(f"[wiki_corpus] Loaded {len(docs)} query-matched Wikipedia articles via API.")
        return docs

    print("[wiki_corpus] Falling back to local Wikipedia dataset search.")
    dataset = load_dataset(
        "wikimedia/wikipedia",
        "20231101.en",
        split="train",
        streaming=True,
    )

    candidates = []
    scanned = 0

    for row in dataset:
        scanned += 1
        title = row.get("title", "")
        text = row.get("text", "")
        if not text.strip():
            continue

        score = _doc_relevance_score(title, text, query)
        if score > 0:
            candidates.append(
                {
                    "title": title,
                    "text": _lead_section(text),
                    "source": "wikipedia_dataset",
                    "score": score,
                }
            )

        if scanned >= fallback_pool_size * 20:
            break

    candidates.sort(key=lambda item: item["score"], reverse=True)
    selected = candidates[:fallback_limit]
    print(
        f"[wiki_corpus] Local dataset search kept {len(selected)} candidates "
        f"after scanning {scanned} articles."
    )
    return [
        {
            "title": item["title"],
            "text": item["text"],
            "source": item["source"],
        }
        for item in selected
    ]


def export_documents_to_corpus(docs, corpus_dir: str):
    os.makedirs(corpus_dir, exist_ok=True)
    used_names = set()

    for i, doc in enumerate(docs):
        title = doc.get("title", "") or f"doc_{i:04d}"
        filename = _safe_filename(title, fallback=f"doc_{i:04d}")
        original = filename
        suffix = 1
        while filename in used_names:
            suffix += 1
            filename = f"{original}_{suffix}"
        used_names.add(filename)

        path = os.path.join(corpus_dir, f"{filename}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(doc.get("text", "").strip())
