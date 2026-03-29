import glob
import os
from typing import Iterable, Dict, Any


def iter_text_files(corpus_dir: str) -> Iterable[Dict[str, Any]]:
    pattern = os.path.join(corpus_dir, "**", "*.txt")
    for path in glob.glob(pattern, recursive=True):
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        yield {
            "doc_id": os.path.splitext(os.path.basename(path))[0],
            "text": text,
            "source_path": path,
        }
