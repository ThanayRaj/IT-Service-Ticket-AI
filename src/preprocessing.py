"""Conservative, shared text preparation for ticket NLP."""

import html
import re
from typing import Iterable, List, Optional, Tuple


_WHITESPACE = re.compile(r"\s+")


def clean_text(value: Optional[str]) -> str:
    """Decode HTML entities and normalize whitespace without deleting meaning.

    Case folding, tokenization, and feature selection belong in the fitted
    vectorizer pipeline so those transformations are consistent at inference.
    """
    if value is None:
        return ""
    decoded = html.unescape(str(value))
    return _WHITESPACE.sub(" ", decoded).strip()


def normalized_body_key(value: Optional[str]) -> str:
    """Return a case-insensitive key for grouping identical bodies in splits."""
    return clean_text(value).casefold()


def prepare_labeled_text(
    rows: Iterable[dict], target: str
) -> Tuple[List[str], List[str], List[str]]:
    """Return cleaned non-empty bodies, labels, and duplicate-group keys.

    Empty bodies are excluded from supervised training, not assigned a label or
    replaced with fabricated text. Duplicate keys let a later split keep every
    copy of a body in the same partition.
    """
    texts = []
    labels = []
    groups = []
    for row in rows:
        body = clean_text(row.get("Body"))
        label = str(row.get(target, "")).strip()
        if not body or not label:
            continue
        texts.append(body)
        labels.append(label)
        groups.append(normalized_body_key(body))
    return texts, labels, groups
