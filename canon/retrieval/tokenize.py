from __future__ import annotations

import re
import unicodedata

TOKEN_RE = re.compile(r"[^\W_]+(?:'[^\W_]+)*", re.UNICODE)

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "do",
    "does",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "its",
    "main",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "to",
    "what",
    "whether",
    "with",
}


def tokenize(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text)
    normalized = "".join(
        " " if unicodedata.category(char).startswith("P") and char not in {"'"} else char
        for char in normalized
    )
    return [token.casefold() for token in TOKEN_RE.findall(normalized) if token.casefold() not in STOPWORDS]
