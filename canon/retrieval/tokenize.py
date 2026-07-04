from __future__ import annotations

import re

TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_'-]*")

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
    return [token.lower() for token in TOKEN_RE.findall(text) if token.lower() not in STOPWORDS]
