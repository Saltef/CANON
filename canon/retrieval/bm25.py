from __future__ import annotations

import math
from collections import Counter

from canon.retrieval.tokenize import tokenize


class BM25Index:
    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.tokens = [tokenize(text) for text in texts]
        self.lengths = [len(tokens) for tokens in self.tokens]
        self.avgdl = sum(self.lengths) / len(self.lengths) if self.lengths else 0.0
        self.term_counts = [Counter(tokens) for tokens in self.tokens]
        document_frequency: Counter[str] = Counter()
        for tokens in self.tokens:
            document_frequency.update(set(tokens))
        self.idf = {
            term: math.log(1 + (len(self.tokens) - freq + 0.5) / (freq + 0.5))
            for term, freq in document_frequency.items()
        }

    def score(self, query: str, index: int) -> float:
        query_terms = tokenize(query)
        if not query_terms or not self.tokens:
            return 0.0
        score = 0.0
        counts = self.term_counts[index]
        doc_len = self.lengths[index]
        for term in query_terms:
            tf = counts.get(term, 0)
            if tf == 0:
                continue
            idf = self.idf.get(term, 0.0)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / max(self.avgdl, 1e-9))
            score += idf * (tf * (self.k1 + 1)) / denominator
        return score

    def scores(self, query: str) -> list[float]:
        return [self.score(query, index) for index in range(len(self.tokens))]
