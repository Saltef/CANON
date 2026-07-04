from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

from canon.retrieval.tokenize import tokenize


@dataclass(frozen=True)
class SparseEmbedding:
    model_id: str
    dimensions: int
    values: dict[int, float]


class HashedSemanticEncoder:
    def __init__(self, dimensions: int = 512, model_id: str = "hashed-semantic-v1") -> None:
        self.dimensions = dimensions
        self.model_id = model_id

    def encode(self, text: str) -> SparseEmbedding:
        features = token_features(text)
        values: dict[int, float] = {}
        for feature, weight in features.items():
            bucket = stable_bucket(feature, self.dimensions)
            values[bucket] = values.get(bucket, 0.0) + weight
        norm = math.sqrt(sum(value * value for value in values.values()))
        if norm > 0:
            values = {bucket: value / norm for bucket, value in values.items()}
        return SparseEmbedding(
            model_id=self.model_id,
            dimensions=self.dimensions,
            values=values,
        )


def token_features(text: str) -> dict[str, float]:
    tokens = tokenize(text)
    features: dict[str, float] = {}
    for token in tokens:
        if len(token) < 3:
            continue
        features[f"tok:{token}"] = features.get(f"tok:{token}", 0.0) + 1.0
    for left, right in zip(tokens, tokens[1:], strict=False):
        if len(left) < 3 or len(right) < 3:
            continue
        features[f"bi:{left}_{right}"] = features.get(f"bi:{left}_{right}", 0.0) + 0.65
    return features


def stable_bucket(feature: str, dimensions: int) -> int:
    digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % dimensions


def cosine(left: SparseEmbedding, right: SparseEmbedding) -> float:
    if not left.values or not right.values:
        return 0.0
    if len(left.values) > len(right.values):
        left, right = right, left
    return sum(value * right.values.get(bucket, 0.0) for bucket, value in left.values.items())


def semantic_scores(query: str, documents: list[str], dimensions: int = 512) -> list[float]:
    encoder = HashedSemanticEncoder(dimensions=dimensions)
    query_embedding = encoder.encode(query)
    return [cosine(query_embedding, encoder.encode(document)) for document in documents]
