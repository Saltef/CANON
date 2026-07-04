from __future__ import annotations

import math


def relevance_at(ranked_ids: list[str], qrels: dict[str, int | float], k: int) -> list[float]:
    return [float(qrels.get(chunk_id, 0.0)) for chunk_id in ranked_ids[:k]]


def dcg(relevances: list[float]) -> float:
    return sum((2.0**relevance - 1.0) / math.log2(index + 2) for index, relevance in enumerate(relevances))


def ndcg_at_k(ranked_ids: list[str], qrels: dict[str, int | float], k: int) -> float:
    if k <= 0:
        return 0.0
    actual = dcg(relevance_at(ranked_ids, qrels, k))
    ideal = dcg(sorted((float(value) for value in qrels.values()), reverse=True)[:k])
    if ideal <= 0:
        return 0.0
    return round(actual / ideal, 6)


def recall_at_k(ranked_ids: list[str], qrels: dict[str, int | float], k: int) -> float:
    relevant = {chunk_id for chunk_id, relevance in qrels.items() if float(relevance) > 0.0}
    if not relevant:
        return 0.0
    retrieved = set(ranked_ids[:k])
    return round(len(relevant & retrieved) / len(relevant), 6)


def average_precision_at_k(ranked_ids: list[str], qrels: dict[str, int | float], k: int) -> float:
    relevant = {chunk_id for chunk_id, relevance in qrels.items() if float(relevance) > 0.0}
    if not relevant or k <= 0:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for index, chunk_id in enumerate(ranked_ids[:k], start=1):
        if chunk_id in relevant:
            hits += 1
            precision_sum += hits / index
    denominator = min(len(relevant), k)
    return round(precision_sum / denominator, 6)


def reciprocal_rank(ranked_ids: list[str], qrels: dict[str, int | float], k: int) -> float:
    relevant = {chunk_id for chunk_id, relevance in qrels.items() if float(relevance) > 0.0}
    if not relevant:
        return 0.0
    for index, chunk_id in enumerate(ranked_ids[:k], start=1):
        if chunk_id in relevant:
            return round(1.0 / index, 6)
    return 0.0


def evaluate_ranking(ranked_ids: list[str], qrels: dict[str, int | float], k: int) -> dict:
    return {
        f"nDCG@{k}": ndcg_at_k(ranked_ids, qrels, k),
        f"Recall@{k}": recall_at_k(ranked_ids, qrels, k),
        f"MAP@{k}": average_precision_at_k(ranked_ids, qrels, k),
        f"MRR@{k}": reciprocal_rank(ranked_ids, qrels, k),
    }


def mean_metric(rows: list[dict], metric: str) -> float:
    values = [float(row.get(metric, 0.0)) for row in rows]
    if not values:
        return 0.0
    return round(sum(values) / len(values), 6)
