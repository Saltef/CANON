import unittest
from pathlib import Path
from types import SimpleNamespace

from canon.baselines import llamaindex_baseline as baseline


class LlamaIndexBaselineTests(unittest.TestCase):
    def test_evaluate_query_scores_ranked_chunk_ids(self):
        retriever = FakeRetriever(["c1", "c2"])
        report = baseline.evaluate_llamaindex_query(
            retriever,
            {"id": "q1", "query": "alpha", "relevant": {"c1": 3.0}},
            k=2,
            parent_qrels=False,
            chunk_to_work={"c1": "w1", "c2": "w2"},
        )

        self.assertEqual(report["ranked_chunk_ids"], ["c1", "c2"])
        self.assertEqual(report["metrics"]["nDCG@2"], 1.0)
        self.assertEqual(report["candidate_recall"]["candidate_recall"], 1.0)
        self.assertEqual(report["relevant_hit_ids"], ["c1"])

    def test_parent_qrels_use_fixed_corpus_map_before_scoring(self):
        retriever = FakeRetriever(["sibling"])
        query = {"id": "q1", "query": "alpha", "relevant": {"gold": 1.0}}
        chunk_map = {"gold": "work-1", "sibling": "work-1"}

        original = baseline.evaluate_llamaindex_query(
            retriever,
            query,
            k=1,
            parent_qrels=False,
            chunk_to_work=chunk_map,
        )
        expanded = baseline.evaluate_llamaindex_query(
            retriever,
            query,
            k=1,
            parent_qrels=True,
            chunk_to_work=chunk_map,
        )

        self.assertEqual(original["metrics"]["nDCG@1"], 0.0)
        self.assertEqual(expanded["metrics"]["nDCG@1"], 1.0)
        self.assertEqual(expanded["qrels_semantics"]["parent_qrels_protocol"], baseline.PARENT_QRELS_PROTOCOL)

    def test_build_report_publishes_primary_standing_verdict_and_compacts_runs(self):
        llama_runs = [
            run("llamaindex_vector_original_qrels", repeat, 0.10)
            for repeat in [1, 2, 3]
        ] + [
            run("llamaindex_vector_fixed_parent_qrels", repeat, 0.11)
            for repeat in [1, 2, 3]
        ]
        canon_runs = [
            run("base_qwen_cohere", repeat, 0.40)
            for repeat in [1, 2, 3]
        ]

        report = baseline.build_report(
            llama_runs=llama_runs,
            canon_runs=canon_runs,
            canon_report={"status": "repeat_spread_complete"},
            configs=baseline.DEFAULT_CONFIGS,
            repeats=3,
            mode="m",
            qrels_path=Path("gold/qrels.json"),
            qrels={"queries": [{"id": "q1", "relevant": {"c1": 1.0}}]},
            k=10,
            candidate_k=250,
            include_query_details=False,
        )

        self.assertEqual(report["standing_verdict"]["winner"], "canon")
        self.assertEqual(report["standing_verdict"]["standing"], "stands_under_repeat_spread_check")
        self.assertFalse(any("queries" in row for row in report["llamaindex_runs"]))


class FakeRetriever:
    def __init__(self, chunk_ids):
        self.chunk_ids = list(chunk_ids)

    def retrieve(self, _query):
        return [
            SimpleNamespace(
                node=SimpleNamespace(
                    metadata={
                        "chunk_id": chunk_id,
                        "work_id": f"work-{chunk_id}",
                        "title": f"Title {chunk_id}",
                        "source_name": "Fixture",
                    },
                    node_id=chunk_id,
                ),
                score=1.0 / index,
            )
            for index, chunk_id in enumerate(self.chunk_ids, start=1)
        ]


def run(configuration: str, repeat: int, ndcg: float):
    return {
        "configuration": configuration,
        "repeat": repeat,
        "status": "ok",
        "metrics": {
            "nDCG@10": ndcg,
            "Recall@10": 0.1,
            "MAP@10": 0.2,
            "MRR@10": 0.3,
            "candidate_recall_mean": 0.4,
            "work_level_hit_rate_mean": 0.5,
        },
        "queries": [{"id": "q1"}],
    }


if __name__ == "__main__":
    unittest.main()
