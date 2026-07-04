import unittest

from canon.eval.external_ir import leaderboard, method_summary
from canon.eval.ir_metrics import (
    average_precision_at_k,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)


class ExternalIRTests(unittest.TestCase):
    def test_ir_metrics_reward_relevant_ranked_results(self):
        ranked = ["c3", "c1", "c2"]
        qrels = {"c1": 3, "c2": 1}
        self.assertGreater(ndcg_at_k(ranked, qrels, 3), 0.0)
        self.assertEqual(recall_at_k(ranked, qrels, 3), 1.0)
        self.assertEqual(average_precision_at_k(ranked, qrels, 3), 0.583333)
        self.assertEqual(reciprocal_rank(ranked, qrels, 3), 0.5)

    def test_method_summary_averages_metrics(self):
        summary = method_summary(
            [
                {
                    "method_runs": [
                        {"method_id": "m1", "metrics": {"nDCG@5": 1.0, "Recall@5": 0.5, "MAP@5": 0.25, "MRR@5": 1.0}},
                    ]
                },
                {
                    "method_runs": [
                        {"method_id": "m1", "metrics": {"nDCG@5": 0.5, "Recall@5": 1.0, "MAP@5": 0.75, "MRR@5": 0.5}},
                    ]
                },
            ],
            ["m1"],
            5,
        )
        self.assertEqual(summary["m1"]["nDCG@5"], 0.75)
        self.assertEqual(summary["m1"]["MAP@5"], 0.5)

    def test_leaderboard_sorts_by_primary_metric(self):
        rows = leaderboard({"a": {"nDCG@5": 0.1}, "b": {"nDCG@5": 0.9}}, "nDCG@5")
        self.assertEqual(rows[0]["method_id"], "b")


if __name__ == "__main__":
    unittest.main()
