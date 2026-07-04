import unittest

from canon.eval.batches import (
    batch_recommendations,
    parse_batch_sizes,
    score_trajectories,
    stability_report,
)


class EvalBatchTests(unittest.TestCase):
    def test_parse_batch_sizes_sorts_and_deduplicates(self):
        self.assertEqual(parse_batch_sizes("3,1,3"), [1, 3])

    def test_parse_batch_sizes_rejects_non_positive(self):
        with self.assertRaises(ValueError):
            parse_batch_sizes("0")

    def test_stability_report_detects_stable_top(self):
        report = stability_report(
            [
                {
                    "leaderboard": [{"method_id": "m1"}],
                    "technical_diagnostics": {"score_margin_top2": 0.1},
                },
                {
                    "leaderboard": [{"method_id": "m1"}],
                    "technical_diagnostics": {"score_margin_top2": 0.2},
                },
            ]
        )
        self.assertEqual(report["stable_top_method"], "m1")
        self.assertEqual(report["top_change_count"], 0)

    def test_score_trajectories_keeps_metrics_by_batch(self):
        trajectories = score_trajectories(
            [
                {
                    "batch_size": 1,
                    "leaderboard": [
                        {
                            "method_id": "m1",
                            "rank": 1,
                            "composite_score": 0.9,
                            "claim_coverage": 1.0,
                            "cluster_diversity": 1.0,
                            "context_relevance": 0.8,
                            "semantic_alignment": 0.7,
                            "conflict_awareness": 0.0,
                        }
                    ],
                }
            ]
        )
        self.assertEqual(trajectories["m1"][0]["batch_size"], 1)

    def test_batch_recommendations_notes_near_tie(self):
        recs = batch_recommendations(
            [
                {
                    "batch_size": 1,
                    "leaderboard": [{"method_id": "m1"}],
                    "technical_diagnostics": {
                        "score_margin_top2": 0.001,
                        "near_ties": [{"left_method_id": "m1", "right_method_id": "m2"}],
                    },
                }
            ]
        )
        self.assertEqual(len(recs), 3)


if __name__ == "__main__":
    unittest.main()
