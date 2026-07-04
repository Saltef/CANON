import unittest

from canon.eval.pipeline import (
    compact_method_report,
    recommendations,
    technical_diagnostics,
)


class EvalPipelineTests(unittest.TestCase):
    def test_compact_method_report_keeps_leaderboard_metrics(self):
        compact = compact_method_report(
            {
                "leaderboard": [
                    {
                        "rank": 1,
                        "method_id": "m1",
                        "label": "Method",
                        "composite_score": 0.9,
                        "average_metrics": {
                            "claim_coverage": 1.0,
                            "cluster_diversity": 1.0,
                            "context_relevance": 0.8,
                            "semantic_alignment": 0.7,
                            "conflict_awareness": 0.2,
                        },
                    }
                ],
                "method_summary": {
                    "m1": {
                        "label": "Method",
                        "composite_score": 0.9,
                        "average_metrics": {"claim_coverage": 1.0},
                    }
                },
                "pairwise": [],
            }
        )
        self.assertEqual(compact["leaderboard"][0]["method_id"], "m1")
        self.assertIn("pairwise_highlights", compact)

    def test_recommendations_notice_tie_and_low_conflict_awareness(self):
        recs = recommendations(
            {
                "leaderboard": [
                    {
                        "method_id": "m1",
                        "composite_score": 0.9,
                        "conflict_awareness": 0.0,
                        "context_relevance": 0.8,
                    },
                    {
                        "method_id": "m2",
                        "composite_score": 0.895,
                        "conflict_awareness": 0.0,
                        "context_relevance": 0.8,
                    },
                ]
            },
            {
                "near_ties": [
                    {
                        "left_method_id": "m1",
                        "right_method_id": "m2",
                        "margin": 0.005,
                    }
                ]
            },
        )
        self.assertEqual(len(recs), 3)

    def test_technical_diagnostics_detects_near_tie(self):
        diagnostics = technical_diagnostics(
            {
                "method_summary": {
                    "m1": {
                        "average_metrics": {
                            "claim_coverage": 1.0,
                            "cluster_diversity": 1.0,
                            "source_diversity": 1.0,
                            "context_relevance": 0.8,
                            "semantic_alignment": 0.7,
                            "citation_support": 1.0,
                            "conflict_awareness": 0.0,
                        }
                    },
                    "m2": {
                        "average_metrics": {
                            "claim_coverage": 1.0,
                            "cluster_diversity": 0.8,
                            "source_diversity": 1.0,
                            "context_relevance": 0.7,
                            "semantic_alignment": 0.6,
                            "citation_support": 1.0,
                            "conflict_awareness": 0.0,
                        }
                    },
                },
                "pairwise": [{"average_retrieval_overlap": 0.5}],
                "queries": [
                    {
                        "id": "q1",
                    }
                ],
            },
            {
                "leaderboard": [
                    {
                        "method_id": "m1",
                        "composite_score": 0.9,
                        "conflict_awareness": 0.0,
                        "context_relevance": 0.8,
                    },
                    {
                        "method_id": "m2",
                        "composite_score": 0.895,
                        "conflict_awareness": 0.0,
                        "context_relevance": 0.7,
                    },
                ]
            },
        )
        self.assertEqual(diagnostics["score_margin_top2"], 0.005)
        self.assertEqual(len(diagnostics["near_ties"]), 1)
        self.assertEqual(diagnostics["retrieval_stability"]["average_pairwise_overlap"], 0.5)


if __name__ == "__main__":
    unittest.main()
