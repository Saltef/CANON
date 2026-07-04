import unittest

from canon.eval.methods import (
    MethodSpec,
    citation_support,
    composite_score,
    leaderboard,
    method_summary,
    rank_matrix,
    safe_ratio,
)


class MethodEvalTests(unittest.TestCase):
    def test_safe_ratio(self):
        self.assertEqual(safe_ratio(1, 2), 0.5)
        self.assertEqual(safe_ratio(1, 0), 0.0)

    def test_citation_support_counts_claim_or_preview(self):
        self.assertEqual(
            citation_support(
                [
                    {"claim": {"id": "c"}, "preview": ""},
                    {"claim": None, "preview": "text"},
                    {"claim": None, "preview": ""},
                ]
            ),
            0.666667,
        )

    def test_composite_score_uses_weights(self):
        score = composite_score({"a": 1.0, "b": 0.0}, {"a": 1.0, "b": 1.0})
        self.assertEqual(score, 0.5)

    def test_rank_matrix_merges_chunks(self):
        rows = rank_matrix(
            [
                {"method_id": "a", "retrieved_chunk_ids": ["c1", "c2"]},
                {"method_id": "b", "retrieved_chunk_ids": ["c2", "c3"]},
            ]
        )
        self.assertEqual(rows[0]["chunk_id"], "c1")
        self.assertEqual(rows[1]["ranks"]["b"], 1)

    def test_method_summary_and_leaderboard(self):
        specs = [MethodSpec("m1", "Method 1", "rag", 5, "template")]
        queries = [
            {
                "method_runs": [
                    {
                        "method_id": "m1",
                        "metrics": {
                            "claim_coverage": 1.0,
                            "cluster_diversity": 0.5,
                        },
                    }
                ]
            }
        ]
        summary = method_summary(
            queries,
            specs,
            {"claim_coverage": 0.5, "cluster_diversity": 0.5},
        )
        self.assertEqual(summary["m1"]["composite_score"], 0.75)
        self.assertEqual(leaderboard(summary)[0]["method_id"], "m1")


if __name__ == "__main__":
    unittest.main()
