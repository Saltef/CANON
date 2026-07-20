import unittest
from unittest.mock import patch

from canon.product.stage1_sweep import run_stage1_sweep


class Stage1SweepTests(unittest.TestCase):
    def test_stage1_sweep_aggregates_candidate_runs(self):
        def fake_suite(**kwargs):
            candidate_k = kwargs["candidate_k"]
            return {
                "status": "automated_pass_human_review_required",
                "model_matrix": {"status": "complete"},
                "leaderboards": {
                    "semantic": [
                        {
                            "rank": 1,
                            "provider": "openrouter:qwen/qwen3-embedding-8b",
                            "mean_primary_score": 0.8 + candidate_k / 1000,
                        }
                    ],
                    "rerank": [
                        {
                            "rank": 1,
                            "provider": "cohere:rerank-v4.0-fast",
                            "mean_primary_score": 0.7 + candidate_k / 1000,
                        }
                    ],
                },
                "benchmarks": [
                    {
                        "id": "unit",
                        "rerank": {
                            "candidate_recall": {
                                "query_count": 2,
                                "mean": 1.0,
                                "min": 1.0,
                                "miss_count": 0,
                            }
                        },
                        "semantic": {"best": {}},
                    }
                ],
            }

        with patch("canon.product.stage1_sweep.run_automated_benchmark_suite", side_effect=fake_suite):
            report = run_stage1_sweep(candidate_values=[25, 50], write_report=False)

        self.assertEqual(report["status"], "automated_pass_human_review_required")
        self.assertEqual(report["best_semantic"][0]["candidate_k"], 50)
        self.assertEqual(report["candidate_recall"][0]["mean"], 1.0)


if __name__ == "__main__":
    unittest.main()
