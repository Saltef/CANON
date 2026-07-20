import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

from canon.product.stage1_optimizer import run_stage1_optimizer


class Stage1OptimizerTests(unittest.TestCase):
    def test_optimizer_ranks_complete_retrieval_stacks(self):
        def fake_rerankers(**kwargs):
            vector_model = kwargs.get("vector_model")
            reranker = kwargs["rerankers"][0]
            ndcg = 0.9 if vector_model == "qwen/qwen3-embedding-8b" else 0.5
            return {
                "rerankers": [
                    {
                        "provider": reranker.split(":", 1)[0],
                        "provider_id": reranker,
                        "status": "ok",
                        "elapsed_ms": 10,
                        "summary": {
                            "nDCG@10": ndcg,
                            "Recall@10": 1.0,
                            "MAP@10": 0.8,
                            "MRR@10": ndcg,
                        },
                        "queries": [
                            {
                                "candidate_recall": {"candidate_recall": 1.0},
                                "score_observability": {
                                    "relevant_nonrelevant_score_gap": 0.2,
                                    "low_confidence_top_k_count": 0,
                                },
                            }
                        ],
                    }
                ]
            }

        suite = {
            "suite_id": "unit_suite",
            "model_providers": ["local", "openrouter:qwen/qwen3-embedding-8b"],
            "rerankers": ["heuristic"],
            "benchmarks": [{"id": "unit", "mode": "unit_mode", "qrels": "gold/unit.json"}],
        }
        with patch("canon.product.stage1_optimizer.load_suite", return_value=suite):
            with patch("canon.product.stage1_optimizer.load_qrels", return_value={"queries": [{"id": "q1"}]}):
                with patch("canon.product.stage1_optimizer.evaluate_rerankers", side_effect=fake_rerankers):
                    report = run_stage1_optimizer(
                        suite_path=Path("conf/benchmark_suites/unit.json"),
                        dense_retrievers=["local", "openrouter:qwen/qwen3-embedding-8b"],
                        rerankers=["heuristic"],
                        candidate_values=[25],
                        resume=False,
                        write_report=False,
                    )

        self.assertEqual(report["status"], "automated_pass_human_review_required")
        self.assertEqual(report["dense_retrievers"], ["local", "openrouter:qwen/qwen3-embedding-8b"])
        self.assertEqual(report["leaderboard"][0]["dense_retriever"], "openrouter:qwen/qwen3-embedding-8b")
        self.assertEqual(report["generation_metrics_boundary"]["status"], "not_stage1_objective")
        self.assertEqual(report["leaderboard"][0]["lexical_search"], "bm25")

    def test_optimizer_reuses_cached_trials(self):
        def fake_rerankers(**kwargs):
            return {
                "rerankers": [
                    {
                        "provider": "heuristic",
                        "provider_id": "heuristic",
                        "status": "ok",
                        "elapsed_ms": 10,
                        "summary": {
                            "nDCG@10": 0.8,
                            "Recall@10": 1.0,
                            "MAP@10": 0.8,
                            "MRR@10": 0.8,
                        },
                        "queries": [
                            {
                                "candidate_recall": {"candidate_recall": 1.0},
                                "score_observability": {
                                    "relevant_nonrelevant_score_gap": 0.2,
                                    "low_confidence_top_k_count": 0,
                                },
                            }
                        ],
                    }
                ]
            }

        suite = {
            "suite_id": "unit_suite",
            "model_providers": ["local"],
            "rerankers": ["heuristic"],
            "benchmarks": [{"id": "unit", "mode": "unit_mode", "qrels": "gold/unit.json"}],
        }
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict("os.environ", {"CANON_REPORTS_DIR": directory}):
                with patch("canon.product.stage1_optimizer.load_suite", return_value=suite):
                    with patch("canon.product.stage1_optimizer.load_qrels", return_value={"queries": [{"id": "q1"}]}):
                        with patch(
                            "canon.product.stage1_optimizer.evaluate_rerankers",
                            side_effect=fake_rerankers,
                        ) as evaluate:
                            first = run_stage1_optimizer(
                                suite_path=Path("conf/benchmark_suites/unit.json"),
                                dense_retrievers=["local"],
                                rerankers=["heuristic"],
                                candidate_values=[25],
                                write_report=False,
                            )
                            second = run_stage1_optimizer(
                                suite_path=Path("conf/benchmark_suites/unit.json"),
                                dense_retrievers=["local"],
                                rerankers=["heuristic"],
                                candidate_values=[25],
                                write_report=False,
                            )

        self.assertEqual(evaluate.call_count, 1)
        self.assertFalse(first["trials"][0]["cache_hit"])
        self.assertTrue(second["trials"][0]["cache_hit"])


if __name__ == "__main__":
    unittest.main()
