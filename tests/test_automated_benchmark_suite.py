import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from canon.product.automated_benchmark_suite import run_automated_benchmark_suite, suite_status


class AutomatedBenchmarkSuiteTests(unittest.TestCase):
    def test_suite_aggregates_multiple_topics_without_human_review(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reports_dir = root / "reports"
            gold_dir = root / "gold"
            conf_dir = root / "conf" / "benchmark_suites"
            reports_dir.mkdir()
            gold_dir.mkdir()
            conf_dir.mkdir(parents=True)
            qrels_path = gold_dir / "unit_qrels.json"
            qrels_path.write_text(
                json.dumps(
                    {
                        "benchmark_id": "unit_qrels",
                        "benchmark_kind": "public_qrels",
                        "queries": [
                            {"id": "q1", "query": "alpha claim", "relevant": {"chunk:a": 1}},
                            {"id": "q2", "query": "beta claim", "relevant": {"chunk:b": 1}},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            suite_path = conf_dir / "suite.json"
            suite_path.write_text(
                json.dumps(
                    {
                        "suite_id": "unit_suite",
                        "model_providers": ["local"],
                        "rerankers": ["heuristic"],
                        "thresholds": {"min_candidate_recall": 0.8, "min_rerank_recall_at_k": 0.5},
                        "benchmarks": [
                            {
                                "id": "unit_benchmark",
                                "topic": "mixed_topic",
                                "mode": "unit_mode",
                                "qrels": "gold/unit_qrels.json",
                                "query_topics": {"q1": "alpha", "q2": "beta"},
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            settings = SimpleNamespace(root=root, reports_dir=reports_dir)
            with patch("canon.product.automated_benchmark_suite.load_settings", return_value=settings):
                with patch("canon.product.automated_benchmark_suite.evaluate_semantic_models") as semantic:
                    with patch("canon.product.automated_benchmark_suite.evaluate_rerankers") as rerank:
                        semantic.return_value = {
                            "providers": [{"provider": "local", "status": "ok"}],
                            "leaderboard": [
                                {"rank": 1, "provider": "local", "primary_score": 0.7, "Recall@k": 1.0}
                            ],
                        }
                        rerank.return_value = {
                            "candidate_generation": {"mode": "pooled_lexical_vector"},
                            "rerankers": [
                                {
                                    "provider": "heuristic",
                                    "status": "ok",
                                    "queries": [
                                        {
                                            "candidate_recall": {"candidate_recall": 1.0},
                                            "score_observability": {
                                                "relevant_nonrelevant_score_gap": 0.2,
                                                "low_confidence_top_k_count": 1,
                                            },
                                        },
                                        {
                                            "candidate_recall": {"candidate_recall": 1.0},
                                            "score_observability": {
                                                "relevant_nonrelevant_score_gap": 0.1,
                                                "low_confidence_top_k_count": 2,
                                            },
                                        },
                                    ],
                                },
                                {
                                    "provider": "cohere",
                                    "status": "unavailable",
                                    "reason": "COHERE_API_KEY is required for Cohere rerank.",
                                },
                            ],
                            "leaderboard": [
                                {"rank": 1, "provider": "heuristic", "primary_score": 0.6, "Recall@k": 1.0}
                            ],
                        }

                        report = run_automated_benchmark_suite(suite_path=suite_path, write_report=False)

        self.assertEqual(report["status"], "automated_pass_human_review_required")
        self.assertEqual(report["benchmark_topic_count"], 1)
        self.assertEqual(report["query_topic_count"], 2)
        self.assertEqual(report["benchmarks"][0]["query_topics"], {"alpha": 1, "beta": 1})
        self.assertEqual(report["query_topic_summary"]["alpha"]["query_count"], 1)
        self.assertEqual(report["leaderboards"]["rerank"][0]["provider"], "heuristic")
        self.assertEqual(report["human_review_boundary"]["status"], "required_for_release_claims")
        self.assertEqual(report["model_matrix"]["status"], "partial_external_models_unavailable")
        self.assertEqual(report["model_matrix"]["rerank"]["unavailable"][0]["provider"], "cohere")

    def test_suite_status_fails_when_candidate_recall_is_too_low(self):
        report = {
            "semantic": {"status": "ok"},
            "rerank": {
                "status": "ok",
                "candidate_recall": {"mean": 0.5},
                "best": {"Recall@k": 1.0},
            },
        }
        self.assertEqual(
            suite_status([report], {"min_candidate_recall": 0.8, "min_rerank_recall_at_k": 0.5}),
            "automated_fail",
        )

    def test_suite_reuses_cached_benchmark_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reports_dir = root / "reports"
            gold_dir = root / "gold"
            conf_dir = root / "conf" / "benchmark_suites"
            reports_dir.mkdir()
            gold_dir.mkdir()
            conf_dir.mkdir(parents=True)
            (gold_dir / "unit_qrels.json").write_text(
                json.dumps(
                    {
                        "benchmark_id": "unit_qrels",
                        "benchmark_kind": "public_qrels",
                        "queries": [{"id": "q1", "query": "alpha", "relevant": {"chunk:a": 1}}],
                    }
                ),
                encoding="utf-8",
            )
            suite_path = conf_dir / "suite.json"
            suite_path.write_text(
                json.dumps(
                    {
                        "suite_id": "unit_suite",
                        "model_providers": ["local"],
                        "rerankers": ["heuristic"],
                        "thresholds": {"min_candidate_recall": 0.8, "min_rerank_recall_at_k": 0.5},
                        "benchmarks": [{"id": "unit_benchmark", "mode": "unit_mode", "qrels": "gold/unit_qrels.json"}],
                    }
                ),
                encoding="utf-8",
            )
            settings = SimpleNamespace(root=root, reports_dir=reports_dir)
            with patch("canon.product.automated_benchmark_suite.load_settings", return_value=settings):
                with patch("canon.product.automated_benchmark_suite.evaluate_semantic_models") as semantic:
                    with patch("canon.product.automated_benchmark_suite.evaluate_rerankers") as rerank:
                        semantic.return_value = {
                            "providers": [{"provider": "local", "status": "ok"}],
                            "leaderboard": [{"rank": 1, "provider": "local", "primary_score": 0.7}],
                        }
                        rerank.return_value = {
                            "candidate_generation": {"mode": "pooled_lexical_vector"},
                            "rerankers": [
                                {
                                    "provider": "heuristic",
                                    "status": "ok",
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
                            ],
                            "leaderboard": [
                                {"rank": 1, "provider": "heuristic", "primary_score": 0.6, "Recall@k": 1.0}
                            ],
                        }

                        first = run_automated_benchmark_suite(suite_path=suite_path, write_report=False)
                        second = run_automated_benchmark_suite(suite_path=suite_path, write_report=False)

        self.assertEqual(semantic.call_count, 1)
        self.assertEqual(rerank.call_count, 1)
        self.assertFalse(first["benchmarks"][0]["cache_hit"])
        self.assertTrue(second["benchmarks"][0]["cache_hit"])
        self.assertEqual(second["cache"]["cached_benchmark_count"], 1)


if __name__ == "__main__":
    unittest.main()
