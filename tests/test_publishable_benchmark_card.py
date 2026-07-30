import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from canon.product.publishable_benchmark_card import build_publishable_benchmark_card


class PublishableBenchmarkCardTests(unittest.TestCase):
    def test_card_summarizes_failure_diagnostics_and_review_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reports = root / "reports"
            reports.mkdir()
            suite_path = reports / "suite.json"
            optimizer_path = reports / "optimizer.json"
            review_path = reports / "review.json"
            disagreement_path = reports / "disagreement.json"
            suite_path.write_text(json.dumps(suite_report()), encoding="utf-8")
            optimizer_path.write_text(json.dumps(optimizer_report()), encoding="utf-8")
            review_path.write_text(json.dumps(human_review_report(status="incomplete")), encoding="utf-8")
            disagreement_path.write_text(json.dumps(disagreement_report()), encoding="utf-8")
            settings = SimpleNamespace(root=root, reports_dir=reports)
            with patch("canon.product.publishable_benchmark_card.load_settings", return_value=settings):
                card = build_publishable_benchmark_card(
                    automated_suite_report=suite_path,
                    optimizer_report=optimizer_path,
                    human_review_status_report=review_path,
                    disagreement_report=disagreement_path,
                    write_report=False,
                )

        self.assertEqual(card["report_id"], "publishable_benchmark_card_v1")
        self.assertEqual(card["status"], "benchmark_card_ready_with_automated_failures")
        self.assertEqual(card["aggregate_diagnostics"]["candidate_recall_failure_count"], 1)
        self.assertEqual(card["aggregate_diagnostics"]["ranking_failure_count"], 1)
        self.assertEqual(card["aggregate_diagnostics"]["candidate_miss_count"], 6)
        self.assertIn("nfcorpus_full", card["aggregate_diagnostics"]["interpretation"])
        self.assertEqual(card["human_review"]["status"], "incomplete")
        self.assertTrue(card["claim_boundary"]["human_review_required"])
        self.assertEqual(card["optimizer"]["fusion_methods"], ["union", "rrf"])
        self.assertEqual(
            [row["fusion"] for row in card["optimizer"]["fusion_summary"]],
            ["rrf", "union"],
        )
        self.assertIn(
            "Improve candidate generation before spending effort on reranker comparisons.",
            card["recommended_next_actions"],
        )

    def test_card_status_moves_to_review_blocked_when_automated_gates_pass(self):
        report = suite_report()
        report["status"] = "automated_pass_human_review_required"
        report["benchmarks"][0]["rerank"]["candidate_recall"] = {
            "query_count": 2,
            "mean": 1.0,
            "min": 1.0,
            "miss_count": 0,
        }
        report["benchmarks"][0]["rerank"]["best"]["Recall@k"] = 0.9
        report["benchmarks"][1]["rerank"]["candidate_recall"] = {
            "query_count": 3,
            "mean": 0.9,
            "min": 0.8,
            "miss_count": 0,
        }
        report["benchmarks"][1]["rerank"]["best"]["Recall@k"] = 0.6
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reports = root / "reports"
            reports.mkdir()
            suite_path = reports / "suite.json"
            optimizer_path = reports / "optimizer.json"
            review_path = reports / "review.json"
            suite_path.write_text(json.dumps(report), encoding="utf-8")
            optimizer_path.write_text(json.dumps(optimizer_report()), encoding="utf-8")
            review_path.write_text(json.dumps(human_review_report(status="incomplete")), encoding="utf-8")
            settings = SimpleNamespace(root=root, reports_dir=reports)
            with patch("canon.product.publishable_benchmark_card.load_settings", return_value=settings):
                card = build_publishable_benchmark_card(
                    automated_suite_report=suite_path,
                    optimizer_report=optimizer_path,
                    human_review_status_report=review_path,
                    disagreement_report=None,
                    write_report=False,
                )

        self.assertEqual(card["aggregate_diagnostics"]["status"], "automated_pass_human_review_required")
        self.assertEqual(card["status"], "benchmark_card_ready_human_review_blocked")


def suite_report():
    return {
        "report_id": "automated_benchmark_suite_v1",
        "suite_id": "stage1_public_full_v1",
        "status": "automated_fail",
        "thresholds": {"min_candidate_recall": 0.8, "min_rerank_recall_at_k": 0.5},
        "model_matrix": {"status": "complete"},
        "benchmark_count": 2,
        "benchmarks": [
            {
                "id": "scifact_claims_full",
                "topic": "claims",
                "mode": "scifact",
                "qrels_path": "gold/scifact.json",
                "query_count": 2,
                "candidate_k": 50,
                "lexical_k": 50,
                "vector_k": 50,
                "pooled_candidates": True,
                "fusion": "union",
                "document_format": "structured",
                "semantic": {"best": metric("local:hashed", 0.3, 0.4, 0.2)},
                "rerank": {
                    "candidate_generation": {"query_variant_count": 1},
                    "best": metric("heuristic", 0.6, 0.7, 0.5),
                    "candidate_recall": {"query_count": 2, "mean": 0.9, "min": 0.8, "miss_count": 1},
                    "score_observability": {
                        "mean_relevant_nonrelevant_score_gap": 0.2,
                        "mean_low_confidence_top_k_count": 1.0,
                    },
                },
            },
            {
                "id": "nfcorpus_full",
                "topic": "nutrition",
                "mode": "nfcorpus",
                "qrels_path": "gold/nfcorpus.json",
                "query_count": 3,
                "candidate_k": 50,
                "lexical_k": 50,
                "vector_k": 50,
                "pooled_candidates": True,
                "fusion": "union",
                "document_format": "structured",
                "semantic": {"best": metric("local:hashed", 0.1, 0.1, 0.2)},
                "rerank": {
                    "candidate_generation": {"query_variant_count": 1},
                    "best": metric("heuristic", 0.2, 0.1, 0.3),
                    "candidate_recall": {"query_count": 3, "mean": 0.2, "min": 0.0, "miss_count": 5},
                    "score_observability": {
                        "mean_relevant_nonrelevant_score_gap": 0.1,
                        "mean_low_confidence_top_k_count": 2.0,
                    },
                },
            },
        ],
    }


def metric(provider_id, ndcg, recall, mrr):
    return {
        "provider_id": provider_id,
        "primary_metric": "nDCG@10",
        "primary_score": ndcg,
        "Recall@k": recall,
        "MRR@k": mrr,
        "elapsed_ms": 10.0,
    }


def optimizer_report():
    return {
        "report_id": "stage1_optimizer_v1",
        "suite_id": "stage1_public_full_v1",
        "status": "automated_fail",
        "fusion_methods": ["union", "rrf"],
        "trial_count": 2,
        "cache": {"cached_trial_count": 2},
        "best_trial": {
            "benchmark_id": "scifact_claims_full",
            "dense_retriever": "local",
            "reranker": "heuristic",
            "candidate_k": 50,
            "fusion": "union",
            "composite_score": 0.5,
            "candidate_recall_mean": 0.9,
            "nDCG@k": 0.6,
            "Recall@k": 0.7,
        },
        "trials": [
            {
                "fusion": "union",
                "status": "ok",
                "composite_score": 0.5,
                "candidate_recall_mean": 0.9,
                "nDCG@k": 0.6,
                "Recall@k": 0.7,
            },
            {
                "fusion": "rrf",
                "status": "ok",
                "composite_score": 0.4,
                "candidate_recall_mean": 0.8,
                "nDCG@k": 0.5,
                "Recall@k": 0.6,
            },
        ],
    }


def human_review_report(status):
    return {
        "report_id": "publishable_human_review_status_v1",
        "suite_id": "stage1_public_full_v1",
        "status": status,
        "retrieval_review": {"complete_query_count": 0, "target": 100},
        "synthesis_review": {"complete_case_count": 0, "target": 30},
        "reviewed_label_count": 0,
        "validation_error_count": 0,
    }


def disagreement_report():
    return {
        "report_id": "disagreement_preservation_benchmark_v1",
        "benchmark_id": "disagreement",
        "status": "automated_pass_human_review_required",
        "case_count": 1,
        "summary": {
            "case_count": 1,
            "average_metrics": {
                "disagreement_preservation_score": 0.9,
                "contradiction_recall": 1.0,
                "citation_integrity": 1.0,
                "unsupported_claim_rate": 0.0,
            },
        },
        "human_review_required": True,
    }


if __name__ == "__main__":
    unittest.main()
