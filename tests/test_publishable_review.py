import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from canon.product.publishable_review import build_publishable_review_status


class PublishableReviewTests(unittest.TestCase):
    def test_status_is_complete_when_review_targets_are_met(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reports = root / "reports"
            reports.mkdir()
            scaffold = write_scaffold(reports, retrieval_target=1, synthesis_target=1)
            retrieval_csv = reports / "retrieval.csv"
            synthesis_csv = reports / "synthesis.csv"
            write_retrieval_csv(
                retrieval_csv,
                [
                    {
                        "review_set": "retrieval_qrels",
                        "reviewer_id": "reviewer-a",
                        "benchmark_id": "unit_qrels",
                        "benchmark_kind": "public_qrels",
                        "query_id": "q1",
                        "query": "alpha",
                        "candidate_chunk_id": "c1",
                        "candidate_title": "Title",
                        "candidate_preview": "Preview",
                        "relevance": "2",
                        "evidence_role": "supports",
                        "reviewer_notes": "",
                    }
                ],
            )
            write_synthesis_csv(
                synthesis_csv,
                [
                    {
                        "review_set": "synthesis_disagreement",
                        "reviewer_id": "reviewer-a",
                        "case_id": "case1",
                        "question": "Where is evidence contested?",
                        "claim_id": "claim1",
                        "claim": "Alpha is contested.",
                        "evidence_id": "E1",
                        "expected_stance": "supports",
                        "citation_valid": "yes",
                        "stance_correct": "yes",
                        "claim_supported": "yes",
                        "missed_key_evidence": "no",
                        "reviewer_action": "accept",
                        "reason": "",
                    }
                ],
            )
            settings = SimpleNamespace(root=root, reports_dir=reports)
            with patch("canon.product.publishable_review.load_settings", return_value=settings):
                report = build_publishable_review_status(
                    retrieval_csv=retrieval_csv,
                    synthesis_csv=synthesis_csv,
                    scaffold_path=scaffold,
                    write_report=False,
                )

        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["suite_id"], "unit_suite")
        self.assertEqual(report["retrieval_review"]["complete_query_count"], 1)
        self.assertEqual(report["synthesis_review"]["complete_case_count"], 1)
        self.assertEqual(report["validation_error_count"], 0)

    def test_status_remains_incomplete_for_blank_scaffold_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reports = root / "reports"
            reports.mkdir()
            scaffold = write_scaffold(reports, retrieval_target=1, synthesis_target=1)
            retrieval_csv = reports / "retrieval.csv"
            synthesis_csv = reports / "synthesis.csv"
            write_retrieval_csv(
                retrieval_csv,
                [
                    {
                        "review_set": "retrieval_qrels",
                        "reviewer_id": "",
                        "benchmark_id": "unit_qrels",
                        "benchmark_kind": "public_qrels",
                        "query_id": "q1",
                        "query": "alpha",
                        "candidate_chunk_id": "c1",
                        "candidate_title": "Title",
                        "candidate_preview": "Preview",
                        "relevance": "",
                        "evidence_role": "",
                        "reviewer_notes": "",
                    }
                ],
            )
            write_synthesis_csv(
                synthesis_csv,
                [
                    {
                        "review_set": "synthesis_disagreement",
                        "reviewer_id": "",
                        "case_id": "case1",
                        "question": "Where is evidence contested?",
                        "claim_id": "",
                        "claim": "",
                        "evidence_id": "E1",
                        "expected_stance": "supports",
                        "citation_valid": "",
                        "stance_correct": "",
                        "claim_supported": "",
                        "missed_key_evidence": "",
                        "reviewer_action": "",
                        "reason": "",
                    }
                ],
            )
            settings = SimpleNamespace(root=root, reports_dir=reports)
            with patch("canon.product.publishable_review.load_settings", return_value=settings):
                report = build_publishable_review_status(
                    retrieval_csv=retrieval_csv,
                    synthesis_csv=synthesis_csv,
                    scaffold_path=scaffold,
                    write_report=False,
                )

        self.assertEqual(report["status"], "incomplete")
        self.assertEqual(report["reviewed_label_count"], 0)
        self.assertEqual(report["retrieval_review"]["missing_row_count"], 1)
        self.assertEqual(report["synthesis_review"]["missing_row_count"], 1)
        self.assertIn("reviewer_id", report["retrieval_review"]["incomplete_examples"][0]["missing_fields"])
        self.assertIn("reviewer_id", report["synthesis_review"]["incomplete_examples"][0]["missing_fields"])

    def test_status_rejects_invalid_values_and_requires_reason_for_problem_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reports = root / "reports"
            reports.mkdir()
            scaffold = write_scaffold(reports, retrieval_target=1, synthesis_target=1)
            retrieval_csv = reports / "retrieval.csv"
            synthesis_csv = reports / "synthesis.csv"
            write_retrieval_csv(
                retrieval_csv,
                [
                    {
                        "review_set": "retrieval_qrels",
                        "reviewer_id": "reviewer-a",
                        "benchmark_id": "unit_qrels",
                        "benchmark_kind": "public_qrels",
                        "query_id": "q1",
                        "query": "alpha",
                        "candidate_chunk_id": "c1",
                        "candidate_title": "Title",
                        "candidate_preview": "Preview",
                        "relevance": "5",
                        "evidence_role": "answer",
                        "reviewer_notes": "",
                    }
                ],
            )
            write_synthesis_csv(
                synthesis_csv,
                [
                    {
                        "review_set": "synthesis_disagreement",
                        "reviewer_id": "reviewer-a",
                        "case_id": "case1",
                        "question": "Where is evidence contested?",
                        "claim_id": "claim1",
                        "claim": "Alpha is contested.",
                        "evidence_id": "E1",
                        "expected_stance": "supports",
                        "citation_valid": "yes",
                        "stance_correct": "partial",
                        "claim_supported": "yes",
                        "missed_key_evidence": "no",
                        "reviewer_action": "revise",
                        "reason": "",
                    }
                ],
            )
            settings = SimpleNamespace(root=root, reports_dir=reports)
            with patch("canon.product.publishable_review.load_settings", return_value=settings):
                report = build_publishable_review_status(
                    retrieval_csv=retrieval_csv,
                    synthesis_csv=synthesis_csv,
                    scaffold_path=scaffold,
                    write_report=False,
                )

        self.assertEqual(report["status"], "incomplete")
        fields = {error["field"] for error in report["validation_errors"]}
        self.assertIn("relevance", fields)
        self.assertIn("evidence_role", fields)
        self.assertIn("reason", report["synthesis_review"]["incomplete_examples"][0]["missing_fields"])


def write_scaffold(reports: Path, retrieval_target: int, synthesis_target: int) -> Path:
    path = reports / "publishable_human_review_scaffold_v1.json"
    path.write_text(
        json.dumps(
            {
                "report_id": "publishable_human_review_scaffold_v1",
                "suite_id": "unit_suite",
                "targets": {
                    "minimum_public_retrieval_queries": retrieval_target,
                    "minimum_synthesis_review_questions": synthesis_target,
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def write_retrieval_csv(path: Path, rows: list[dict[str, str]]) -> None:
    write_csv(
        path,
        [
            "review_set",
            "reviewer_id",
            "benchmark_id",
            "benchmark_kind",
            "query_id",
            "query",
            "candidate_chunk_id",
            "candidate_title",
            "candidate_preview",
            "relevance",
            "evidence_role",
            "reviewer_notes",
        ],
        rows,
    )


def write_synthesis_csv(path: Path, rows: list[dict[str, str]]) -> None:
    write_csv(
        path,
        [
            "review_set",
            "reviewer_id",
            "case_id",
            "question",
            "claim_id",
            "claim",
            "evidence_id",
            "expected_stance",
            "citation_valid",
            "stance_correct",
            "claim_supported",
            "missed_key_evidence",
            "reviewer_action",
            "reason",
        ],
        rows,
    )


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
