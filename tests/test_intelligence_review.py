import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from canon.product.intelligence_review import (
    build_feedback_report,
    build_review_status_report,
    build_review_tasks,
    export_review_csv,
    import_review_csv,
    missing_review_fields,
    validate_review_record,
)


class IntelligenceReviewTests(unittest.TestCase):
    def test_build_review_tasks_preserves_human_review_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            queries_path = Path(directory) / "queries.json"
            queries_path.write_text(json.dumps([{"id": "q1", "query": "AI risk", "type": "regional"}]), encoding="utf-8")
            with patch("canon.product.intelligence_review.run_intelligence_brief", return_value=brief_report()):
                packet = build_review_tasks(
                    mode="unit",
                    queries_path=queries_path,
                    write_report=False,
                )

        self.assertEqual(packet["report_id"], "intelligence_brief_review_tasks_v1")
        self.assertEqual(len(packet["records"]), 1)
        self.assertIsNone(packet["records"][0]["review"]["usefulness_1_5"])
        self.assertEqual(
            packet["records"][0]["brief_report"]["evidence_packet_contract_validation"]["status"],
            "pass",
        )
        self.assertEqual(
            packet["records"][0]["brief_report"]["evidence_scope_summary"],
            {"private_corpus": 1, "external_source": 0},
        )
        self.assertIn("human-owned", " ".join(packet["instructions"]))

    def test_review_status_is_incomplete_until_required_fields_are_present(self):
        record = review_record()
        self.assertIn("usefulness_1_5", missing_review_fields(record))
        record["review"] = complete_review()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.json"
            path.write_text(json.dumps({"records": [record]}), encoding="utf-8")
            status = build_review_status_report(path, write_report=False)

        self.assertEqual(status["status"], "complete")
        self.assertEqual(status["reviewed_question_count"], 1)

    def test_review_validation_rejects_invalid_labels(self):
        record = review_record()
        record["review"] = complete_review()
        record["review"]["usefulness_1_5"] = 7
        record["review"]["overclaim_risk"] = "extreme"
        errors = validate_review_record(record)

        self.assertTrue(any(error["field"] == "usefulness_1_5" for error in errors))
        self.assertTrue(any(error["field"] == "overclaim_risk" for error in errors))

    def test_review_csv_export_and_import_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            records_path = Path(directory) / "records.json"
            csv_path = Path(directory) / "reviews.csv"
            output_path = Path(directory) / "completed.json"
            records_path.write_text(json.dumps({"records": [review_record()]}), encoding="utf-8")

            export_review_csv(records_path, csv_path)
            with csv_path.open("r", newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
                fieldnames = list(rows[0].keys())
            self.assertEqual(rows[0]["packet_contract_status"], "pass")
            self.assertEqual(rows[0]["external_expansion_status"], "planned")
            self.assertEqual(rows[0]["external_expansion_executed"], "False")
            self.assertEqual(json.loads(rows[0]["evidence_scope_summary"]), {"private_corpus": 1, "external_source": 0})
            rows[0].update(
                {
                    "usefulness_1_5": "5",
                    "actionability_1_5": "4",
                    "evidence_trust_1_5": "4",
                    "uncertainty_clarity_1_5": "5",
                    "missing_perspective": "no",
                    "unsupported_claim": "no",
                    "overclaim_risk": "low",
                    "final_review_status": "accepted",
                    "reviewer_notes": "Looks grounded.",
                }
            )
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

            result = import_review_csv(records_path, csv_path, output_path)
            completed = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(result["status"], "imported")
        self.assertEqual(completed["records"][0]["review"]["usefulness_1_5"], 5)
        self.assertEqual(completed["records"][0]["review"]["final_review_status"], "accepted")

    def test_review_csv_import_refuses_unknown_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            records_path = Path(directory) / "records.json"
            csv_path = Path(directory) / "reviews.csv"
            output_path = Path(directory) / "completed.json"
            records_path.write_text(json.dumps({"records": [review_record()]}), encoding="utf-8")
            export_review_csv(records_path, csv_path)
            with csv_path.open("r", newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
                fieldnames = list(rows[0].keys())
            rows[0]["id"] = "stale"
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

            result = import_review_csv(records_path, csv_path, output_path)

        self.assertEqual(result["status"], "failed_validation")
        self.assertFalse(output_path.exists())
        self.assertTrue(any(error["message"] == "CSV review id does not exist in the review packet" for error in result["validation_errors"]))

    def test_feedback_report_summarizes_labels_and_regression_candidates(self):
        record = review_record()
        record["review"] = complete_review()
        risky = review_record()
        risky["id"] = "q2"
        risky["query"] = "Water risk"
        risky["brief_report"]["evidence_packet_contract_validation"] = {"status": "fail"}
        risky["brief_report"]["evidence_scope_summary"] = {"private_corpus": 1, "external_source": 1}
        risky["brief_report"]["external_expansion"] = {"status": "planned", "executed": True}
        risky["review"] = {
            **complete_review(),
            "usefulness_1_5": 2,
            "missing_perspective": "yes",
            "unsupported_claim": "yes",
            "overclaim_risk": "high",
            "final_review_status": "needs_more_evidence",
            "reviewer_notes": "Needs local evidence.",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.json"
            path.write_text(json.dumps({"records": [record, risky]}), encoding="utf-8")
            report = build_feedback_report(path, write_report=False)

        self.assertEqual(report["report_id"], "intelligence_brief_feedback_v1")
        self.assertEqual(report["status"], "ready")
        self.assertEqual(report["metrics"]["average_usefulness_1_5"], 3.5)
        self.assertEqual(report["issue_counts"]["unsupported_claim_yes"], 1)
        self.assertEqual(report["metrics"]["final_review_status_counts"]["needs_more_evidence"], 1)
        self.assertEqual(
            report["packet_metadata_summary"]["evidence_scope_totals"],
            {"private_corpus": 2, "external_source": 1},
        )
        self.assertEqual(report["packet_metadata_summary"]["packet_contract_status_counts"], {"pass": 1, "fail": 1})
        self.assertEqual(report["packet_metadata_summary"]["external_expansion_status_counts"], {"planned": 2})
        self.assertEqual(report["packet_metadata_summary"]["external_expansion_executed_count"], 1)
        self.assertEqual(report["regression_candidates"][0]["id"], "q2")
        self.assertIn("unsupported_claim", report["regression_candidates"][0]["reasons"])
        self.assertIn("packet_contract_not_passed", report["regression_candidates"][0]["reasons"])
        self.assertIn("external_expansion_executed", report["regression_candidates"][0]["reasons"])


def review_record():
    return {
        "id": "q1",
        "type": "regional",
        "query": "AI risk",
        "brief_report": brief_report(),
        "review": {
            "usefulness_1_5": None,
            "actionability_1_5": None,
            "evidence_trust_1_5": None,
            "uncertainty_clarity_1_5": None,
            "missing_perspective": None,
            "unsupported_claim": None,
            "overclaim_risk": None,
            "final_review_status": None,
            "reviewer_notes": "",
        },
    }


def complete_review():
    return {
        "usefulness_1_5": 5,
        "actionability_1_5": 4,
        "evidence_trust_1_5": 4,
        "uncertainty_clarity_1_5": 5,
        "missing_perspective": "no",
        "unsupported_claim": "no",
        "overclaim_risk": "low",
        "final_review_status": "accepted",
        "reviewer_notes": "",
    }


def brief_report():
    return {
        "status": "ready_for_human_review",
        "question": "AI risk",
        "evidence_packet_request_id": "req-ai-risk",
        "evidence_packet_contract_validation": {"status": "pass"},
        "evidence_scope_summary": {"private_corpus": 1, "external_source": 0},
        "external_expansion": {"status": "planned", "executed": False},
        "grounding_report": {"grounded_claim_ratio": 1.0},
        "duplicate_agent_output": {"duplicate_agent_output_rate": 0.0},
        "red_team": {"objections": []},
        "report_quality": {
            "checks": [
                {"id": "minimum_agent_contribution", "details": {"completed_agents": 5}},
            ]
        },
        "brief": {
            "executive_summary": "Summary",
            "what_changed": ["Change"],
            "source_gaps": [],
            "contradictions_and_uncertainty": "Uncertainty",
            "next_signals_to_watch": ["Signal"],
            "citation_appendix": [{"evidence_id": "C1", "title": "Evidence", "source_name": "Source"}],
        },
    }


if __name__ == "__main__":
    unittest.main()
