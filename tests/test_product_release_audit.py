import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from canon.product.release_audit import build_release_audit, write_json


class ProductReleaseAuditTests(unittest.TestCase):
    def test_release_audit_blocks_when_human_review_is_incomplete(self):
        with TemporaryDirectory() as temp_dir:
            write_source_reports(Path(temp_dir), review_status="incomplete", industry_status="fail")
            with patch(
                "canon.product.release_audit.load_settings",
                return_value=SimpleNamespace(reports_dir=Path(temp_dir)),
            ):
                report = build_release_audit(write_report=False)
        self.assertEqual(report["status"], "blocked")
        self.assertEqual(report["components"][0]["id"], "source_report_integrity")
        self.assertTrue(report["components"][0]["passed"])
        self.assertIn("Automated product gates pass", report["claim"])
        self.assertIn("Complete human review labels", report["next_required_action"])

    def test_release_audit_passes_when_all_components_pass(self):
        with TemporaryDirectory() as temp_dir:
            write_source_reports(Path(temp_dir), review_status="complete", industry_status="pass")
            with patch(
                "canon.product.release_audit.load_settings",
                return_value=SimpleNamespace(reports_dir=Path(temp_dir)),
            ):
                report = build_release_audit(write_report=False)
        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["next_required_action"], "No required action remains.")

    def test_release_audit_blocks_wrong_mode_source_report(self):
        with TemporaryDirectory() as temp_dir:
            reports_dir = Path(temp_dir)
            write_source_reports(reports_dir, review_status="complete", industry_status="pass")
            mode = "social_science_ir_v1_harvest10"
            write_json(
                reports_dir / f"product_smoke_{mode}.json",
                {"report_id": "product_smoke_v1", "mode": "other_mode", "status": "pass"},
            )
            with patch(
                "canon.product.release_audit.load_settings",
                return_value=SimpleNamespace(reports_dir=reports_dir),
            ):
                report = build_release_audit(write_report=False)
        integrity = report["components"][0]
        self.assertEqual(report["status"], "blocked")
        self.assertFalse(integrity["passed"])
        self.assertEqual(report["claim"], "Product release is not ready: source reports are missing or inconsistent.")
        self.assertTrue(any(error["field"] == "mode" for error in integrity["details"]["errors"]))

    def test_release_audit_blocks_missing_source_report(self):
        with TemporaryDirectory() as temp_dir:
            reports_dir = Path(temp_dir)
            write_source_reports(reports_dir, review_status="complete", industry_status="pass")
            (reports_dir / "human_review_status_v1.json").unlink()
            with patch(
                "canon.product.release_audit.load_settings",
                return_value=SimpleNamespace(reports_dir=reports_dir),
            ):
                report = build_release_audit(write_report=False)
        integrity = report["components"][0]
        self.assertEqual(report["status"], "blocked")
        self.assertFalse(integrity["passed"])
        self.assertIn("Regenerate missing", report["next_required_action"])
        self.assertTrue(any(error["message"] == "source report is missing" for error in integrity["details"]["errors"]))

    def test_release_audit_blocks_unreadable_source_report_without_crashing(self):
        with TemporaryDirectory() as temp_dir:
            reports_dir = Path(temp_dir)
            write_source_reports(reports_dir, review_status="complete", industry_status="pass")
            mode = "social_science_ir_v1_harvest10"
            (reports_dir / f"product_readiness_{mode}.json").write_text("", encoding="utf-8")
            with patch(
                "canon.product.release_audit.load_settings",
                return_value=SimpleNamespace(reports_dir=reports_dir),
            ):
                report = build_release_audit(write_report=False)
        integrity = report["components"][0]
        self.assertEqual(report["status"], "blocked")
        self.assertFalse(integrity["passed"])
        self.assertTrue(any(error["field"] == "json" for error in integrity["details"]["errors"]))

    def test_release_audit_blocks_contradictory_complete_human_review_status(self):
        with TemporaryDirectory() as temp_dir:
            reports_dir = Path(temp_dir)
            write_source_reports(reports_dir, review_status="complete", industry_status="pass")
            mode = "social_science_ir_v1_harvest10"
            write_json(
                reports_dir / "human_review_status_v1.json",
                {
                    "report_id": "human_review_status_v1",
                    "mode": mode,
                    "status": "complete",
                    "reviewed_question_count": 30,
                    "minimum_question_count": 30,
                    "missing_field_count": 1,
                    "validation_error_count": 0,
                },
            )
            with patch(
                "canon.product.release_audit.load_settings",
                return_value=SimpleNamespace(reports_dir=reports_dir),
            ):
                report = build_release_audit(write_report=False)
        review_component = next(item for item in report["components"] if item["id"] == "human_review_status")
        self.assertEqual(report["status"], "blocked")
        self.assertFalse(review_component["passed"])
        self.assertEqual(review_component["details"]["missing_field_count"], 1)

    def test_release_audit_blocks_complete_human_review_status_with_validation_errors(self):
        with TemporaryDirectory() as temp_dir:
            reports_dir = Path(temp_dir)
            write_source_reports(reports_dir, review_status="complete", industry_status="pass")
            mode = "social_science_ir_v1_harvest10"
            write_json(
                reports_dir / "human_review_status_v1.json",
                {
                    "report_id": "human_review_status_v1",
                    "mode": mode,
                    "status": "complete",
                    "reviewed_question_count": 30,
                    "minimum_question_count": 30,
                    "missing_field_count": 0,
                    "validation_error_count": 1,
                },
            )
            with patch(
                "canon.product.release_audit.load_settings",
                return_value=SimpleNamespace(reports_dir=reports_dir),
            ):
                report = build_release_audit(write_report=False)
        review_component = next(item for item in report["components"] if item["id"] == "human_review_status")
        self.assertEqual(report["status"], "blocked")
        self.assertFalse(review_component["passed"])
        self.assertEqual(review_component["details"]["validation_error_count"], 1)


def write_source_reports(path: Path, review_status: str, industry_status: str) -> None:
    mode = "social_science_ir_v1_harvest10"
    write_json(path / f"product_smoke_{mode}.json", {"report_id": "product_smoke_v1", "mode": mode, "status": "pass"})
    write_json(path / f"product_readiness_{mode}.json", {"mode": mode, "status": "pass"})
    reviewed = 30 if review_status == "complete" else 0
    write_json(
        path / "human_review_status_v1.json",
        {
            "report_id": "human_review_status_v1",
            "mode": mode,
            "status": review_status,
            "reviewed_question_count": reviewed,
            "minimum_question_count": 30,
            "missing_field_count": 0 if review_status == "complete" else 300,
            "validation_error_count": 0,
        },
    )
    write_json(
        path / "industry_pilot_acceptance_v1.json",
        {
            "report_id": "industry_pilot_acceptance_v1",
            "mode": mode,
            "status": industry_status,
            "release_blockers": [] if industry_status == "pass" else ["acceptance_question_count"],
            "reviewed_question_count": reviewed,
            "minimum_question_count": 30,
        },
    )


if __name__ == "__main__":
    unittest.main()
