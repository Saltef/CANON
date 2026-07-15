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
            write_public_repo_files(Path(temp_dir))
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
            write_public_repo_files(Path(temp_dir))
            with patch(
                "canon.product.release_audit.load_settings",
                return_value=SimpleNamespace(reports_dir=Path(temp_dir)),
            ):
                report = build_release_audit(write_report=False)
        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["next_required_action"], "No required action remains.")

    def test_release_audit_blocks_public_release_when_license_is_missing(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_source_reports(root, review_status="complete", industry_status="pass")
            write_public_repo_files(root, include_license=False)
            with patch(
                "canon.product.release_audit.load_settings",
                return_value=SimpleNamespace(root=root, reports_dir=root),
            ):
                report = build_release_audit(write_report=False)
        hygiene = next(item for item in report["components"] if item["id"] == "public_repository_hygiene")
        self.assertEqual(report["status"], "blocked")
        self.assertFalse(hygiene["passed"])
        self.assertIn("public repository hygiene", report["claim"])
        self.assertIn("Choose a license", report["next_required_action"])
        failed = [item["id"] for item in hygiene["details"]["checks"] if not item["passed"]]
        self.assertIn("license_file_present", failed)
        self.assertIn("package_license_declared", failed)

    def test_release_audit_uses_intelligence_review_track_for_ai_demo(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            mode = "ai_infra_geo_risk_demo"
            write_public_repo_files(root)
            write_json(root / f"product_smoke_{mode}.json", {"report_id": "product_smoke_v1", "mode": mode, "status": "pass"})
            write_json(root / f"product_readiness_{mode}.json", {"mode": mode, "status": "pass"})
            write_json(
                root / "intelligence_brief_review_status_v1.json",
                {
                    "report_id": "intelligence_brief_review_status_v1",
                    "status": "incomplete",
                    "question_count": 5,
                    "reviewed_question_count": 0,
                    "minimum_question_count": 5,
                    "missing_field_count": 40,
                    "validation_error_count": 0,
                },
            )
            write_json(
                root / "intelligence_brief_feedback_v1.json",
                {
                    "report_id": "intelligence_brief_feedback_v1",
                    "status": "needs_review_labels",
                    "record_count": 5,
                    "reviewed_record_count": 0,
                    "regression_candidates": [],
                },
            )
            with patch(
                "canon.product.release_audit.load_settings",
                return_value=SimpleNamespace(root=root, reports_dir=root),
            ):
                report = build_release_audit(mode=mode, write_report=False)

        self.assertEqual(report["acceptance_track"], "intelligence")
        self.assertIn("intelligence review", report["claim"])
        self.assertIn("5 remaining", report["next_required_action"])
        self.assertTrue(any(item["id"] == "intelligence_feedback" for item in report["components"]))

    def test_release_audit_blocks_wrong_mode_source_report(self):
        with TemporaryDirectory() as temp_dir:
            reports_dir = Path(temp_dir)
            write_source_reports(reports_dir, review_status="complete", industry_status="pass")
            write_public_repo_files(reports_dir)
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
            write_public_repo_files(reports_dir)
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
            write_public_repo_files(reports_dir)
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
            write_public_repo_files(reports_dir)
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
            write_public_repo_files(reports_dir)
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


def write_public_repo_files(path: Path, include_license: bool = True) -> None:
    (path / "README.md").write_text("# CANON\n", encoding="utf-8")
    (path / "SECURITY.md").write_text("# Security Policy\n", encoding="utf-8")
    (path / ".env.example").write_text("OPENAI_API_KEY=\nCOHERE_API_KEY=\n", encoding="utf-8")
    (path / ".gitignore").write_text(".env\n.env.*\n!.env.example\n", encoding="utf-8")
    if include_license:
        (path / "LICENSE").write_text("Test license fixture.\n", encoding="utf-8")
        license_metadata = 'license = { text = "Test license fixture" }\n'
    else:
        license_metadata = ""
    (path / "pyproject.toml").write_text(
        "[project]\n"
        'name = "canon-rag"\n'
        'version = "0.1.0"\n'
        f"{license_metadata}",
        encoding="utf-8",
    )


if __name__ == "__main__":
    unittest.main()
