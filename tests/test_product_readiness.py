import unittest
import tomllib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from tempfile import TemporaryDirectory

from canon.product.readiness import build_readiness_report, endpoints


class ProductReadinessTests(unittest.TestCase):
    def test_readiness_report_passes_for_current_bundle(self):
        with TemporaryDirectory() as temp_dir:
            with patch("canon.product.readiness.load_settings", return_value=SimpleNamespace(reports_dir=Path(temp_dir))):
                with patch("canon.product.readiness.load_json", return_value={"status": "pass"}):
                    report = build_readiness_report("social_science_ir_v1_harvest10")
        self.assertEqual(report["status"], "pass")
        self.assertIn("POST /v1/answer", report["endpoints"])
        self.assertIn("POST /v1/projects/start", report["endpoints"])
        self.assertIn("POST /v1/evidence-packets", report["endpoints"])
        self.assertIn("POST /v1/frame-coverage", report["endpoints"])
        self.assertIn("POST /v1/query-diagnostics", report["endpoints"])
        self.assertIn("POST /v1/intelligence-brief", report["endpoints"])
        self.assertIn("POST /v1/report-quality", report["endpoints"])
        self.assertIn("POST /v1/alert-digest", report["endpoints"])
        self.assertIn("POST /v1/flagship-handoff", report["endpoints"])
        self.assertIn("POST /v1/intelligence-review/prepare", report["endpoints"])
        self.assertGreater(report["route_metadata"]["route_count"], 0)
        self.assertTrue(report["route_metadata"]["human_review_boundary"])
        self.assertIn("project_start_endpoint_documented", [check["id"] for check in report["checks"]])
        self.assertIn("evidence_packets_endpoint_documented", [check["id"] for check in report["checks"]])
        self.assertIn("frame_coverage_endpoint_documented", [check["id"] for check in report["checks"]])
        self.assertIn("report_quality_endpoint_documented", [check["id"] for check in report["checks"]])
        self.assertIn("routes_metadata_examples_present", [check["id"] for check in report["checks"]])
        self.assertIn("product_smoke_passed", [check["id"] for check in report["checks"]])

    def test_readiness_requires_passing_smoke_report(self):
        with TemporaryDirectory() as temp_dir:
            with patch("canon.product.readiness.load_settings", return_value=SimpleNamespace(reports_dir=Path(temp_dir))):
                with patch("canon.product.readiness.load_json", return_value={"status": "fail"}):
                    report = build_readiness_report("social_science_ir_v1_harvest10")
        self.assertEqual(report["status"], "fail")
        self.assertIn(
            {"id": "product_smoke_passed", "passed": False},
            report["checks"],
        )

    def test_endpoints_are_derived_from_route_metadata(self):
        current = endpoints()
        self.assertIn("GET /v1/routes", current)
        self.assertIn("POST /v1/intelligence-brief/evaluate", current)
        self.assertIn("POST /v1/alert-digest/evaluate", current)

    def test_release_console_scripts_are_registered(self):
        pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
        scripts = pyproject["project"]["scripts"]
        self.assertEqual(scripts["canon-project"], "canon.product.project:main")
        self.assertEqual(scripts["canon-frame-coverage"], "canon.product.frame_coverage:main")
        self.assertEqual(scripts["canon-report-quality"], "canon.product.report_quality:main")
        self.assertEqual(scripts["canon-readiness"], "canon.product.readiness:main")
        self.assertEqual(scripts["canon-release-audit"], "canon.product.release_audit:main")
        self.assertEqual(scripts["canon-product-final-check"], "canon.product.final_check:main")


if __name__ == "__main__":
    unittest.main()
