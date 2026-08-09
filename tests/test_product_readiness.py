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
            settings = SimpleNamespace(root=Path.cwd(), reports_dir=Path(temp_dir))
            with patch("canon.product.readiness.load_settings", return_value=settings):
                with patch("canon.product.readiness.load_json", return_value={"status": "pass"}):
                    report = build_readiness_report("social_science_ir_v1_harvest10")
        self.assertEqual(report["status"], "pass")
        self.assertIn("GET /app", report["endpoints"])
        self.assertIn("GET /v1/production/status", report["endpoints"])
        self.assertIn("POST /v1/production/evidence-workbench", report["endpoints"])
        self.assertIn("POST /v1/production/feedback", report["endpoints"])
        self.assertIn("POST /v1/production/corpus-setup", report["endpoints"])
        self.assertIn("POST /v1/production/corpus-refresh", report["endpoints"])
        self.assertIn("GET /metrics", report["endpoints"])
        self.assertIn("POST /v1/evidence-packets", report["endpoints"])
        self.assertIn("POST /v1/stage2-synthesis", report["endpoints"])
        self.assertEqual(report["serving_target"], "asgi")
        self.assertGreater(report["route_metadata"]["route_count"], 0)
        self.assertTrue(report["route_metadata"]["human_review_boundary"])
        self.assertIn("evidence_packets_endpoint_documented", [check["id"] for check in report["checks"]])
        self.assertIn("stage2_synthesis_endpoint_documented", [check["id"] for check in report["checks"]])
        self.assertIn("routes_metadata_descriptions_present", [check["id"] for check in report["checks"]])
        self.assertIn("serving_target_is_asgi", [check["id"] for check in report["checks"]])
        self.assertIn("browser_workbench_endpoint_documented", [check["id"] for check in report["checks"]])
        self.assertIn("production_status_endpoint_documented", [check["id"] for check in report["checks"]])
        self.assertIn("production_workbench_endpoint_documented", [check["id"] for check in report["checks"]])
        self.assertIn("production_feedback_endpoint_documented", [check["id"] for check in report["checks"]])
        self.assertIn("production_corpus_setup_endpoint_documented", [check["id"] for check in report["checks"]])
        self.assertIn("production_corpus_refresh_endpoint_documented", [check["id"] for check in report["checks"]])
        self.assertIn("product_smoke_passed", [check["id"] for check in report["checks"]])
        self.assertEqual(report["package_metadata"]["name"], "canon-rag")
        self.assertEqual(report["package_metadata"]["readme"], "README.md")
        self.assertEqual(report["package_metadata"]["urls"]["Repository"], "https://github.com/Saltef/CANON")
        self.assertIn(
            "package_metadata_has_public_urls",
            [check["id"] for check in report["checks"]],
        )

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
        self.assertIn("GET /app", current)
        self.assertIn("POST /v1/production/evidence-workbench", current)
        self.assertIn("POST /v1/production/corpus-setup", current)
        self.assertIn("POST /v1/production/corpus-refresh", current)
        self.assertIn("GET /v1/routes", current)
        self.assertIn("POST /v1/stage2-synthesis", current)
        self.assertNotIn("POST /v1/intelligence-brief/evaluate", current)
        self.assertNotIn("POST /v1/alert-digest/evaluate", current)

    def test_release_console_scripts_are_registered(self):
        pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
        scripts = pyproject["project"]["scripts"]
        self.assertEqual(scripts["canon-acceptance-scenario"], "canon.product.acceptance_scenario:main")
        self.assertEqual(scripts["canon-project"], "canon.product.project:main")
        self.assertEqual(scripts["canon-frame-coverage"], "canon.product.frame_coverage:main")
        self.assertEqual(scripts["canon-report-quality"], "canon.product.report_quality:main")
        self.assertEqual(scripts["canon-readiness"], "canon.product.readiness:main")
        self.assertEqual(scripts["canon-release-audit"], "canon.product.release_audit:main")
        self.assertEqual(scripts["canon-product-final-check"], "canon.product.final_check:main")

    def test_public_package_metadata_is_registered(self):
        pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
        project = pyproject["project"]
        self.assertEqual(project["readme"], "README.md")
        self.assertEqual(project["urls"]["Repository"], "https://github.com/Saltef/CANON")
        self.assertEqual(project["urls"]["Documentation"], "https://github.com/Saltef/CANON#readme")
        self.assertIn("human-review", project["keywords"])
        self.assertIn("Programming Language :: Python :: 3.12", project["classifiers"])
        self.assertIn("Development Status :: 3 - Alpha", project["classifiers"])


if __name__ == "__main__":
    unittest.main()
