import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from tempfile import TemporaryDirectory

from canon.product.readiness import build_readiness_report


class ProductReadinessTests(unittest.TestCase):
    def test_readiness_report_passes_for_current_bundle(self):
        with TemporaryDirectory() as temp_dir:
            with patch("canon.product.readiness.load_settings", return_value=SimpleNamespace(reports_dir=Path(temp_dir))):
                with patch("canon.product.readiness.load_json", return_value={"status": "pass"}):
                    report = build_readiness_report("social_science_ir_v1_harvest10")
        self.assertEqual(report["status"], "pass")
        self.assertIn("POST /v1/answer", report["endpoints"])
        self.assertIn("POST /v1/evidence-packets", report["endpoints"])
        self.assertIn("POST /v1/query-diagnostics", report["endpoints"])
        self.assertIn("evidence_packets_endpoint_documented", [check["id"] for check in report["checks"]])
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


if __name__ == "__main__":
    unittest.main()
