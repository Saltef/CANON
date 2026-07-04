import unittest

from canon.product.readiness import build_readiness_report


class ProductReadinessTests(unittest.TestCase):
    def test_readiness_report_passes_for_current_bundle(self):
        report = build_readiness_report("social_science_ir_v1_harvest10")
        self.assertEqual(report["status"], "pass")
        self.assertIn("POST /v1/answer", report["endpoints"])


if __name__ == "__main__":
    unittest.main()
