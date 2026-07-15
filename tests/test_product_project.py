import tempfile
import unittest
from pathlib import Path

from canon.product.project import build_project_config


class ProductProjectTests(unittest.TestCase):
    def test_project_config_requires_source_boundary_for_ready_status(self):
        config = build_project_config(
            project_name="AI Infrastructure Geopolitical Risk",
            domain="AI infrastructure and geopolitical risk",
            regions=["Latin America"],
            languages=["English", "Spanish"],
            issue_categories=["energy demand"],
            desired_report_types=["weekly_intelligence_brief"],
            source_boundaries=[],
            write_report=False,
        )

        self.assertEqual(config["status"], "needs_source_boundary")
        self.assertFalse(config["source_plan"]["monitor_ready"])
        self.assertEqual(config["monitoring"]["blocked_reason"], "source_boundary_required")
        self.assertFalse(next(row for row in config["validation"]["checks"] if row["id"] == "source_boundary_declared")["passed"])

    def test_project_config_is_ready_with_declared_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            config = build_project_config(
                project_name="AI Infrastructure Geopolitical Risk",
                project_id="ai_infra_geo_risk",
                domain="AI infrastructure and geopolitical risk",
                regions=["Latin America", "Brazil"],
                languages=["English", "Portuguese"],
                issue_categories=["energy demand", "water and cooling"],
                desired_report_types=["weekly_intelligence_brief", "alert_digest"],
                source_boundaries=["G:/My Drive/CANON Corpus"],
                corpus_id="ai_infra_geo_risk_corpus",
                output_dir=Path(directory),
            )

            output_path = Path(directory) / "project_config.json"
            self.assertTrue(output_path.exists())

        self.assertEqual(config["status"], "ready")
        self.assertTrue(config["source_plan"]["monitor_ready"])
        self.assertEqual(config["source_plan"]["corpus_id"], "ai_infra_geo_risk_corpus")
        self.assertIn("data centers", config["ontology"]["entities"]["infrastructure"])


if __name__ == "__main__":
    unittest.main()
