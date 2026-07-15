import unittest
import tomllib
from pathlib import Path
from unittest.mock import patch

from canon.product.demo import compact_demo_summary, run_demo


class ProductDemoTests(unittest.TestCase):
    def test_compact_demo_summary_keeps_handoff_boundary_visible(self):
        summary = compact_demo_summary(handoff_report())

        self.assertEqual(summary["status"], "automated_pass_human_review_required")
        self.assertEqual(summary["human_review"]["status"], "incomplete")
        self.assertEqual(summary["human_review"]["question_count"], 5)
        self.assertEqual(summary["automated_steps"][0]["id"], "ingest_fixture")
        self.assertEqual(len(summary["key_artifacts"]), 2)
        self.assertTrue(all("path" in artifact for artifact in summary["key_artifacts"]))

    def test_run_demo_wraps_flagship_handoff(self):
        with patch("canon.product.demo.run_flagship_handoff", return_value=handoff_report()) as handoff:
            summary = run_demo(
                mode="demo",
                question="What changed?",
                fixture_path=Path("data/fixtures/ai_infra_geo_risk_sample.jsonl"),
                queries_path=Path("gold/ai_infra_geo_risk_seed_queries.json"),
                write_report=False,
            )

        self.assertEqual(summary["status"], "automated_pass_human_review_required")
        handoff.assert_called_once()
        self.assertEqual(handoff.call_args.kwargs["mode"], "demo")
        self.assertEqual(handoff.call_args.kwargs["question"], "What changed?")
        self.assertFalse(handoff.call_args.kwargs["write_report"])

    def test_canon_demo_console_script_is_registered(self):
        pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
        self.assertEqual(pyproject["project"]["scripts"]["canon-demo"], "canon.product.demo:main")


def handoff_report():
    return {
        "status": "automated_pass_human_review_required",
        "mode": "ai_infra_geo_risk_demo",
        "policy": "rag",
        "question": "What are the emerging geopolitical risks?",
        "steps": [
            {"id": "ingest_fixture", "status": "ok", "passed": True},
            {"id": "human_review_status", "status": "incomplete", "passed": False},
        ],
        "human_review_boundary": {
            "required": True,
            "status": "incomplete",
            "reviewed_question_count": 0,
            "question_count": 5,
            "next_required_action": "Complete human review labels for 5 intelligence brief review task(s).",
        },
        "artifacts": [
            {
                "id": "flagship_handoff_json",
                "path": "reports/flagship_handoff_ai_infra_geo_risk_demo.json",
                "exists": True,
                "sha256": "abc",
            },
            {
                "id": "intelligence_review_tasks_json",
                "path": "reports/intelligence_brief_review_tasks_ai_infra_geo_risk_demo.json",
                "exists": True,
                "sha256": "def",
            },
            {
                "id": "non_key_artifact",
                "path": "reports/extra.json",
                "exists": True,
                "sha256": "ghi",
            },
        ],
        "claim": "Flagship handoff passed automated gates; human review remains required.",
        "limitations": ["Human review remains required."],
    }


if __name__ == "__main__":
    unittest.main()
