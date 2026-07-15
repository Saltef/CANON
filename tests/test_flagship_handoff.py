import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from canon.product.flagship_handoff import handoff_status, run_flagship_handoff, step_results


class FlagshipHandoffTests(unittest.TestCase):
    def test_handoff_status_preserves_human_review_boundary(self):
        self.assertEqual(handoff_status(True, True), "pass")
        self.assertEqual(handoff_status(True, False), "automated_pass_human_review_required")
        self.assertEqual(handoff_status(False, False), "fail")

    def test_step_results_marks_human_review_incomplete_as_not_passed(self):
        steps = step_results(
            ingest_report={"chunk_count": 3},
            brief_report={"status": "ready_for_human_review"},
            brief_eval={"status": "pass"},
            alert_digest={"status": "ready_for_human_review"},
            alert_eval={"status": "pass"},
            review_packet={"records": [{"id": "q1"}]},
            review_status={"status": "incomplete", "question_count": 1, "reviewed_question_count": 0},
        )

        self.assertTrue(all(step["passed"] for step in steps if step["id"] != "human_review_status"))
        self.assertFalse(next(step for step in steps if step["id"] == "human_review_status")["passed"])

    def test_run_flagship_handoff_orchestrates_automated_handoff(self):
        with tempfile.TemporaryDirectory() as directory:
            reports_dir = Path(directory) / "reports"
            reports_dir.mkdir()
            with patch.dict("os.environ", {"CANON_REPORTS_DIR": str(reports_dir)}):
                with patch("canon.product.flagship_handoff.ingest_unstructured_jsonl", return_value={"chunk_count": 3, "work_count": 2, "limitations": []}):
                    with patch("canon.product.flagship_handoff.run_intelligence_brief", return_value=brief_report()):
                        with patch("canon.product.flagship_handoff.evaluate_intelligence_briefs", return_value={"status": "pass", "summary": {"pass_rate": 1.0}}):
                            with patch("canon.product.flagship_handoff.run_alert_digest", return_value={"status": "ready_for_human_review", "alert_count": 2, "duplicate_alerts": {"duplicate_alert_rate": 0.0}}):
                                with patch("canon.product.flagship_handoff.evaluate_alert_digests", return_value={"status": "pass", "summary": {"pass_rate": 1.0}}):
                                    with patch("canon.product.flagship_handoff.build_review_tasks", return_value={"records": [{"id": "q1"}]}):
                                        report = run_flagship_handoff(
                                            fixture_path=Path("data/fixtures/ai_infra_geo_risk_sample.jsonl"),
                                            queries_path=Path("gold/ai_infra_geo_risk_seed_queries.json"),
                                            write_report=False,
                                        )

        self.assertEqual(report["status"], "automated_pass_human_review_required")
        self.assertEqual(report["human_review_boundary"]["reviewed_question_count"], 0)
        self.assertIn("human review", report["claim"].lower())

    def test_written_handoff_fingerprints_itself(self):
        with tempfile.TemporaryDirectory() as directory:
            reports_dir = Path(directory) / "reports"
            reports_dir.mkdir()
            with patch.dict("os.environ", {"CANON_REPORTS_DIR": str(reports_dir)}):
                with patch("canon.product.flagship_handoff.ingest_unstructured_jsonl", return_value={"chunk_count": 3, "work_count": 2, "limitations": []}):
                    with patch("canon.product.flagship_handoff.run_intelligence_brief", return_value=brief_report()):
                        with patch("canon.product.flagship_handoff.evaluate_intelligence_briefs", return_value={"status": "pass", "summary": {"pass_rate": 1.0}}):
                            with patch("canon.product.flagship_handoff.run_alert_digest", return_value={"status": "ready_for_human_review", "alert_count": 2, "duplicate_alerts": {"duplicate_alert_rate": 0.0}}):
                                with patch("canon.product.flagship_handoff.evaluate_alert_digests", return_value={"status": "pass", "summary": {"pass_rate": 1.0}}):
                                    with patch("canon.product.flagship_handoff.build_review_tasks", return_value={"records": [{"id": "q1"}]}):
                                        with patch("canon.product.flagship_handoff.build_review_handoff", return_value={"status": "ready_for_human_review"}):
                                            report = run_flagship_handoff(write_report=True)

        self_artifact = next(item for item in report["artifacts"] if item["path"].endswith("flagship_handoff_ai_infra_geo_risk_demo.json"))
        self.assertTrue(self_artifact["exists"])
        self.assertEqual(self_artifact["id"], "flagship_handoff_ai_infra_geo_risk_demo_json")
        self.assertEqual(len(self_artifact["sha256"]), 64)


def brief_report():
    return {
        "status": "ready_for_human_review",
        "grounding_report": {"status": "pass"},
        "report_quality": {"status": "pass"},
        "agent_outputs": [{"status": "complete"} for _ in range(5)],
    }


if __name__ == "__main__":
    unittest.main()
