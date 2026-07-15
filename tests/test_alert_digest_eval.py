import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from canon.eval.alert_digest import evaluate_alert_digests, summarize_rows


class AlertDigestEvalTests(unittest.TestCase):
    def test_evaluate_alert_digests_passes_when_seed_alerts_are_well_formed(self):
        with tempfile.TemporaryDirectory() as directory:
            queries_path = Path(directory) / "queries.json"
            queries_path.write_text(json.dumps([{"id": "q1", "query": "AI risk"}]), encoding="utf-8")
            with patch("canon.eval.alert_digest.run_alert_digest", return_value=digest_report()):
                report = evaluate_alert_digests(mode="unit", queries_path=queries_path, write_report=False)

        self.assertEqual(report["report_id"], "alert_digest_eval_v1")
        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["summary"]["total_alert_count"], 1)

    def test_evaluate_alert_digests_marks_review_on_bad_alerts(self):
        digest = digest_report()
        digest["alerts"][0]["evidence_trigger"] = ""
        with tempfile.TemporaryDirectory() as directory:
            queries_path = Path(directory) / "queries.json"
            queries_path.write_text(json.dumps([{"id": "q1", "query": "AI risk"}]), encoding="utf-8")
            with patch("canon.eval.alert_digest.run_alert_digest", return_value=digest):
                report = evaluate_alert_digests(mode="unit", queries_path=queries_path, write_report=False)

        self.assertEqual(report["status"], "review")
        self.assertFalse(report["summary"]["all_alert_gates_pass"])

    def test_summarize_rows_aggregates_alert_metrics(self):
        summary = summarize_rows(
            [
                {"status": "pass", "metrics": {"alert_count": 2, "duplicate_alert_rate": 0.0}},
                {"status": "review", "metrics": {"alert_count": 1, "duplicate_alert_rate": 0.2}},
            ]
        )

        self.assertEqual(summary["query_count"], 2)
        self.assertEqual(summary["pass_count"], 1)
        self.assertEqual(summary["total_alert_count"], 3)
        self.assertEqual(summary["max_duplicate_alert_rate"], 0.2)


def digest_report():
    return {
        "status": "ready_for_human_review",
        "alert_count": 1,
        "duplicate_alerts": {"duplicate_alert_rate": 0.0},
        "quality": {"status": "pass"},
        "alerts": [
            {
                "alert_id": "alert_001",
                "severity": "high",
                "confidence": "medium",
                "issue": "grid",
                "affected_regions": ["Chile"],
                "affected_entities": ["Chile Energy Regulator"],
                "evidence_trigger": "Regulator evidence",
                "evidence_ids": ["C1"],
                "uncertainty": "Needs review",
                "recommended_follow_up": "Check official filings",
            }
        ],
    }


if __name__ == "__main__":
    unittest.main()
