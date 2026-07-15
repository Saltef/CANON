import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from canon.eval.intelligence_brief import evaluate_intelligence_briefs, summarize_rows


class IntelligenceBriefEvalTests(unittest.TestCase):
    def test_evaluate_intelligence_briefs_summarizes_release_gates(self):
        with tempfile.TemporaryDirectory() as directory:
            queries_path = Path(directory) / "queries.json"
            queries_path.write_text(
                json.dumps(
                    [
                        {"id": "q1", "query": "AI data center risk", "type": "regional"},
                        {"id": "q2", "query": "Water risk", "type": "water"},
                    ]
                ),
                encoding="utf-8",
            )
            with patch(
                "canon.eval.intelligence_brief.run_intelligence_brief",
                side_effect=[passing_report("q1"), passing_report("q2")],
            ) as runner:
                report = evaluate_intelligence_briefs(
                    mode="unit",
                    queries_path=queries_path,
                    write_report=False,
                )

        self.assertEqual(report["report_id"], "intelligence_brief_eval_v1")
        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["summary"]["query_count"], 2)
        self.assertEqual(report["summary"]["pass_rate"], 1.0)
        self.assertTrue(report["summary"]["all_release_gates_pass"])
        self.assertEqual(runner.call_count, 2)

    def test_evaluate_intelligence_briefs_marks_review_when_gate_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            queries_path = Path(directory) / "queries.json"
            queries_path.write_text(json.dumps([{"id": "q1", "query": "AI data center risk"}]), encoding="utf-8")
            with patch("canon.eval.intelligence_brief.run_intelligence_brief", return_value=failing_report()):
                report = evaluate_intelligence_briefs(
                    mode="unit",
                    queries_path=queries_path,
                    write_report=False,
                )

        self.assertEqual(report["status"], "review")
        self.assertFalse(report["summary"]["all_release_gates_pass"])
        self.assertEqual(report["queries"][0]["status"], "review")

    def test_summarize_rows_aggregates_metrics(self):
        summary = summarize_rows(
            [
                {
                    "status": "pass",
                    "metrics": {
                        "grounded_claim_ratio": 1.0,
                        "completed_agents": 5,
                        "duplicate_agent_output_rate": 0.0,
                        "red_team_blockers": 0,
                        "red_team_review_issues": 1,
                    },
                },
                {
                    "status": "review",
                    "metrics": {
                        "grounded_claim_ratio": 0.5,
                        "completed_agents": 3,
                        "duplicate_agent_output_rate": 0.4,
                        "red_team_blockers": 1,
                        "red_team_review_issues": 0,
                    },
                },
            ]
        )

        self.assertEqual(summary["query_count"], 2)
        self.assertEqual(summary["pass_count"], 1)
        self.assertEqual(summary["pass_rate"], 0.5)
        self.assertEqual(summary["average_completed_agents"], 4.0)
        self.assertEqual(summary["red_team_blocker_count"], 1)


def passing_report(query_id):
    return {
        "status": "ready_for_human_review",
        "agent_outputs": [{"status": "complete"} for _ in range(5)],
        "grounding_report": {"status": "pass", "claim_count": 5, "grounded_claim_ratio": 1.0},
        "duplicate_agent_output": {"duplicate_agent_output_rate": 0.0},
        "red_team": {"blocking_issue_count": 0, "review_issue_count": 0},
        "report_quality": {"status": "pass"},
        "brief": {"citation_appendix": [{"evidence_id": f"C-{query_id}"}]},
    }


def failing_report():
    return {
        "status": "blocked",
        "agent_outputs": [{"status": "complete"}],
        "grounding_report": {"status": "fail", "claim_count": 2, "grounded_claim_ratio": 0.5},
        "duplicate_agent_output": {"duplicate_agent_output_rate": 0.4},
        "red_team": {"blocking_issue_count": 1, "review_issue_count": 0},
        "report_quality": {"status": "fail"},
        "brief": {"citation_appendix": []},
    }


if __name__ == "__main__":
    unittest.main()
