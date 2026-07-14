import unittest

from canon.labeling.judge import judge_task
from canon.labeling.calibration import calibrate_label_tasks, technical_labels_for_task
from canon.labeling.tasks import LABEL_SCHEMA, task_for_run
from canon.eval.technical_calibration import policy_summary, technical_row


class LabelingTests(unittest.TestCase):
    def test_task_for_run_creates_empty_label_slots(self):
        task = task_for_run(
            {"id": "q1", "query": "Question?", "importance_expectation": "Expectation"},
            {
                "policy": "rag",
                "answer": "Answer",
                "citations": [],
                "citation_count": 0,
                "claim_citation_count": 0,
                "conflict_note_count": 0,
                "distinct_sources": 0,
                "distinct_clusters": 0,
            },
        )
        self.assertEqual(set(task["labels"]), set(LABEL_SCHEMA))

    def test_judge_task_scores_complete_answer_context(self):
        judgment = judge_task(
            {
                "id": "q1::rag",
                "policy": "rag",
                "metrics": {
                    "citation_count": 3,
                    "claim_citation_count": 1,
                    "distinct_sources": 2,
                    "distinct_clusters": 2,
                    "conflict_note_count": 1,
                },
            }
        )
        self.assertEqual(judgment["heuristic_score"], 1.0)
        self.assertIn("has_claim_backing", judgment["reasons"])

    def test_technical_labels_for_task_generate_proxy_labels(self):
        task = task_for_run(
            {"id": "q1", "query": "Which studies disagree?", "importance_expectation": "Conflict"},
            {
                "policy": "rag",
                "answer": "Answer",
                "citations": [{"citation_id": "C1"} for _ in range(3)],
                "citation_count": 3,
                "claim_citation_count": 1,
                "conflict_note_count": 1,
                "distinct_sources": 2,
                "distinct_clusters": 2,
                "average_components": {"relevance": 0.5, "source_quality": 0.5},
                "limitations": [],
            },
        )
        labels = technical_labels_for_task(task)
        self.assertEqual(labels["citation_support"], "supported")
        self.assertEqual(labels["evidence_role_fit"], "right_role")
        self.assertEqual(labels["source_trust_fit"], "appropriate")

    def test_calibration_reports_human_technical_agreement(self):
        task = task_for_run(
            {"id": "q1", "query": "Question?", "importance_expectation": "Expectation"},
            {
                "policy": "rag",
                "answer": "Answer",
                "citations": [{"citation_id": "C1"}],
                "citation_count": 1,
                "claim_citation_count": 0,
                "conflict_note_count": 0,
                "distinct_sources": 1,
                "distinct_clusters": 1,
            },
        )
        task["labels"]["citation_support"] = "partially_supported"
        report = calibrate_label_tasks({"mode": "dry_run", "tasks": [task]})
        self.assertEqual(report["completed_human_task_count"], 1)
        self.assertEqual(report["agreement"]["overall_exact_agreement_rate"], 1.0)

    def test_technical_calibration_policy_summary(self):
        task = task_for_run(
            {"id": "q1", "query": "Question?", "importance_expectation": "Expectation"},
            {
                "policy": "rag",
                "answer": "Answer",
                "citations": [{"citation_id": "C1"}],
                "citation_count": 1,
                "claim_citation_count": 0,
                "conflict_note_count": 0,
                "distinct_sources": 1,
                "distinct_clusters": 1,
            },
        )
        row = technical_row(task, {"average_components": {}})
        summary = policy_summary([row])
        self.assertIn("rag", summary)
        self.assertIn("average_technical_calibration_score", summary["rag"])


if __name__ == "__main__":
    unittest.main()
