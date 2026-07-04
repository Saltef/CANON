import unittest

from canon.labeling.judge import judge_task
from canon.labeling.tasks import LABEL_SCHEMA, task_for_run


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
                    "distinct_clusters": 2,
                    "conflict_note_count": 1,
                },
            }
        )
        self.assertEqual(judgment["heuristic_score"], 1.0)
        self.assertIn("has_claim_backing", judgment["reasons"])


if __name__ == "__main__":
    unittest.main()
