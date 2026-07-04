import unittest

from canon.eval.rag import average, policy_summary, summarize_synthesis


class RagEvalTests(unittest.TestCase):
    def test_average(self):
        self.assertEqual(average([1, 2, 3]), 2.0)
        self.assertEqual(average([]), 0.0)

    def test_summarize_synthesis(self):
        summary = summarize_synthesis(
            {
                "policy": "rag",
                "answer": "Answer",
                "citations": [{"citation_id": "C1"}],
                "evidence": [{"claim": {"id": "claim-a"}}],
                "conflict_notes": [{"id": "conflict-a"}],
                "importance_summary": {
                    "distinct_works": 1,
                    "distinct_sources": 1,
                    "distinct_clusters": 1,
                    "average_components": {"relevance": 1.0},
                },
                "limitations": ["limited"],
            }
        )
        self.assertEqual(summary["claim_citation_count"], 1)
        self.assertEqual(summary["conflict_note_count"], 1)

    def test_policy_summary(self):
        summary = policy_summary(
            [
                {
                    "policy_runs": [
                        {
                            "policy": "rag",
                            "citation_count": 2,
                            "claim_citation_count": 1,
                            "conflict_note_count": 1,
                            "distinct_clusters": 2,
                            "limitations": [],
                        }
                    ]
                }
            ],
            ["rag"],
        )
        self.assertEqual(summary["rag"]["average_citation_count"], 2.0)
        self.assertEqual(summary["rag"]["runs_with_limitations"], 0)


if __name__ == "__main__":
    unittest.main()
