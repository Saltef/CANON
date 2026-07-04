import unittest

from canon.eval.slices import build_slices, cross_slice_summary, parse_method_ids


class EvalSlicesTests(unittest.TestCase):
    def test_parse_method_ids(self):
        self.assertEqual(parse_method_ids("a,b"), ["a", "b"])
        self.assertIsNone(parse_method_ids(""))

    def test_build_slices_groups_by_type_and_topic(self):
        queries = [
            {
                "id": "q1",
                "type": "synthesis",
                "query": "Do democracies go to war?",
            },
            {
                "id": "q2",
                "type": "methodological",
                "query": "civil war methods",
            },
        ]
        slices = build_slices(queries)
        self.assertEqual(len(slices["all"]), 2)
        self.assertEqual(len(slices["type_synthesis"]), 1)
        self.assertEqual(slices["topic_democratic_peace"][0]["id"], "q1")
        self.assertEqual(slices["topic_civil_war"][0]["id"], "q2")

    def test_cross_slice_summary_counts_wins_and_ties(self):
        summary = cross_slice_summary(
            [
                {
                    "leaderboard": [{"method_id": "m1"}],
                    "technical_diagnostics": {
                        "near_ties": [{"left_method_id": "m1", "right_method_id": "m2"}]
                    },
                },
                {
                    "leaderboard": [{"method_id": "m2"}],
                    "technical_diagnostics": {"near_ties": []},
                },
            ]
        )
        self.assertEqual(summary["wins_by_method"], {"m1": 1, "m2": 1})
        self.assertEqual(summary["near_tie_mentions_by_method"], {"m1": 1, "m2": 1})


if __name__ == "__main__":
    unittest.main()
