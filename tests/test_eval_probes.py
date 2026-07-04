import unittest

from canon.eval.probes import (
    category_summary,
    expectation_metric,
    failure_summary,
    parse_method_ids,
    probe_query_record,
    probe_score,
    probe_summary,
    score_expectations,
)


class Spec:
    def __init__(self, method_id="m1", label="Method 1"):
        self.id = method_id
        self.label = label


class EvalProbeTests(unittest.TestCase):
    def test_parse_method_ids(self):
        self.assertEqual(parse_method_ids("a,b"), ["a", "b"])
        self.assertIsNone(parse_method_ids(""))

    def test_expectation_metric_parses_min_and_max(self):
        self.assertEqual(expectation_metric("min_context_relevance"), ("context_relevance", "min"))
        self.assertEqual(expectation_metric("max_semantic_alignment"), ("semantic_alignment", "max"))

    def test_score_expectations_handles_minimums_and_maximums(self):
        result = score_expectations(
            {"context_relevance": 0.2, "semantic_alignment": 0.1},
            {"min_context_relevance": 0.15, "max_semantic_alignment": 0.2},
        )
        self.assertTrue(result["passed"])
        self.assertEqual(result["pass_rate"], 1.0)

    def test_score_expectations_reports_failures(self):
        result = score_expectations(
            {"context_relevance": 0.4},
            {"max_context_relevance": 0.25},
        )
        self.assertFalse(result["passed"])
        self.assertEqual(result["pass_rate"], 0.0)

    def test_probe_score_blends_technical_and_expectation_scores(self):
        score = probe_score({"claim_coverage": 1.0}, {"claim_coverage": 1.0}, 0.5)
        self.assertEqual(score, 0.85)

    def test_probe_query_record_adapts_probe_for_method_runner(self):
        record = probe_query_record(
            {
                "id": "p1",
                "category": "off_topic",
                "query": "What about unrelated hardware?",
                "rationale": "weak support check",
            }
        )
        self.assertEqual(record["type"], "off_topic")
        self.assertEqual(record["importance_expectation"], "weak support check")

    def test_probe_summary_and_category_summary(self):
        probe_reports = [
            {
                "category": "off_topic",
                "method_runs": [
                    {
                        "method_id": "m1",
                        "probe_score": 0.8,
                        "expectation_result": {"pass_rate": 1.0, "passed": True},
                    }
                ],
            }
        ]
        specs = [Spec()]
        summary = probe_summary(probe_reports, specs)
        categories = category_summary(probe_reports, specs)
        self.assertEqual(summary["leaderboard"][0]["method_id"], "m1")
        self.assertEqual(summary["by_method"]["m1"]["full_probe_passes"], 1)
        self.assertEqual(categories["off_topic"]["m1"]["expectation_pass_rate"], 1.0)

    def test_failure_summary_groups_failed_expectations(self):
        summary = failure_summary(
            [
                {
                    "id": "p1",
                    "category": "off_topic",
                    "method_runs": [
                        {
                            "method_id": "m1",
                            "probe_score": 0.4,
                            "expectation_result": {
                                "checks": [
                                    {"expectation": "max_context_relevance", "passed": False},
                                    {"expectation": "max_semantic_alignment", "passed": True},
                                ]
                            },
                        }
                    ],
                }
            ]
        )
        self.assertEqual(summary["failure_count"], 1)
        self.assertEqual(summary["by_category"], {"off_topic": 1})
        self.assertEqual(summary["by_expectation"], {"max_context_relevance": 1})


if __name__ == "__main__":
    unittest.main()
