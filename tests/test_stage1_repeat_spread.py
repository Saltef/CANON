import unittest

from canon.product.stage1_repeat_spread import (
    comparison_decision,
    paired_delta,
    query_work_level_summary,
    summarize_runs,
    summarize_values,
)
from canon.product.stage1_repeat_spread import RepeatConfig


class Stage1RepeatSpreadTests(unittest.TestCase):
    def test_summarize_values_reports_spread_and_sample_std(self):
        summary = summarize_values([0.39, 0.44, 0.40])

        self.assertEqual(summary["n"], 3)
        self.assertEqual(summary["min"], 0.39)
        self.assertEqual(summary["max"], 0.44)
        self.assertEqual(summary["spread"], 0.05)
        self.assertEqual(summary["spread_pp"], 5.0)
        self.assertGreater(summary["sample_std"], 0)

    def test_query_work_level_summary_counts_parent_document_hits(self):
        row = query_work_level_summary(
            {
                "id": "q1",
                "scored_candidates": [
                    {"chunk_id": "chunk:sibling"},
                    {"chunk_id": "chunk:other"},
                ],
            },
            {"chunk:gold": 3.0, "chunk:missing": 1.0},
            {
                "chunk:gold": "work:a",
                "chunk:sibling": "work:a",
                "chunk:missing": "work:b",
                "chunk:other": "work:c",
            },
        )

        self.assertEqual(row["relevant_work_count"], 2)
        self.assertEqual(row["relevant_work_hit_count"], 1)
        self.assertEqual(row["relevant_work_hit_rate"], 0.5)
        self.assertTrue(row["any_relevant_work_in_pool"])

    def test_paired_delta_compares_runs_by_repeat(self):
        runs = [
            run("base", 1, 0.40),
            run("treatment", 1, 0.42),
            run("base", 2, 0.41),
            run("treatment", 2, 0.40),
        ]

        delta = paired_delta(
            runs,
            treatment_config="treatment",
            baseline_config="base",
            metric="nDCG@10",
        )

        self.assertEqual(delta["n"], 2)
        self.assertEqual(delta["rows"][0]["delta"], 0.02)
        self.assertEqual(delta["rows"][1]["delta"], -0.01)
        self.assertEqual(delta["mean_delta"], 0.005)

    def test_claim_check_rejects_effect_smaller_than_spread(self):
        configs = [
            RepeatConfig("base", "Base", False, False, 0, "original"),
            RepeatConfig("treatment", "Treatment", True, True, 0, "parent"),
        ]
        runs = [
            run("base", 1, 0.39),
            run("base", 2, 0.44),
            run("base", 3, 0.40),
            run("treatment", 1, 0.392),
            run("treatment", 2, 0.442),
            run("treatment", 3, 0.402),
        ]
        summaries = summarize_runs(runs, configs, 10)
        delta = paired_delta(
            runs,
            treatment_config="treatment",
            baseline_config="base",
            metric="nDCG@10",
        )

        decision = comparison_decision(delta, summaries)

        self.assertEqual(
            decision["standing"],
            "does_not_stand_effect_smaller_than_repeat_spread",
        )
        self.assertEqual(decision["mean_delta_pp"], 0.2)
        self.assertEqual(decision["max_within_config_spread_pp"], 5.0)

    def test_claim_check_keeps_repeatable_tiny_effect_separate(self):
        configs = [
            RepeatConfig("base", "Base", False, False, 0, "original"),
            RepeatConfig("treatment", "Treatment", True, True, 0, "parent"),
        ]
        runs = [
            run("base", 1, 0.390),
            run("base", 2, 0.390),
            run("base", 3, 0.390),
            run("treatment", 1, 0.392),
            run("treatment", 2, 0.392),
            run("treatment", 3, 0.392),
        ]
        summaries = summarize_runs(runs, configs, 10)
        delta = paired_delta(
            runs,
            treatment_config="treatment",
            baseline_config="base",
            metric="nDCG@10",
        )

        decision = comparison_decision(delta, summaries)

        self.assertEqual(decision["standing"], "stands_as_repeatable_but_tiny")


def run(configuration: str, repeat: int, ndcg: float):
    return {
        "configuration": configuration,
        "repeat": repeat,
        "status": "ok",
        "metrics": {
            "nDCG@10": ndcg,
            "Recall@10": 0.1,
            "MAP@10": 0.2,
            "MRR@10": 0.3,
            "candidate_recall_mean": 0.4,
            "work_level_hit_rate_mean": 0.5,
        },
    }


if __name__ == "__main__":
    unittest.main()
