import unittest
from pathlib import Path

from canon.eval.disagreement_fixture import build_disagreement_benchmark
from canon.eval.disagreement_preservation import (
    evaluate_disagreement_preservation,
    score_stage2,
)


class DisagreementPreservationTests(unittest.TestCase):
    def test_score_stage2_rewards_linked_contradiction_cluster(self):
        stage2 = {
            "cited_claims": [
                {
                    "claim_id": "S2C1",
                    "links": [
                        {"evidence_id": "C1", "stance": "supports"},
                        {"evidence_id": "C2", "stance": "contradicts"},
                    ],
                }
            ],
            "quality_gate": {
                "checks": [{"id": "citation_integrity", "status": "pass"}],
                "unsupported_claims": [],
            },
        }

        metrics = score_stage2(stage2, {"C1": "supports", "C2": "contradicts"})

        self.assertEqual(metrics["contradiction_recall"], 1.0)
        self.assertEqual(metrics["stance_accuracy"], 1.0)
        self.assertEqual(metrics["disagreement_preservation_score"], 1.0)

    def test_score_stage2_penalizes_flattened_or_missing_contradiction(self):
        stage2 = {
            "cited_claims": [
                {
                    "claim_id": "S2C1",
                    "links": [
                        {"evidence_id": "C1", "stance": "supports"},
                        {"evidence_id": "C2", "stance": "supports"},
                        {"evidence_id": "C3", "stance": "supports"},
                    ],
                }
            ],
            "quality_gate": {
                "checks": [{"id": "citation_integrity", "status": "pass"}],
                "unsupported_claims": [{"claim_id": "S2C1"}],
            },
        }

        metrics = score_stage2(stage2, {"C1": "supports", "C2": "contradicts"})

        self.assertEqual(metrics["contradiction_recall"], 0.0)
        self.assertLess(metrics["stance_accuracy"], 1.0)
        self.assertIn("C3", metrics["unexpected_linked_evidence"])
        self.assertLess(metrics["disagreement_preservation_score"], 0.75)

    def test_generated_publishable_fixture_has_target_sized_case_set(self):
        benchmark = build_disagreement_benchmark()

        self.assertEqual(benchmark["benchmark_id"], "disagreement_preservation_publishable_v1")
        self.assertEqual(benchmark["case_count"], 30)
        self.assertEqual(len(benchmark["cases"]), 30)
        self.assertEqual(
            {row["expected_stance"] for row in benchmark["cases"][0]["evidence"]},
            {"supports", "contradicts", "qualifies", "neutral"},
        )

    def test_publishable_fixture_runs_through_stage2_synthesis(self):
        report = evaluate_disagreement_preservation(
            benchmark_path=Path("gold/disagreement_preservation_publishable.json"),
            write_report=False,
        )

        self.assertEqual(report["report_id"], "disagreement_preservation_benchmark_v1")
        self.assertEqual(report["case_count"], 30)
        self.assertEqual(report["status"], "automated_pass_human_review_required")
        self.assertTrue(report["human_review_required"])
        self.assertEqual(report["summary"]["average_metrics"]["contradiction_recall"], 1.0)
        self.assertEqual(report["summary"]["average_metrics"]["citation_integrity"], 1.0)
        self.assertGreaterEqual(
            report["summary"]["average_metrics"]["disagreement_preservation_score"],
            0.75,
        )


if __name__ == "__main__":
    unittest.main()
