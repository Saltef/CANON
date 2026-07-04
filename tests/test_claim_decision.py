import unittest

from canon.reports.claim_decision import warning_resolutions


class ClaimDecisionTests(unittest.TestCase):
    def test_warning_resolutions_block_overclaiming(self):
        resolutions = warning_resolutions(
            {
                "batch_eval": {
                    "stability": {
                        "stable_top_method": None,
                        "minimum_top2_margin": 0.001,
                    }
                },
                "external_ir": {
                    "benchmark_kind": "internal_qrels",
                    "benchmark_id": "internal",
                },
                "bootstrap_ir": {
                    "topline": {"winner_probability": 0.4},
                },
                "public_qrels_validation": {
                    "status": "pass",
                    "benchmark_id": "public_smoke",
                },
                "evaluation_suite": {
                    "technical_diagnostics": {
                        "dominance": [
                            {
                                "dominant_method_id": "rag",
                                "dominated_method_id": "source_quality",
                            }
                        ]
                    }
                },
            },
            {"source_quality": "diagnostic_baseline"},
        )
        self.assertEqual(
            set(resolutions),
            {
                "unstable_top_method",
                "narrow_top2_margin",
                "internal_qrels_only",
                "low_bootstrap_winner_probability",
                "dominated_methods",
            },
        )
        self.assertFalse(resolutions["dominated_methods"].get("unresolved", False))


if __name__ == "__main__":
    unittest.main()
