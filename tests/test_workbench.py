import unittest

from canon.workbench.build import render_workbench


class WorkbenchTests(unittest.TestCase):
    def test_render_workbench_contains_policy_summary(self):
        html = render_workbench(
            "unit",
            {
                "rag_eval": {
                    "query_count": 1,
                    "policy_summary": {"rag": {"average_citation_count": 5}},
                    "queries": [],
                },
                "claims": {"claim_count": 2},
                "conflicts": {"conflict_count": 1},
                "quality": {"work_count": 3},
            },
        )
        self.assertIn("CANON Workbench", html)
        self.assertIn("rag", html)
        self.assertIn("Claims", html)

    def test_render_workbench_contains_diversity_audit(self):
        html = render_workbench(
            "unit",
            {
                "rag_eval": {"query_count": 0, "policy_summary": {}, "queries": []},
                "claims": {},
                "conflicts": {},
                "quality": {},
                "diversity": {
                    "query_count": 1,
                    "baseline_method_id": "lexical_k5_template",
                    "diverse_method_id": "diverse_k5_template",
                    "aggregate": {
                        "average_cluster_delta": 1.5,
                        "average_breadth_precision": 0.4,
                        "average_noise_rate": 0.0,
                        "verdict_counts": {"useful_breadth": 1},
                    },
                    "queries": [
                        {
                            "query_id": "q1",
                            "query": "How effective are sanctions?",
                            "verdict": "useful_breadth",
                            "cluster_delta": 2,
                            "breadth_precision": 1.0,
                            "noise_rate": 0.0,
                            "interpretation": "Diversity-first added useful breadth.",
                            "added_titles": [{"title": "Sanctions evidence"}],
                        }
                    ],
                },
                "diversity_diagnostics": {
                    "headline": {
                        "finding": "Diversity-first is conditional.",
                        "claim_language": "Report as conditional breadth.",
                    },
                    "aggregate": {
                        "failure_mode_counts": {"low_focus_coverage": 1},
                    },
                    "next_actions": [
                        {"priority": "high", "action": "Add focus gating."},
                    ],
                },
                "focus_diversity": {
                    "aggregate": {
                        "average_breadth_precision": 0.6,
                        "average_noise_rate": 0.0,
                        "verdict_counts": {
                            "useful_breadth": 2,
                            "mixed_breadth": 0,
                            "off_topic_breadth_risk": 0,
                            "no_diversity_gain": 0,
                        },
                    },
                },
                "focus_diversity_diagnostics": {
                    "headline": {"claim_language": "Focus gating improves precision."},
                    "next_actions": [
                        {"priority": "high", "action": "Validate gates."},
                    ],
                },
            },
        )
        self.assertIn("Diversity-First Audit", html)
        self.assertIn("useful_breadth", html)
        self.assertIn("Diversity-first added useful breadth.", html)
        self.assertIn("Sanctions evidence", html)
        self.assertIn("Original Diversity Diagnosis", html)
        self.assertIn("low_focus_coverage", html)
        self.assertIn("Focus-Gated Diversity Test", html)
        self.assertIn("Focus gating improves precision.", html)


if __name__ == "__main__":
    unittest.main()
