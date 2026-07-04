import unittest

from canon.reports.portfolio import diversity_delta, render_markdown, summarize_diversity


class PortfolioReportTests(unittest.TestCase):
    def test_summarize_diversity_extracts_verdict_counts(self):
        summary = summarize_diversity(
            {
                "diverse_method_id": "focus_diverse_k5_template",
                "query_count": 25,
                "aggregate": {
                    "verdict_counts": {
                        "useful_breadth": 12,
                        "mixed_breadth": 8,
                    },
                    "average_cluster_delta": 1.64,
                    "average_breadth_precision": 0.422,
                    "average_noise_rate": 0.0,
                },
            }
        )
        self.assertEqual(summary["useful_breadth"], 12)
        self.assertEqual(summary["mixed_breadth"], 8)
        self.assertEqual(summary["no_diversity_gain"], 0)

    def test_diversity_delta_compares_numeric_fields(self):
        delta = diversity_delta(
            {"useful_breadth": 12, "average_noise_rate": 0.0},
            {"useful_breadth": 10, "average_noise_rate": 0.013333},
        )
        self.assertEqual(delta["useful_breadth"], 2.0)
        self.assertEqual(delta["average_noise_rate"], -0.013333)

    def test_render_markdown_contains_claim_limits(self):
        markdown = render_markdown(
            {
                "positioning": "Importance-aware scholarly RAG.",
                "corpus": {"work_count": 10, "chunk_count": 8, "source_modes": ["a", "b"]},
                "main_finding": "Diversity is conditional.",
                "diversity_comparison": {
                    "original": {
                        "method_id": "diverse",
                        "useful_breadth": 10,
                        "mixed_breadth": 10,
                        "off_topic_breadth_risk": 1,
                        "no_diversity_gain": 4,
                        "average_breadth_precision": 0.38,
                        "average_noise_rate": 0.01,
                        "average_cluster_delta": 1.7,
                    },
                    "focus_gated": {
                        "method_id": "focus",
                        "useful_breadth": 12,
                        "mixed_breadth": 8,
                        "off_topic_breadth_risk": 1,
                        "no_diversity_gain": 4,
                        "average_breadth_precision": 0.42,
                        "average_noise_rate": 0.0,
                        "average_cluster_delta": 1.6,
                    },
                    "delta_focus_minus_original": {
                        "useful_breadth": 2,
                        "mixed_breadth": -2,
                        "average_breadth_precision": 0.04,
                        "average_noise_rate": -0.01,
                    },
                },
                "allowed_claim": "Allowed.",
                "claim_limits": ["No global method-winner claim."],
                "reproducibility": ["python -m canon.reports.portfolio"],
            }
        )
        self.assertIn("CANON Portfolio Summary", markdown)
        self.assertIn("No global method-winner claim.", markdown)
        self.assertIn("focus", markdown)


if __name__ == "__main__":
    unittest.main()
