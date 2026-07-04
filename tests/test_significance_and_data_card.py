import unittest

from canon.eval.significance import paired_significance_report, sign_flip_p_value
from canon.reports.data_card import coverage, limitations


class SignificanceAndDataCardTests(unittest.TestCase):
    def test_paired_significance_detects_consistent_advantage(self):
        report = paired_significance_report(
            {
                "benchmark_id": "toy",
                "mode": "m",
                "method_set_id": "methods",
                "method_summary": {"a": {}, "b": {}},
                "queries": [
                    {
                        "method_runs": [
                            {"method_id": "a", "metrics": {"nDCG@10": 1.0}},
                            {"method_id": "b", "metrics": {"nDCG@10": 0.0}},
                        ]
                    },
                    {
                        "method_runs": [
                            {"method_id": "a", "metrics": {"nDCG@10": 0.8}},
                            {"method_id": "b", "metrics": {"nDCG@10": 0.1}},
                        ]
                    },
                ],
            },
            metric="nDCG@10",
            samples=50,
            seed=3,
        )
        comparison = report["comparisons"][0]
        self.assertEqual(comparison["direction"], "left_better")
        self.assertGreater(comparison["bootstrap_ci"]["low"], 0.0)

    def test_sign_flip_p_value_is_bounded(self):
        p_value = sign_flip_p_value([0.2, -0.1, 0.3], samples=50, seed=5)
        self.assertGreaterEqual(p_value, 0.0)
        self.assertLessEqual(p_value, 1.0)

    def test_data_card_coverage_counts_fields(self):
        works = [
            {
                "abstract": "A",
                "is_open_access": True,
                "pdf_url": "https://example.org/a.pdf",
                "referenced_work_count": 3,
                "author_display_names": ["A"],
                "max_author_cited_by_count": 9,
            },
            {"abstract": "", "is_open_access": False},
        ]
        cov = coverage(works, [{"id": "c1"}, {"id": "c2"}])
        self.assertEqual(cov["abstract_coverage"], 0.5)
        self.assertEqual(cov["chunks_per_work"], 1.0)

    def test_data_card_limitations_flag_small_corpus(self):
        limits = limitations([], [], {})
        self.assertIn("Corpus is small; results are suitable for technical testing, not field-level claims.", limits)


if __name__ == "__main__":
    unittest.main()
