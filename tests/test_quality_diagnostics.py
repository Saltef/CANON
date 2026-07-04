import unittest

from canon.quality.diagnostics import pearson, summarize


class QualityDiagnosticsTests(unittest.TestCase):
    def test_summarize_reports_basic_stats(self):
        self.assertEqual(summarize([1.0, 3.0])["mean"], 2.0)

    def test_pearson_detects_positive_correlation(self):
        self.assertEqual(pearson([1, 2, 3], [2, 4, 6]), 1.0)

    def test_pearson_handles_flat_values(self):
        self.assertEqual(pearson([1, 1, 1], [2, 3, 4]), 0.0)


if __name__ == "__main__":
    unittest.main()
