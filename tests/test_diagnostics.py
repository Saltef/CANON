import unittest

from canon.ingest.diagnostics import corpus_diagnostics
from canon.models import Work


class DiagnosticsTests(unittest.TestCase):
    def test_corpus_diagnostics_counts_coverage(self):
        works = [
            Work("W1", None, "A", 2020, "en", "Abstract", "J", True, False, 1, 2, "pdf", None, {}),
            Work("W2", None, "B", 2021, "en", None, "J", False, True, 0, 0, None, None, {}),
        ]
        report = corpus_diagnostics(works)
        self.assertEqual(report["total_works"], 2)
        self.assertEqual(report["abstract_coverage"], 0.5)
        self.assertEqual(report["retracted_works"], 1)


if __name__ == "__main__":
    unittest.main()
