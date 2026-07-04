import unittest

from canon.models import Work
from canon.quality.signals import source_quality_vector


class QualityTests(unittest.TestCase):
    def test_retracted_work_gets_zero_retraction_score(self):
        work = Work("W", None, "T", 2020, "en", None, None, False, True, 0, 0, None, None, {})
        vector = source_quality_vector(work)
        self.assertEqual(vector["retraction"], 0.0)

    def test_pdf_open_access_scores_highest(self):
        work = Work("W", None, "T", 2020, "en", None, None, True, False, 0, 0, "pdf", None, {})
        vector = source_quality_vector(work)
        self.assertEqual(vector["open_access"], 1.0)

    def test_author_score_uses_author_impact(self):
        work = Work(
            "W",
            None,
            "T",
            2020,
            "en",
            None,
            None,
            True,
            False,
            0,
            0,
            None,
            None,
            {},
            max_author_cited_by_count=2500,
        )
        vector = source_quality_vector(work)
        self.assertEqual(vector["author_score"], 0.5)


if __name__ == "__main__":
    unittest.main()
