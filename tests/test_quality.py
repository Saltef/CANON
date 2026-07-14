import unittest

from canon.models import Work
from canon.quality.signals import source_quality_vector, source_trust_vector


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

    def test_source_trust_does_not_treat_pdf_as_quality(self):
        with_pdf = Work("W", None, "T", 2020, "en", None, None, True, False, 0, 0, "pdf", None, {})
        without_pdf = Work("W", None, "T", 2020, "en", None, None, False, False, 0, 0, None, None, {})
        self.assertEqual(source_trust_vector(with_pdf)["source_trust"], source_trust_vector(without_pdf)["source_trust"])
        self.assertGreater(source_trust_vector(with_pdf)["accessibility"], source_trust_vector(without_pdf)["accessibility"])

    def test_retracted_source_trust_is_blocked(self):
        work = Work("W", "10/demo", "T", 2020, "en", None, "Journal", True, True, 100, 30, "pdf", None, {})
        vector = source_trust_vector(work)
        self.assertEqual(vector["trust_tier"], "blocked")
        self.assertEqual(vector["source_trust"], 0.0)

    def test_domain_profile_changes_trust_interpretation(self):
        work = Work(
            "W",
            "10/demo",
            "T",
            2020,
            "en",
            "Archive method and historical sources.",
            "Journal",
            False,
            False,
            40,
            25,
            None,
            "https://example.test",
            {"source_type": "report"},
        )
        history = source_trust_vector(work, domain="history")
        psychology = source_trust_vector(work, domain="psychology")
        self.assertGreater(history["publication_channel"], psychology["publication_channel"])


if __name__ == "__main__":
    unittest.main()
