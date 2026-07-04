import unittest

from canon.ingest.pdf_resolver import pdf_candidates
from canon.models import Work


class PdfResolverTests(unittest.TestCase):
    def test_pdf_candidate_prefers_pdf_url(self):
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
            "https://example.org/paper.pdf",
            "https://example.org/paper",
            {},
        )
        candidates = pdf_candidates(work)
        self.assertEqual(candidates[0].source, "openalex")
        self.assertEqual(candidates[0].priority, 1)


if __name__ == "__main__":
    unittest.main()
