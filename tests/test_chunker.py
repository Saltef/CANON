import unittest

from canon.ingest.chunker import chunk_work_abstract, claim_density, section_importance
from canon.models import Work


class ChunkerTests(unittest.TestCase):
    def test_chunk_work_abstract_produces_stable_metadata(self):
        work = Work(
            id="W1",
            doi=None,
            title="Demo",
            year=2020,
            language="en",
            abstract=" ".join(f"token{i}" for i in range(20)),
            source_name="Journal",
            is_open_access=True,
            is_retracted=False,
            cited_by_count=10,
            referenced_work_count=2,
            pdf_url=None,
            landing_page_url=None,
            raw={},
        )
        chunks = chunk_work_abstract(work, chunk_tokens=8, overlap_tokens=2)
        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0].token_start, 0)
        self.assertEqual(chunks[1].token_start, 6)
        self.assertEqual(chunks[0].section, "abstract")

    def test_section_importance_prioritizes_abstract(self):
        self.assertGreater(section_importance("abstract"), section_importance("introduction"))

    def test_claim_density_detects_claim_markers(self):
        self.assertGreater(claim_density("We find a significant effect."), 0.0)


if __name__ == "__main__":
    unittest.main()
