import unittest

from canon.ingest.openalex import (
    author_api_url,
    author_ids,
    batched,
    build_author_batch_url,
    openalex_entity_id,
    summarize_authorships,
)


class AuthorEnrichmentTests(unittest.TestCase):
    def test_author_ids_extracts_openalex_author_urls(self):
        ids = author_ids(
            {
                "authorships": [
                    {"author": {"id": "https://openalex.org/A1"}},
                    {"author": {"id": "https://openalex.org/A2"}},
                ]
            }
        )
        self.assertEqual(ids, ["https://openalex.org/A1", "https://openalex.org/A2"])

    def test_missing_author_metrics_are_zero(self):
        summary = summarize_authorships([{"author": {"display_name": "A"}}])
        self.assertEqual(summary["max_cited_by_count"], 0)
        self.assertEqual(summary["max_works_count"], 0)

    def test_author_api_url_converts_public_entity_url(self):
        self.assertEqual(
            author_api_url("https://openalex.org/A5056371148"),
            "https://api.openalex.org/authors/A5056371148",
        )

    def test_openalex_entity_id_extracts_short_id(self):
        self.assertEqual(openalex_entity_id("https://openalex.org/A1"), "A1")

    def test_batched_splits_author_ids(self):
        self.assertEqual(batched(["A1", "A2", "A3"], 2), [["A1", "A2"], ["A3"]])

    def test_build_author_batch_url_uses_author_index_filter(self):
        url = build_author_batch_url(
            "https://api.openalex.org",
            "person@example.com",
            ["https://openalex.org/A1", "https://openalex.org/A2"],
            api_key="secret",
        )
        self.assertIn("filter=openalex%3AA1%7CA2", url)
        self.assertIn("api_key=secret", url)
        self.assertIn("select=id%2Cdisplay_name%2Ccited_by_count%2Cworks_count%2Csummary_stats", url)


if __name__ == "__main__":
    unittest.main()
