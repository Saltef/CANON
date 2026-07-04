import unittest

from canon.ingest.openalex import (
    abstract_from_inverted_index,
    build_filter,
    build_url,
    normalize_work,
    summarize_authorships,
)


class OpenAlexTests(unittest.TestCase):
    def test_abstract_from_inverted_index_orders_tokens(self):
        self.assertEqual(
            abstract_from_inverted_index({"world": [1], "hello": [0]}),
            "hello world",
        )

    def test_normalize_work_extracts_pdf_fallback(self):
        work = normalize_work(
            {
                "id": "W",
                "title": "Title",
                "publication_year": 2020,
                "language": "en",
                "abstract_inverted_index": {"A": [0]},
                "primary_location": {"source": {"display_name": "Journal"}},
                "best_oa_location": {"pdf_url": "https://example.org/a.pdf"},
                "open_access": {"is_oa": True},
                "referenced_works": ["R1"],
            }
        )
        self.assertEqual(work.pdf_url, "https://example.org/a.pdf")
        self.assertEqual(work.referenced_work_count, 1)

    def test_build_filter_contains_scope(self):
        filters = build_filter(
            {
                "openalex": {
                    "from_publication_year": 2010,
                    "to_publication_year": 2026,
                    "language": "en",
                }
            },
            ["C1", "C2"],
        )
        self.assertIn("type:article", filters)
        self.assertIn("concepts.id:C1|C2", filters)

    def test_build_url_includes_sort_strategy(self):
        url = build_url(
            "https://api.openalex.org",
            "person@example.com",
            "type:article",
            "democratic peace",
            50,
            sort="cited_by_count:desc",
            api_key="secret",
        )
        self.assertIn("sort=cited_by_count%3Adesc", url)
        self.assertIn("search=democratic+peace", url)
        self.assertIn("api_key=secret", url)

    def test_summarize_authorships_keeps_max_author_impact(self):
        summary = summarize_authorships(
            [
                {"author": {"display_name": "A", "cited_by_count": 10, "works_count": 2}},
                {"author": {"display_name": "B", "cited_by_count": 25, "works_count": 5}},
            ]
        )
        self.assertEqual(summary["display_names"], ["A", "B"])
        self.assertEqual(summary["author_count"], 2)
        self.assertEqual(summary["max_cited_by_count"], 25)


if __name__ == "__main__":
    unittest.main()
