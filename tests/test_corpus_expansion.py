import unittest

from canon.corpus.expansion import (
    corpus_coverage,
    expansion_readiness,
    index_database_usage,
    topic_balance,
)


class CorpusExpansionTests(unittest.TestCase):
    def test_corpus_coverage_tracks_author_index_inputs(self):
        works = [
            {
                "abstract": "Text",
                "is_open_access": True,
                "pdf_url": "https://example.org/a.pdf",
                "referenced_work_count": 3,
                "author_display_names": ["A"],
                "max_author_cited_by_count": 10,
            },
            {"abstract": None, "referenced_work_count": 0, "author_display_names": []},
        ]
        coverage = corpus_coverage(works, [{"id": "c1"}, {"id": "c2"}])
        self.assertEqual(coverage["author_index_coverage"], 0.5)
        self.assertEqual(coverage["chunks_per_work"], 1.0)

    def test_index_database_usage_counts_openalex_author_metrics(self):
        usage = index_database_usage(
            [
                {
                    "authorships": [
                        {"author": {"id": "https://openalex.org/A1", "cited_by_count": 25}},
                        {"author": {"id": "https://openalex.org/A2", "works_count": 5}},
                    ]
                }
            ],
            [{"max_author_cited_by_count": 25}],
        )
        self.assertEqual(usage["primary_index"], "OpenAlex")
        self.assertEqual(usage["unique_author_ids"], 2)
        self.assertEqual(usage["works_with_author_score_inputs"], 1)

    def test_expansion_readiness_requires_larger_corpus(self):
        readiness = expansion_readiness(
            {
                "minimum_abstract_coverage": 0.8,
                "minimum_reference_coverage": 0.8,
                "minimum_author_index_coverage": 0.8,
            },
            100,
            [{"id": "w1"}],
            {
                "abstract_coverage": 1.0,
                "reference_coverage": 1.0,
                "author_index_coverage": 1.0,
            },
        )
        self.assertEqual(readiness["status"], "needs_expansion")

    def test_topic_balance_counts_unknown_when_no_modes(self):
        balance = topic_balance([{"corpus_source_modes": ["a"]}, {}])
        self.assertEqual(balance["counts"]["a"], 1)
        self.assertEqual(balance["counts"]["unknown"], 1)


if __name__ == "__main__":
    unittest.main()
