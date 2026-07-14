import unittest

from canon.eval.source_diversity import aggregate_rows, source_diversity_row


class SourceDiversityTests(unittest.TestCase):
    def test_source_diversity_row_passes_with_distinct_sources_and_clusters(self):
        row = source_diversity_row(
            query={"id": "q1", "type": "synthesis", "query": "query"},
            evidence=[
                evidence("c1", "w1", "Source A", 1),
                evidence("c2", "w2", "Source B", 2),
                evidence("c3", "w3", "Source C", 3),
            ],
            top_k=3,
            min_distinct_sources=3,
            max_dominant_source_share=0.5,
            min_cluster_count=2,
        )

        self.assertTrue(row["passed"])
        self.assertEqual(row["distinct_sources"], 3)
        self.assertEqual(row["dominant_source_share"], 0.333333)

    def test_source_diversity_row_flags_source_concentration(self):
        row = source_diversity_row(
            query={"id": "q1", "type": "synthesis", "query": "query"},
            evidence=[
                evidence("c1", "w1", "Source A", 1),
                evidence("c2", "w2", "Source A", 1),
                evidence("c3", "w3", "Source B", 1),
            ],
            top_k=3,
            min_distinct_sources=3,
            max_dominant_source_share=0.5,
            min_cluster_count=2,
        )

        self.assertFalse(row["passed"])
        warning_ids = {warning["id"] for warning in row["warnings"]}
        self.assertIn("low_source_diversity", warning_ids)
        self.assertIn("dominant_source_concentration", warning_ids)
        self.assertIn("low_graph_neighborhood_diversity", warning_ids)

    def test_aggregate_rows_counts_warnings(self):
        rows = [
            {"passed": True, "distinct_sources": 3, "distinct_works": 3, "distinct_clusters": 2, "dominant_source_share": 0.4, "warnings": [], "id": "q1"},
            {"passed": False, "distinct_sources": 1, "distinct_works": 2, "distinct_clusters": 1, "dominant_source_share": 0.8, "warnings": [{"id": "low_source_diversity"}], "id": "q2"},
        ]

        summary = aggregate_rows(rows)

        self.assertEqual(summary["query_pass_rate"], 0.5)
        self.assertEqual(summary["warning_counts"]["low_source_diversity"], 1)


def evidence(chunk_id, work_id, source, cluster):
    return {
        "chunk_id": chunk_id,
        "work_id": work_id,
        "source_name": source,
        "title": f"Title {chunk_id}",
        "cluster_id": cluster,
        "rank": 1,
        "final_score": 0.5,
    }


if __name__ == "__main__":
    unittest.main()
