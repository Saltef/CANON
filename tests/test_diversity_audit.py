import unittest

from canon.eval.diversity import (
    aggregate_reports,
    analyze_diversity_gain,
    diversity_verdict,
    query_focus_terms,
    result_focus_coverage,
)


def result(chunk_id, title, cluster_id, relevance, preview=""):
    return {
        "chunk_id": chunk_id,
        "rank": int(chunk_id[1:]),
        "title": title,
        "cluster_id": cluster_id,
        "preview": preview,
        "components": {
            "relevance": relevance,
            "semantic_similarity": relevance,
        },
    }


class DiversityAuditTests(unittest.TestCase):
    def test_query_focus_terms_remove_generic_words(self):
        self.assertEqual(
            query_focus_terms("What does the corpus say about democratic peace evidence?"),
            ["democratic", "peace"],
        )

    def test_result_focus_coverage_uses_title_and_preview(self):
        coverage = result_focus_coverage(
            result("c1", "Democratic Peace", 1, 1.0, preview="war and institutions"),
            ["democratic", "peace", "institutions"],
        )
        self.assertEqual(coverage, 1.0)

    def test_analyze_diversity_gain_marks_useful_breadth(self):
        query = {
            "id": "q1",
            "type": "comparison",
            "query": "Compare democratic peace and institutions evidence.",
        }
        baseline = [
            result("c1", "Democratic Peace", 1, 0.9),
            result("c2", "Democratic Peace Evidence", 1, 0.8),
        ]
        diverse = [
            result("c1", "Democratic Peace", 1, 0.9),
            result("c3", "Institutions Evidence", 2, 0.7),
        ]
        report = analyze_diversity_gain(query, baseline, diverse, "rag", "diverse")
        self.assertEqual(report["cluster_delta"], 1)
        self.assertEqual(report["useful_breadth_count"], 1)
        self.assertEqual(report["verdict"], "useful_breadth")
        self.assertIn("useful breadth", report["interpretation"])
        self.assertIn("focus_coverage", report["added_titles"][0])

    def test_analyze_diversity_gain_marks_off_topic_breadth_risk(self):
        query = {
            "id": "q1",
            "type": "off_topic",
            "query": "What does this corpus say about quantum computing hardware?",
        }
        baseline = [
            result("c1", "Quantum Governance", 1, 0.9),
            result("c2", "High Risk Technologies", 1, 0.8),
        ]
        diverse = [
            result("c1", "Quantum Governance", 1, 0.9),
            result("c3", "Cloud Computing in Universities", 2, 0.7),
        ]
        report = analyze_diversity_gain(query, baseline, diverse, "lexical", "diverse")
        self.assertEqual(report["verdict"], "off_topic_breadth_risk")
        self.assertEqual(report["useful_breadth_count"], 0)
        self.assertIn("off-topic", report["interpretation"])

    def test_diversity_verdict_marks_noisy_breadth(self):
        self.assertEqual(
            diversity_verdict(
                {
                    "cluster_delta": 1,
                    "breadth_precision": 0.0,
                    "noise_rate": 0.75,
                }
            ),
            "noisy_breadth",
        )

    def test_aggregate_reports_counts_verdicts(self):
        aggregate = aggregate_reports(
            [
                {
                    "query_id": "q1",
                    "verdict": "useful_breadth",
                    "cluster_delta": 1,
                    "rank_overlap": 0.5,
                    "breadth_precision": 1.0,
                    "noise_rate": 0.0,
                },
                {
                    "query_id": "q2",
                    "verdict": "noisy_breadth",
                    "cluster_delta": 1,
                    "rank_overlap": 0.2,
                    "breadth_precision": 0.0,
                    "noise_rate": 1.0,
                },
            ]
        )
        self.assertEqual(aggregate["verdict_counts"]["useful_breadth"], 1)
        self.assertEqual(aggregate["queries_with_noisy_breadth"], ["q2"])


if __name__ == "__main__":
    unittest.main()
