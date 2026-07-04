import unittest

from canon.eval.diversity_diagnostics import (
    aggregate_diagnostics,
    classify_failure_mode,
    diagnose_query,
    risk_signals,
)


def query(verdict="mixed_breadth", query_type="synthesis", added=None, removed=None, **overrides):
    payload = {
        "query_id": "q1",
        "query_type": query_type,
        "query": "How effective are sanctions?",
        "verdict": verdict,
        "focus_terms": ["effective", "sanctions"],
        "cluster_delta": 1,
        "rank_overlap": 0.25,
        "added_count": len(added or []),
        "removed_count": len(removed or []),
        "useful_breadth_count": 0,
        "noise_addition_count": 0,
        "breadth_precision": 0.0,
        "noise_rate": 0.0,
        "added_titles": added or [],
        "removed_titles": removed or [],
    }
    payload.update(overrides)
    return payload


def item(title, focus, relevance, cluster_id=2):
    return {
        "rank": 2,
        "chunk_id": title.lower().replace(" ", "-"),
        "title": title,
        "cluster_id": cluster_id,
        "relevance": relevance,
        "semantic_similarity": relevance,
        "focus_coverage": focus,
    }


class DiversityDiagnosticsTests(unittest.TestCase):
    def test_diagnose_query_marks_successful_cluster_breadth(self):
        diagnostic = diagnose_query(
            query(
                verdict="useful_breadth",
                added=[item("Sanctions Design", 0.5, 0.8)],
                useful_breadth_count=1,
                breadth_precision=1.0,
            )
        )
        self.assertEqual(diagnostic["failure_mode"], "successful_cluster_breadth")
        self.assertEqual(diagnostic["strong_additions"][0]["title"], "Sanctions Design")
        self.assertIn("Keep diversity pressure", diagnostic["recommendation"])

    def test_diagnose_query_marks_off_topic_breadth(self):
        diagnostic = diagnose_query(
            query(
                query_type="off_topic",
                verdict="off_topic_breadth_risk",
                added=[item("Cloud Computing", 0.0, 0.7)],
            )
        )
        self.assertEqual(diagnostic["failure_mode"], "off_topic_breadth")
        self.assertIn("off_topic_query", diagnostic["risk_signals"])

    def test_classify_failure_mode_marks_lost_more_focused_items(self):
        mode = classify_failure_mode(
            {"query_type": "synthesis"},
            {
                "cluster_delta": 1,
                "noise_rate": 0.0,
                "breadth_precision": 0.333333,
                "average_added_focus_coverage": 0.3,
                "relevance_delta": -0.1,
                "focus_coverage_delta": -0.4,
            },
        )
        self.assertEqual(mode, "lost_more_focused_items")

    def test_risk_signals_include_relevance_tradeoff(self):
        signals = risk_signals(
            {"query_type": "synthesis"},
            {
                "focus_term_count": 4,
                "rank_overlap": 0.2,
                "average_added_focus_coverage": 0.2,
                "added_count": 2,
                "relevance_delta": -0.3,
                "noise_rate": 0.25,
            },
        )
        self.assertIn("large_rank_shift", signals)
        self.assertIn("added_items_less_relevant_than_removed_items", signals)

    def test_aggregate_diagnostics_counts_failure_modes(self):
        diagnostics = [
            diagnose_query(
                query(
                    verdict="useful_breadth",
                    added=[item("Sanctions Design", 0.5, 0.8)],
                    useful_breadth_count=1,
                    breadth_precision=1.0,
                )
            ),
            diagnose_query(
                query(
                    query_type="off_topic",
                    verdict="off_topic_breadth_risk",
                    added=[item("Cloud Computing", 0.0, 0.7)],
                )
            ),
        ]
        aggregate = aggregate_diagnostics(diagnostics)
        self.assertEqual(aggregate["failure_mode_counts"]["successful_cluster_breadth"], 1)
        self.assertEqual(aggregate["failure_mode_counts"]["off_topic_breadth"], 1)


if __name__ == "__main__":
    unittest.main()
