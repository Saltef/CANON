import unittest

from canon.retrieval.corpus import RetrievalDocument
from canon.retrieval.query_diagnostics import diagnose_query


class QueryDiagnosticsTests(unittest.TestCase):
    def test_diagnostics_extracts_lingo_from_semantic_result_neighborhood(self):
        report = diagnose_query(
            "do sanctions work",
            [
                doc("c1", "economic sanctions effectiveness depends on sanctions compliance and enforcement"),
                doc("c2", "targeted sanctions coercive diplomacy can face sanctions evasion"),
                doc("c3", "humanitarian effects of sanctions complicate regime change outcomes"),
                doc("c4", "crop insurance adoption and rainfall shocks"),
            ],
            top_k=3,
            candidate_k=4,
            freedom_level="exploratory",
        )

        self.assertIn("sanctions", report["query_to_corpus"]["matched_terms"])
        self.assertIn("work", report["query_to_corpus"]["weak_terms"])
        phrases = report["result_neighborhood"]["field_phrases"]
        self.assertTrue(any("sanctions" in phrase for phrase in phrases))
        self.assertTrue(report["query_variants"])
        self.assertIn(report["stability"]["status"], {"stable", "sensitive", "unstable"})

    def test_reviewer_patterns_boost_accepted_and_filter_rejected_phrases(self):
        report = diagnose_query(
            "battery storage safety",
            [
                doc("c1", "battery storage safety concerns appear in public comments"),
                doc("c2", "interconnection queue delay and fire code compliance shape project risk"),
                doc("c3", "battery storage safety disclosure mentions emergency response planning"),
            ],
            top_k=3,
            candidate_k=3,
            freedom_level="exploratory",
            reviewer_patterns={
                "accepted_phrases": ["fire code compliance"],
                "rejected_phrases": ["public comments"],
            },
        )

        details = report["result_neighborhood"]["phrase_details"]
        by_phrase = {row["phrase"]: row for row in details}
        self.assertEqual(by_phrase["fire code compliance"]["reviewer_history"], "accepted")
        self.assertNotIn("public comments", by_phrase)

    def test_diagnostics_keep_domain_acronyms_and_plural_term_matches(self):
        report = diagnose_query(
            "AI risks",
            [
                doc("c1", "civil war grievances and mobilization"),
                doc("c2", "AI infrastructure risk depends on grid capacity and water use"),
            ],
            top_k=2,
            candidate_k=2,
            freedom_level="balanced",
        )

        self.assertIn("ai", report["query_to_corpus"]["matched_terms"])
        self.assertIn("risks", report["query_to_corpus"]["matched_terms"])
        self.assertNotIn("ai", report["query_to_corpus"]["weak_terms"])


def doc(chunk_id, text):
    return RetrievalDocument(
        chunk_id=chunk_id,
        work_id=f"work:{chunk_id}",
        title=f"Title {chunk_id}",
        year=2024,
        source_name="Journal",
        section="abstract",
        text=text,
        chunk_importance={
            "section_role": 0.9,
            "claim_density": 0.4,
            "method_signal": 0.0,
            "limitation_signal": 0.0,
            "evidence_specificity": 0.0,
            "uncertainty_signal": 0.0,
        },
        work_signals={
            "source_quality": 0.6,
            "source_trust": 0.6,
            "integrity_gate": 1.0,
            "citation_centrality": 0.1,
            "author_score": 0.0,
            "recency": 1.0,
        },
        cluster_id=1,
    )


if __name__ == "__main__":
    unittest.main()
