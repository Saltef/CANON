import unittest

from canon.retrieval.corpus import RetrievalDocument
from canon.retrieval.engine import retrieve
from canon.retrieval.syntax import (
    corpus_syntax_profile,
    document_syntax_vector,
    primary_syntax_category,
    query_syntax_targets,
    syntax_relevance_scores,
)


class SyntaxRetrievalTests(unittest.TestCase):
    def test_query_syntax_targets_detect_evidence_form(self):
        targets = query_syntax_targets("What methods and sampling limitations did the study use?")

        self.assertGreater(targets["method_detail"], 0.0)
        self.assertGreater(targets["limitation_caveat"], 0.0)
        self.assertEqual(targets["generic_claim"], 0.0)

    def test_document_syntax_vector_uses_metadata_and_text_form(self):
        vector = document_syntax_vector(
            doc(
                "legal",
                "The court held that section 10 applies to the defendant.",
                document_type="legal_authority",
                evidence_family="official_record",
            )
        )

        self.assertEqual(primary_syntax_category(vector), "legal_authority")
        self.assertGreaterEqual(vector["legal_authority"], 0.85)

    def test_corpus_profile_boosts_discriminative_relevant_syntax(self):
        documents = [
            doc("generic1", "Battery storage is important for the market."),
            doc("generic2", "Battery storage may affect the market."),
            doc(
                "method",
                "Methods: the sample uses interview coding and regression design.",
                importance={"method_signal": 0.9, "claim_density": 0.1},
            ),
        ]
        scores = syntax_relevance_scores("What methods and sample design were used?", documents)
        profile = corpus_syntax_profile(documents)

        self.assertGreater(scores[2], scores[0])
        self.assertGreater(profile.category_weights["method_detail"], 0.0)

    def test_retrieval_can_rank_by_learned_syntax_component(self):
        documents = [
            doc(
                "generic",
                "Battery storage adoption changes market expectations.",
                importance={"claim_density": 0.7},
            ),
            doc(
                "method",
                "Methods: the sample design codes battery storage projects by region.",
                importance={"method_signal": 0.95, "claim_density": 0.1},
            ),
        ]

        trace = retrieve(
            "What methods and sample design were used for battery storage?",
            documents,
            weights={"syntax_relevance": 1.0, "relevance": 0.0, "semantic_similarity": 0.0},
            top_k=1,
            preview_chars=80,
        )

        self.assertEqual(trace[0].chunk_id, "method")
        self.assertIn("syntax_relevance", trace[0].components)
        self.assertIn("high_syntax_form_alignment", trace[0].explanation["reasons"])


def doc(
    chunk_id,
    text,
    importance=None,
    document_type="unknown_unstructured_text",
    evidence_family="unknown",
):
    return RetrievalDocument(
        chunk_id=chunk_id,
        work_id=f"work:{chunk_id}",
        title=f"Title {chunk_id}",
        year=2024,
        source_name="Fixture",
        section="body",
        text=text,
        chunk_importance={
            "section_role": 0.5,
            "claim_density": 0.0,
            "method_signal": 0.0,
            "limitation_signal": 0.0,
            "evidence_specificity": 0.0,
            "uncertainty_signal": 0.0,
            **(importance or {}),
        },
        work_signals={
            "source_quality": 0.5,
            "source_trust": 0.5,
            "integrity_gate": 1.0,
            "document_type": document_type,
            "evidence_family": evidence_family,
            "provenance": "academic",
            "domain": "general_academic",
        },
        cluster_id=1,
    )


if __name__ == "__main__":
    unittest.main()
