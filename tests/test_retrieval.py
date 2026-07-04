import unittest

from canon.retrieval.bm25 import BM25Index
from canon.retrieval.corpus import RetrievalDocument
from canon.retrieval.engine import (
    DISAGREEMENT_MIN_DIVERSITY,
    apply_query_controls,
    excluded_query_terms,
    has_disagreement_intent,
    retrieve,
    select_with_diversity,
    weights_with_query_intent,
    weights_without_diversity,
)
from canon.retrieval.policies import normalize


class RetrievalTests(unittest.TestCase):
    def test_bm25_scores_matching_document_higher(self):
        index = BM25Index(["democratic peace conflict", "economic sanctions policy"])
        scores = index.scores("democratic peace")
        self.assertGreater(scores[0], scores[1])

    def test_normalize_handles_flat_values(self):
        self.assertEqual(normalize([0.0, 0.0]), [0.0, 0.0])
        self.assertEqual(normalize([2.0, 2.0]), [1.0, 1.0])

    def test_retrieve_returns_trace_items(self):
        documents = [
            RetrievalDocument(
                chunk_id="c1",
                work_id="w1",
                title="Democratic Peace",
                year=2020,
                source_name="Journal",
                section="abstract",
                text="democratic peace research finds lower conflict",
                chunk_importance={"section_role": 0.9, "claim_density": 0.5},
                work_signals={
                    "source_quality": 0.9,
                    "citation_centrality": 0.2,
                    "author_score": 0.8,
                    "recency": 0.8,
                },
                cluster_id=1,
            ),
            RetrievalDocument(
                chunk_id="c2",
                work_id="w2",
                title="Sanctions",
                year=2021,
                source_name="Journal",
                section="abstract",
                text="economic sanctions policy change",
                chunk_importance={"section_role": 0.9, "claim_density": 0.2},
                work_signals={
                    "source_quality": 0.8,
                    "citation_centrality": 0.1,
                    "author_score": 0.2,
                    "recency": 0.9,
                },
                cluster_id=2,
            ),
        ]
        trace = retrieve(
            "democratic peace",
            documents,
            weights={"relevance": 1.0, "source_quality": 0.0},
            top_k=1,
            preview_chars=50,
        )
        self.assertEqual(trace[0].chunk_id, "c1")
        self.assertIn("relevance", trace[0].components)
        self.assertIn("semantic_similarity", trace[0].components)
        self.assertEqual(trace[0].cluster_id, 1)

    def test_diversity_selection_prefers_new_cluster_when_close(self):
        documents = [
            RetrievalDocument("c1", "w1", "A", 2020, "J", "abstract", "a", {}, {}, cluster_id=1),
            RetrievalDocument("c2", "w2", "B", 2020, "J", "abstract", "b", {}, {}, cluster_id=1),
            RetrievalDocument("c3", "w3", "C", 2020, "J", "abstract", "c", {}, {}, cluster_id=2),
        ]
        selected = select_with_diversity(
            [
                (0.9, 0.9, documents[0], object()),
                (0.89, 0.89, documents[1], object()),
                (0.7, 0.7, documents[2], object()),
            ],
            top_k=2,
            diversity_weight=0.25,
        )
        self.assertEqual([row[2].chunk_id for row in selected], ["c1", "c3"])

    def test_disagreement_intent_lifts_diversity_weight(self):
        weights = weights_with_query_intent(
            "Where does democratic peace evidence conflict with covert intervention findings?",
            {"relevance": 1.0},
        )
        self.assertEqual(weights["diversity"], DISAGREEMENT_MIN_DIVERSITY)
        self.assertTrue(has_disagreement_intent("Which studies disagree?"))

    def test_neutral_query_keeps_diversity_weight(self):
        weights = weights_with_query_intent(
            "What predicts civil war recurrence?",
            {"relevance": 1.0},
        )
        self.assertNotIn("diversity", weights)

    def test_excluded_query_terms_follow_marker(self):
        self.assertEqual(
            excluded_query_terms(
                "How effective are sanctions, excluding democratic peace theory?"
            ),
            ["democratic", "peace", "theory"],
        )

    def test_query_controls_penalize_excluded_document_terms(self):
        document = RetrievalDocument(
            "c1",
            "w1",
            "Democratic Peace",
            2020,
            "J",
            "abstract",
            "democratic peace theory",
            {},
            {},
            cluster_id=1,
        )
        score = apply_query_controls(
            "sanctions excluding democratic peace theory",
            document,
            relevance=1.0,
            score=0.8,
            weights={"exclusion_penalty": 0.3},
        )
        self.assertLess(score, 0.8)

    def test_control_weights_do_not_enter_weighted_score_components(self):
        weights = weights_without_diversity(
            {
                "relevance": 1.0,
                "diversity": 0.5,
                "relevance_floor": 0.25,
                "low_relevance_penalty": 0.18,
                "exclusion_penalty": 0.3,
            }
        )
        self.assertEqual(weights, {"relevance": 1.0})


if __name__ == "__main__":
    unittest.main()
