import unittest

from canon.synthesis.answer import (
    build_evidence,
    claims_for_chunks,
    compose_answer,
    conflict_candidate_chunk_ids,
    conflict_aware_base_results,
    conflict_notes,
    grounded_answer_with_support,
    limitations,
    summarize_importance,
    support_assessment,
)


class SynthesisTests(unittest.TestCase):
    def test_claims_for_chunks_groups_only_retrieved_claims(self):
        claims = [
            {"id": "a", "chunk_id": "c1", "confidence": 0.2},
            {"id": "b", "chunk_id": "c1", "confidence": 0.9},
            {"id": "c", "chunk_id": "c2", "confidence": 1.0},
        ]
        grouped = claims_for_chunks(claims, [{"chunk_id": "c1"}])
        self.assertEqual([claim["id"] for claim in grouped["c1"]], ["b", "a"])
        self.assertNotIn("c2", grouped)

    def test_build_evidence_attaches_top_claim(self):
        results = [
            {
                "chunk_id": "c1",
                "work_id": "w1",
                "title": "A",
                "source_name": "J",
                "year": 2020,
                "rank": 1,
                "cluster_id": 1,
                "final_score": 0.7,
                "components": {"relevance": 1.0},
                "explanation": {"reasons": ["high_lexical_relevance"]},
                "preview": "preview",
            }
        ]
        evidence = build_evidence(results, {"c1": [{"id": "claim-a"}]})
        self.assertEqual(evidence[0]["citation_id"], "C1")
        self.assertEqual(evidence[0]["claim"]["id"], "claim-a")
        self.assertEqual(evidence[0]["explanation"]["reasons"], ["high_lexical_relevance"])

    def test_build_evidence_can_attach_safe_parent_context(self):
        chunks = [
            chunk("c1", "The sample uses records.", 0, 4),
            chunk("c2", "A limitation is selection bias.", 4, 9),
        ]
        results = [
            {
                "chunk_id": "c1",
                "work_id": "w1",
                "title": "A",
                "source_name": "J",
                "year": 2020,
                "rank": 1,
                "cluster_id": 1,
                "final_score": 0.7,
                "components": {"relevance": 1.0},
                "decision": {"safety_decision": "allow", "stage": {"signals": {"provenance": {"provenance": "academic"}}}},
                "explanation": {},
                "preview": "The sample uses records.",
            }
        ]

        evidence = build_evidence(results, {"c1": []}, chunks=chunks)

        self.assertTrue(evidence[0]["parent_context"]["included"])
        self.assertIn("selection bias", evidence[0]["parent_context"]["text"])

    def test_compose_answer_uses_claim_text(self):
        answer = compose_answer(
            "democratic peace",
            [
                {
                    "citation_id": "C1",
                    "title": "A",
                    "claim": {"stance": "supportive", "text": "Democracies rarely fight."},
                }
            ],
        )
        self.assertIn("C1 reports", answer)
        self.assertIn("Democracies rarely fight", answer)

    def test_summarize_importance_counts_diversity(self):
        summary = summarize_importance(
            [
                {
                    "work_id": "w1",
                    "source_name": "J1",
                    "cluster_id": 1,
                    "components": {"semantic_similarity": 1.0},
                },
                {
                    "work_id": "w2",
                    "source_name": "J2",
                    "cluster_id": 2,
                    "components": {"semantic_similarity": 0.0},
                },
            ]
        )
        self.assertEqual(summary["distinct_clusters"], 2)
        self.assertEqual(summary["average_components"]["semantic_similarity"], 0.5)

    def test_conflict_notes_filters_to_selected_claims(self):
        notes = conflict_notes(
            [{"left_claim_id": "a", "right_claim_id": "b"}],
            {"c1": [{"id": "a"}]},
        )
        self.assertEqual(len(notes), 1)

    def test_conflict_notes_can_use_query_topic_tokens(self):
        notes = conflict_notes(
            [{"left_claim_id": "a", "right_claim_id": "b", "shared_topic_tokens": ["democratic", "peace"]}],
            {"c1": [{"id": "z"}]},
            "democratic peace disagreement",
        )
        self.assertEqual(len(notes), 1)

    def test_conflict_notes_ignore_single_generic_query_overlap(self):
        notes = conflict_notes(
            [{"left_claim_id": "a", "right_claim_id": "b", "shared_topic_tokens": ["evidence"]}],
            {"c1": [{"id": "z"}]},
            "what evidence says",
        )
        self.assertEqual(notes, [])

    def test_conflict_notes_are_ranked_and_capped(self):
        conflicts = [
            {
                "id": f"conflict:{index}",
                "left_claim_id": f"a{index}",
                "right_claim_id": f"b{index}",
                "shared_topic_tokens": ["democratic", "peace"],
                "score": 0.8,
            }
            for index in range(20)
        ]
        notes = conflict_notes(conflicts, {"c1": [{"id": "z"}]}, "democratic peace")
        self.assertEqual(len(notes), 12)

    def test_conflict_candidate_chunk_ids_uses_query_or_selected_claims(self):
        chunk_ids = conflict_candidate_chunk_ids(
            conflicts=[
                {
                    "left_claim_id": "a",
                    "right_claim_id": "b",
                    "shared_topic_tokens": ["democratic", "peace"],
                },
                {
                    "left_claim_id": "c",
                    "right_claim_id": "d",
                    "shared_topic_tokens": ["unrelated"],
                },
            ],
            claims_by_id={
                "a": {"chunk_id": "c1"},
                "b": {"chunk_id": "c2"},
                "c": {"chunk_id": "c3"},
                "d": {"chunk_id": "c4"},
            },
            selected_claim_ids={"c"},
            query_tokens={"democratic", "peace"},
        )
        self.assertEqual(chunk_ids, ["c1", "c2", "c3", "c4"])

    def test_conflict_aware_base_results_prefers_focus_overlap(self):
        selected = conflict_aware_base_results(
            "democratic peace covert intervention",
            [
                {"title": "Generic evidence", "preview": "unrelated medical evidence", "final_score": 0.9},
                {"title": "Overt Peace, Covert War", "preview": "democratic intervention", "final_score": 0.7},
                {"title": "Public Opinion and the Democratic Peace", "preview": "democratic peace", "final_score": 0.6},
            ],
            2,
        )
        self.assertEqual([item["title"] for item in selected], ["Overt Peace, Covert War", "Public Opinion and the Democratic Peace"])

    def test_limitations_flags_low_diversity(self):
        limits = limitations(
            [{"claim": {"id": "a"}, "cluster_id": 1}, {"claim": None, "cluster_id": 1}]
        )
        self.assertIn("The answer is based on fewer than three retrieved evidence items.", limits)
        self.assertIn("The retrieved evidence has limited graph-cluster diversity.", limits)

    def test_support_assessment_flags_weak_off_topic_support(self):
        support = support_assessment(
            "quantum computing hardware error correction",
            [
                {
                    "title": "Public Opinion and the Democratic Peace",
                    "preview": "democratic peace and public opinion",
                    "claim": {"text": "Democracy changes threat perception."},
                }
            ],
            [{"components": {"relevance": 0.1, "semantic_similarity": 0.1}}],
        )
        self.assertEqual(support["support_level"], "weak")

    def test_support_assessment_caps_high_scores_without_focus_terms(self):
        support = support_assessment(
            "what does this corpus say about quantum computing hardware",
            [
                {
                    "title": "Public Opinion and the Democratic Peace",
                    "preview": "this article discusses democratic peace evidence",
                    "claim": {"text": "Democracy changes threat perception."},
                }
            ],
            [{"components": {"relevance": 0.95, "semantic_similarity": 0.95}}],
        )
        self.assertEqual(support["support_level"], "weak")
        self.assertEqual(support["covered_focus_terms"], [])
        self.assertIn("quantum", support["missing_focus_terms"])

    def test_support_assessment_keeps_acronyms_and_plural_matches(self):
        support = support_assessment(
            "AI risks",
            [
                {
                    "title": "AI Infrastructure Risk",
                    "preview": "Grid risk and AI infrastructure constraints.",
                    "claim": {"text": "AI systems can create infrastructure risk."},
                }
            ],
            [{"components": {"relevance": 0.8, "semantic_similarity": 0.8}}],
        )

        self.assertIn("ai", support["covered_focus_terms"])
        self.assertIn("risks", support["covered_focus_terms"])
        self.assertEqual(support["missing_focus_terms"], [])

    def test_grounded_answer_with_support_prepends_abstention(self):
        answer = grounded_answer_with_support("A normal answer.", {"support_level": "weak"})
        self.assertIn("abstention", answer)


def chunk(chunk_id, text, token_start, token_end):
    return {
        "id": chunk_id,
        "work_id": "w1",
        "section": "methods",
        "text": text,
        "token_start": token_start,
        "token_end": token_end,
        "importance": {
            "parent_context_id": "parent:w1-methods",
            "parent_context_available": 1.0,
            "parent_context_mode": "section",
            "parent_context_recommended": 1.0,
            "context_expansion_policy": "safe_parent_expansion_allowed",
            "generation_context_role": "evidence_context",
        },
    }


if __name__ == "__main__":
    unittest.main()
