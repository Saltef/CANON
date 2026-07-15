import unittest

from canon.eval.contracts import CONTRACT_VERSION, validate_report
from canon.eval.security import evaluate_adversarial_rag_security


class ContractTests(unittest.TestCase):
    def test_security_report_satisfies_contract(self):
        report = evaluate_adversarial_rag_security()
        validation = validate_report(report, "adversarial_rag_security")
        self.assertEqual(report["contract_version"], CONTRACT_VERSION)
        self.assertEqual(report["contract_validation"]["status"], "pass")
        self.assertEqual(validation["status"], "pass")
        self.assertEqual(report["metrics"]["attack_context_exposure_count"], 0)

    def test_retrieval_contract_rejects_quarantined_allowed_result(self):
        report = retrieval_report()
        report["results"][0]["decision"]["safety_decision"] = "quarantine"
        validation = validate_report(report, "retrieval")
        self.assertEqual(validation["status"], "fail")
        self.assertTrue(
            any("blocked or quarantined evidence cannot be allowed" in error for error in validation["errors"])
        )

    def test_retrieval_contract_requires_signal_families(self):
        report = retrieval_report()
        del report["results"][0]["decision"]["stage"]["signals"]["source_trust"]
        validation = validate_report(report, "retrieval")
        self.assertEqual(validation["status"], "fail")
        self.assertTrue(any("source_trust" in error for error in validation["errors"]))

    def test_retrieval_contract_allows_committee_excluded_candidate(self):
        report = retrieval_report()
        report["results"] = []
        report["stage_summary"]["allowed_after_safety_gate"] = 1
        report["rejected_candidates"] = [
            {
                "chunk_id": "social",
                "work_id": "w-social",
                "title": "Public post",
                "committee_decision": "aggregate_only",
                "allowed_uses": ["audit", "aggregate_public_opinion"],
                "conflict_tags": ["aggregation_required"],
                "stage": {
                    "candidate": {"candidate_rank": 1, "lexical_relevance": 1.0},
                    "signals": {
                        "query_text_relevance": {"candidate_rank": 1, "lexical_relevance": 1.0},
                        "syntax_profile": {
                            "document_syntax_category": "public_expression",
                            "document_syntax_vector": {"public_expression": 1.0},
                            "query_syntax_targets": {"public_expression": 1.0},
                            "corpus_syntax_profile": {},
                        },
                        "evidence_role": {"method_signal": 0.0},
                        "source_trust": {"source_trust": 0.2},
                        "provenance": {"provenance": "social_media"},
                    },
                    "safety_gate": {
                        "decision": "allow",
                        "reasons": [],
                        "safety": {
                            "risk_score": 0.0,
                            "decision": "allow",
                            "categories": [],
                            "matched_patterns": [],
                        },
                    },
                    "committee_gate": {
                        "committee_decision": "aggregate_only",
                        "allowed_uses": ["audit", "aggregate_public_opinion"],
                    },
                },
            }
        ]

        validation = validate_report(report, "retrieval")

        self.assertEqual(validation["status"], "pass")

    def test_evidence_packet_response_contract_accepts_scope_and_expansion_boundary(self):
        validation = validate_report(evidence_packet_response(), "evidence_packet_response")

        self.assertEqual(validation["status"], "pass")
        self.assertEqual(validation["checked_counts"]["evidence_packets"], 1)
        self.assertEqual(validation["checked_counts"]["packet_supporting_evidence"], 2)

    def test_evidence_packet_response_contract_rejects_executed_expansion(self):
        report = evidence_packet_response()
        report["external_expansion"]["executed"] = True
        validation = validate_report(report, "evidence_packet_response")

        self.assertEqual(validation["status"], "fail")
        self.assertTrue(any("execution must be false" in error for error in validation["errors"]))

    def test_evidence_packet_response_contract_rejects_scope_summary_mismatch(self):
        report = evidence_packet_response()
        report["evidence_packets"][0]["evidence_scope_summary"]["external_source"] = 0
        validation = validate_report(report, "evidence_packet_response")

        self.assertEqual(validation["status"], "fail")
        self.assertTrue(any("evidence_scope_summary.external_source" in error for error in validation["errors"]))


def retrieval_report():
    return {
        "contract_version": CONTRACT_VERSION,
        "query": "democratic peace",
        "policy": "rag",
        "mode": "dry_run",
        "top_k": 1,
        "stage_summary": {
            "candidate_count": 1,
            "enriched_count": 1,
            "allowed_after_safety_gate": 1,
            "rejected_after_safety_gate": 0,
            "safety_decision_counts": {"allow": 1},
        },
        "rejected_candidates": [],
        "results": [
            {
                "rank": 1,
                "chunk_id": "c1",
                "work_id": "w1",
                "title": "Study",
                "source_name": "Journal",
                "year": 2024,
                "section": "abstract",
                "cluster_id": 1,
                "preview": "democratic peace lowers conflict",
                "final_score": 0.8,
                "base_score": 0.8,
                "diversity_bonus": 0.0,
                "components": {
                    "relevance": 1.0,
                    "semantic_similarity": 0.8,
                    "syntax_relevance": 0.4,
                    "source_quality": 0.7,
                    "citation_centrality": 0.0,
                    "author_score": 0.0,
                    "section_role": 0.8,
                    "claim_density": 0.3,
                    "method_signal": 0.1,
                    "limitation_signal": 0.0,
                    "evidence_specificity": 0.4,
                    "uncertainty_signal": 0.0,
                    "recency": 0.5,
                },
                "weights": {"relevance": 1.0},
                "decision": {
                    "score": 0.8,
                    "base_score": 0.8,
                    "decision": "allow",
                    "reasons": [],
                    "safety_decision": "allow",
                    "safety_risk": 0.0,
                    "utility": {},
                    "stage": {
                        "candidate": {"candidate_rank": 1, "lexical_relevance": 1.0},
                        "signals": {
                            "query_text_relevance": {"candidate_rank": 1, "lexical_relevance": 1.0},
                            "syntax_profile": {
                                "document_syntax_category": "generic_claim",
                                "document_syntax_vector": {"generic_claim": 1.0},
                                "query_syntax_targets": {"generic_claim": 1.0},
                                "corpus_syntax_profile": {},
                            },
                            "evidence_role": {"method_signal": 0.1},
                            "source_trust": {"source_trust": 0.7},
                            "provenance": {"provenance": "academic"},
                        },
                        "safety_gate": {
                            "decision": "allow",
                            "reasons": [],
                            "safety": {
                                "risk_score": 0.0,
                                "decision": "allow",
                                "categories": [],
                                "matched_patterns": [],
                            },
                        },
                    },
                },
                "explanation": {"summary": "ok"},
            }
        ],
    }


def evidence_packet_response():
    return {
        "report_id": "evidence_packet_response_v1",
        "contract_version": CONTRACT_VERSION,
        "request_id": "req_001",
        "project_id": "ai_infra_geo_risk",
        "status": "complete",
        "query": "AI data center risk",
        "mode": "m",
        "policy": "rag",
        "research_frame": {},
        "evidence_packets": [
            {
                "packet_id": "packet_req_001",
                "claim": "Retrieved evidence addresses: AI data center risk",
                "support_level": "mixed",
                "confidence": "medium",
                "supporting_evidence": [
                    {
                        "evidence_id": "C1",
                        "chunk_id": "chunk:1",
                        "document_id": "doc:1",
                        "citation": "C1",
                        "title": "Internal memo",
                        "source_name": "Team Drive",
                        "text": "Private corpus text.",
                        "evidence_scope": "private_corpus",
                        "retrieval_stage": "private_corpus_retrieval",
                    },
                    {
                        "evidence_id": "E1",
                        "chunk_id": "external:1",
                        "document_id": "external-doc:1",
                        "citation": "E1",
                        "title": "External filing",
                        "source_name": "Regulator",
                        "text": "External source text.",
                        "evidence_scope": "external_source",
                        "retrieval_stage": "external_expansion",
                    },
                ],
                "evidence_scope_summary": {"private_corpus": 1, "external_source": 1},
                "source_diversity": {
                    "distinct_sources": 2,
                    "distinct_clusters": 2,
                    "dominant_source_share": 0.5,
                    "warnings": [],
                },
            }
        ],
        "query_diagnostics": {},
        "frame_coverage": {},
        "coverage_gaps": [],
        "external_expansion": {
            "status": "planned",
            "enabled": True,
            "executed": False,
            "allowed_source_types": ["official"],
            "suggested_queries": [
                {
                    "query": "AI data center risk filing",
                    "reason": "Corroborate source.",
                    "allowed_source_types": ["official"],
                    "priority": "medium",
                    "review_status": "needs_human_approval",
                }
            ],
            "boundary": "No external source was queried.",
            "human_review_required": True,
        },
        "retrieval_metrics": {"estimated_confidence": "medium"},
    }


if __name__ == "__main__":
    unittest.main()
