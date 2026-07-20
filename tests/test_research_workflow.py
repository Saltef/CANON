import unittest
from unittest.mock import patch

from canon.product.research_workflow import (
    construct_research_question,
    run_research_workflow,
)


class ResearchWorkflowTests(unittest.TestCase):
    def test_construct_research_question_classifies_risk_report(self):
        parsed = construct_research_question(
            "What are the geopolitical risks of AI data center expansion in Latin America?"
        )

        self.assertEqual(parsed["question_type"], "risk_report")
        self.assertEqual(parsed["risk_level"], "high")
        self.assertEqual(parsed["routing"]["retrieval_lens"], "broad_discovery_with_conflicts")

    @patch("canon.product.research_workflow.service.evidence_packets")
    def test_research_workflow_runs_layers_1_to_6(self, evidence_packets):
        evidence_packets.return_value = packet_response()

        report = run_research_workflow(
            {
                "question": "What are the risks of AI data center water use in Brazil?",
                "mode": "m",
                "policy": "rag",
                "research_frame": {"regions": ["Brazil"]},
                "evidence_requirements": {
                    "top_k": 5,
                    "include_source_diversity": True,
                    "minimum_source_types": ["official", "local_media"],
                },
                "write_report": False,
            }
        )

        layers = report["layers"]
        self.assertEqual(report["report_id"], "research_workflow_layers_1_6_v1")
        self.assertIn("research_question", layers)
        self.assertIn("research_frame_constructor", layers)
        self.assertIn("semantic_retrieval_engine", layers)
        self.assertIn("evidence_neighborhood_mapper", layers)
        self.assertIn("concept_and_discipline_analyzer", layers)
        self.assertIn("research_guidance_layer", layers)
        self.assertEqual(layers["research_question"]["question_type"], "risk_report")
        self.assertIn("topic", layers["research_frame_constructor"]["constructed_fields"])
        self.assertEqual(layers["semantic_retrieval_engine"]["evidence_count"], 2)
        self.assertEqual(layers["evidence_neighborhood_mapper"]["distinct_sources"], 2)
        self.assertTrue(layers["concept_and_discipline_analyzer"]["disciplines"])
        self.assertEqual(
            layers["research_guidance_layer"]["synthesis_gate"]["status"],
            "blocked_pending_human_review",
        )

    @patch("canon.product.service.research_workflow")
    def test_service_route_wrapper_calls_workflow(self, workflow):
        from canon.product import service

        workflow.return_value = {"status": "ready_for_human_review"}

        report = service.research_workflow({"query": "q", "write_report": False})

        self.assertEqual(report["status"], "ready_for_human_review")


def packet_response():
    return {
        "report_id": "evidence_packet_response_v1",
        "status": "complete",
        "contract_validation": {"status": "pass"},
        "query_diagnostics": {
            "matched_terms": ["water"],
            "weak_terms": ["permit"],
            "field_phrases": ["grid stress"],
            "query_variants": [{"query": "water permits Brazil"}],
            "drift_risk": "needs_caution",
        },
        "frame_coverage": {
            "status": "partial",
            "diagnostics": [{"field": "source_types", "status": "missing"}],
        },
        "coverage_gaps": [{"gap": "Missing official source.", "severity": "medium"}],
        "retrieval_metrics": {
            "estimated_confidence": "medium",
            "source_diversity_status": "pass",
            "human_review_status": "not_reviewed",
            "support_confidence": 0.5,
        },
        "evidence_packets": [
            {
                "source_diversity": {
                    "distinct_sources": 2,
                    "distinct_clusters": 2,
                    "dominant_source_share": 0.5,
                    "warnings": [],
                },
                "supporting_evidence": [
                    {
                        "evidence_id": "C1",
                        "chunk_id": "chunk:1",
                        "document_id": "doc:1",
                        "title": "Brazil data center water risk",
                        "source_name": "Local News",
                        "source_type": "local_media",
                        "language": "English",
                        "domain": "ai infrastructure",
                        "cluster_id": "water",
                        "text": "Water and cooling risk for AI data centers.",
                    },
                    {
                        "evidence_id": "C2",
                        "chunk_id": "chunk:2",
                        "document_id": "doc:2",
                        "title": "Grid and water permits",
                        "source_name": "Regulator",
                        "source_type": "official",
                        "language": "English",
                        "domain": "energy policy",
                        "cluster_id": "energy",
                        "text": "Government permit review mentions grid stress.",
                    },
                ],
            }
        ],
    }


if __name__ == "__main__":
    unittest.main()
