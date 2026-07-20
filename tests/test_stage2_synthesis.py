import unittest
from unittest.mock import patch

from canon.product.stage2_synthesis import run_stage2_synthesis, stage2_quality_gate


class Stage2SynthesisTests(unittest.TestCase):
    @patch("canon.product.stage2_synthesis.run_research_workflow")
    def test_stage2_synthesis_builds_cited_review_ready_report(self, workflow):
        workflow.return_value = workflow_report()

        report = run_stage2_synthesis(
            {
                "question": "What are the water and grid risks for AI data centers in Brazil?",
                "mode": "m",
                "policy": "rag",
                "write_report": False,
            }
        )

        self.assertEqual(report["report_id"], "stage2_evidence_synthesis_v1")
        self.assertEqual(report["status"], "ready_for_human_review")
        self.assertTrue(report["human_review_required"])
        self.assertEqual(len(report["cited_claims"]), 2)
        self.assertEqual(report["quality_gate"]["status"], "pass_pending_human_review")
        self.assertIn("[C1]", report["synthesis"]["answer"])

    def test_quality_gate_blocks_unknown_or_unsupported_citations(self):
        gate = stage2_quality_gate(
            claims=[
                {
                    "claim_id": "S2C1",
                    "claim": "Unsupported material about unrelated finance.",
                    "evidence_ids": ["C9"],
                }
            ],
            evidence=[{"evidence_id": "C1", "title": "Water", "text": "Water evidence.", "source_name": "A"}],
            workflow={"status": "ready"},
            min_evidence_count=1,
            min_distinct_sources=1,
        )

        self.assertEqual(gate["status"], "blocked")
        failed = {row["id"] for row in gate["checks"] if row["status"] == "fail"}
        self.assertIn("citation_integrity", failed)
        self.assertIn("claim_evidence_overlap", failed)


def workflow_report():
    return {
        "report_id": "research_workflow_layers_1_6_v1",
        "status": "needs_revision",
        "mode": "m",
        "policy": "rag",
        "layers": {
            "research_guidance_layer": {
                "next_actions": [
                    {"description": "Review water permit coverage."},
                ],
                "synthesis_gate": {"status": "blocked_pending_human_review"},
            }
        },
        "evidence_packet_response": {
            "coverage_gaps": [{"gap": "Missing local community evidence.", "severity": "medium"}],
            "evidence_packets": [
                {
                    "supporting_evidence": [
                        {
                            "evidence_id": "C1",
                            "chunk_id": "chunk:1",
                            "document_id": "doc:1",
                            "title": "Brazil water permits",
                            "source_name": "Regulator",
                            "source_type": "official",
                            "language": "English",
                            "domain": "water_environment",
                            "cluster_id": "water",
                            "text": "Brazil water permits mention cooling demand for AI data centers.",
                        },
                        {
                            "evidence_id": "C2",
                            "chunk_id": "chunk:2",
                            "document_id": "doc:2",
                            "title": "Grid stress",
                            "source_name": "Local News",
                            "source_type": "local_media",
                            "language": "English",
                            "domain": "energy_policy",
                            "cluster_id": "grid",
                            "text": "Local reporting says grid stress may grow near new data center sites.",
                        },
                    ]
                }
            ],
        },
    }


if __name__ == "__main__":
    unittest.main()
