import unittest
from unittest.mock import patch

from canon.product.stage2_synthesis import (
    build_cited_claims,
    build_disagreement_map,
    build_model_cited_claims,
    run_stage2_synthesis,
    stage2_quality_gate,
)


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
        self.assertIn("disagreement_map", report)
        self.assertIn("stage2_metrics", report)
        self.assertTrue(report["cited_claims"][0]["links"])
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

    def test_claim_model_supports_many_to_many_links_and_hedges(self):
        evidence = [
            evidence_row("C1", "Grid pressure may increase near data centers.", "Regulator"),
            evidence_row("C2", "Grid pressure may increase near data centers.", "Local News"),
            evidence_row("C3", "Grid pressure does not increase near data centers.", "Regulator"),
        ]

        claims = build_cited_claims("grid pressure data centers", evidence, max_claims=3)
        disagreement = build_disagreement_map(claims, evidence)

        self.assertTrue(any(len(claim["links"]) > 1 for claim in claims))
        self.assertTrue(any(link["hedge_score"] > 0 for claim in claims for link in claim["links"]))
        self.assertIn("claim_clusters", disagreement)
        self.assertGreaterEqual(disagreement["claim_cluster_count"], 1)
        self.assertGreaterEqual(disagreement["contested_cluster_count"], 1)
        all_stances = {
            link["stance"]
            for claim in claims
            for link in claim["links"]
        }
        self.assertIn("qualifies", all_stances)
        self.assertIn("contradicts", all_stances)

    @patch("canon.product.stage2_synthesis.call_stage2_model")
    def test_model_backed_claim_builder_uses_model_links(self, model_call):
        model_call.return_value = {
            "claims": [
                {
                    "claim": "Grid pressure near data centers is contested.",
                    "confidence": 0.82,
                    "disagreement_axis": "measurement_or_operationalization",
                    "links": [
                        {
                            "evidence_id": "C1",
                            "stance": "supports",
                            "entailment_score": 0.91,
                            "hedge_score": 0.2,
                            "excerpt": "Grid pressure may increase near data centers.",
                        },
                        {
                            "evidence_id": "C3",
                            "stance": "contradicts",
                            "entailment_score": 0.88,
                            "hedge_score": 0.1,
                            "excerpt": "Grid pressure does not increase near data centers.",
                        },
                    ],
                }
            ]
        }
        evidence = [
            evidence_row("C1", "Grid pressure may increase near data centers.", "Regulator"),
            evidence_row("C3", "Grid pressure does not increase near data centers.", "Regulator"),
        ]

        claims = build_model_cited_claims(
            question="grid pressure data centers",
            evidence=evidence,
            max_claims=3,
            provider="openrouter",
            model="test/model",
            allow_external_data=True,
        )

        self.assertEqual(len(claims), 1)
        self.assertTrue(claims[0]["contested"])
        self.assertEqual(claims[0]["support_status"], "contested")
        self.assertEqual(claims[0]["evidence_ids"], ["C1", "C3"])
        model_call.assert_called_once()

    @patch("canon.product.stage2_synthesis.call_stage2_model")
    @patch("canon.product.stage2_synthesis.run_research_workflow")
    def test_stage2_synthesis_reports_model_backed_status(self, workflow, model_call):
        workflow.return_value = workflow_report()
        model_call.return_value = {
            "claims": [
                {
                    "claim": "Water permitting creates a data center cooling constraint.",
                    "confidence": 0.9,
                    "disagreement_axis": "qualification_or_boundary_condition",
                    "links": [
                        {
                            "evidence_id": "C1",
                            "stance": "qualifies",
                            "entailment_score": 0.9,
                            "hedge_score": 0.2,
                            "excerpt": "Brazil water permits mention cooling demand for AI data centers.",
                        }
                    ],
                }
            ]
        }

        report = run_stage2_synthesis(
            {
                "question": "What are the water and grid risks for AI data centers in Brazil?",
                "mode": "m",
                "policy": "rag",
                "stage2_model_provider": "openrouter",
                "stage2_model": "test/model",
                "allow_external_stage2_data": True,
                "min_distinct_sources": 1,
                "write_report": False,
            }
        )

        self.assertEqual(report["stage2_model"]["status"], "model_backed")
        self.assertEqual(report["cited_claims"][0]["links"][0]["stance"], "qualifies")

    def test_external_model_requires_explicit_data_approval(self):
        with self.assertRaisesRegex(RuntimeError, "allow_external_stage2_data"):
            build_model_cited_claims(
                question="grid pressure data centers",
                evidence=[evidence_row("C1", "Grid pressure may increase near data centers.", "Regulator")],
                max_claims=1,
                provider="openrouter",
                model="test/model",
            )


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


def evidence_row(evidence_id, text, source):
    return {
        "evidence_id": evidence_id,
        "chunk_id": f"chunk:{evidence_id}",
        "document_id": f"doc:{evidence_id}",
        "title": "Grid Evidence",
        "source_name": source,
        "source_type": "fixture",
        "language": "English",
        "domain": "energy_policy",
        "cluster_id": "grid",
        "text": text,
        "metadata": {},
    }


if __name__ == "__main__":
    unittest.main()
