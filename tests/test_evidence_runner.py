import unittest
from unittest.mock import patch

from canon.intelligence import evidence_runner


class EvidenceRunnerTests(unittest.TestCase):
    def test_runner_builds_grounded_brief_from_packet_response(self):
        report = evidence_runner.run_from_packet_response(
            packet_response=packet_response(),
            question="AI data center risk",
            mode="unit",
            policy="rag",
            write_report=False,
        )

        self.assertEqual(report["report_id"], evidence_runner.REPORT_ID)
        self.assertEqual(report["status"], "ready_for_human_review")
        self.assertEqual(len(report["agent_outputs"]), 5)
        self.assertEqual(report["grounding_report"]["status"], "pass")
        self.assertGreaterEqual(report["grounding_report"]["grounded_claim_ratio"], 0.95)
        self.assertEqual(report["report_quality"]["status"], "pass")
        self.assertEqual(report["evidence_packet_contract_validation"]["status"], "pass")
        self.assertEqual(report["evidence_scope_summary"], {"private_corpus": 5, "external_source": 0})
        self.assertEqual(report["external_expansion"]["status"], "planned")
        self.assertFalse(report["external_expansion"]["executed"])
        self.assertTrue(report["brief"]["citation_appendix"])

    def test_grounding_fails_claims_without_evidence_ids(self):
        grounding = evidence_runner.evaluate_grounding(
            [
                {
                    "agent_id": "energy_water",
                    "claims": [{"claim_id": "c1", "claim": "Unsupported", "evidence_ids": []}],
                }
            ],
            {"C1": {"evidence_id": "C1"}},
        )

        self.assertEqual(grounding["status"], "fail")
        self.assertEqual(grounding["unsupported_claim_count"], 1)
        self.assertEqual(grounding["unsupported_claims"][0]["reason"], "missing_evidence_ids")

    def test_red_team_blocks_unsupported_claims(self):
        review = evidence_runner.red_team_review(
            agent_outputs=[],
            grounding={
                "unsupported_claim_count": 1,
            },
            duplicate_report={"duplicate_agent_output_rate": 0.0},
            packet_response={"frame_coverage": {"status": "pass"}},
        )

        self.assertEqual(review["status"], "block")
        self.assertEqual(review["blocking_issue_count"], 1)

    def test_report_quality_fails_missing_sections_or_low_grounding(self):
        quality = evidence_runner.assess_report_quality(
            brief={
                "executive_summary": "",
                "what_changed": [],
                "evidence_backed_developments": [],
                "contradictions_and_uncertainty": "",
                "source_gaps": [],
                "next_signals_to_watch": [],
                "citation_appendix": [],
            },
            agent_outputs=[{"status": "complete"}],
            grounding={"claim_count": 2, "grounded_claim_ratio": 0.5},
            red_team={"blocking_issue_count": 0},
            duplicate_report={"duplicate_agent_output_rate": 0.0},
        )

        self.assertEqual(quality["status"], "fail")
        self.assertGreater(quality["failed_check_count"], 0)

    def test_duplicate_agent_output_rate_is_measured(self):
        report = evidence_runner.duplicate_agent_output_report(
            [
                {"agent_id": "a", "claims": [{"claim": "Same claim about energy"}]},
                {"agent_id": "b", "claims": [{"claim": "Same claim about energy"}]},
                {"agent_id": "c", "claims": [{"claim": "Different claim about water"}]},
            ]
        )

        self.assertEqual(report["duplicate_agent_count"], 2)
        self.assertAlmostEqual(report["duplicate_agent_output_rate"], 2 / 3, places=4)

    def test_run_intelligence_brief_calls_evidence_packet_service(self):
        with patch("canon.intelligence.evidence_runner.service.evidence_packets", return_value=packet_response()) as packets:
            report = evidence_runner.run_intelligence_brief(
                question="AI data center risk",
                mode="unit",
                write_report=False,
            )

        packets.assert_called_once()
        self.assertEqual(report["status"], "ready_for_human_review")


def packet_response():
    evidence = [
        evidence_item(
            "C1",
            "Chile Grid Interconnection Note",
            "Chile Energy Regulator",
            "official_report",
            "official",
            "en",
            "Chile",
            "Large data center electricity requests can increase grid interconnection pressure.",
        ),
        evidence_item(
            "C2",
            "Brazilian Local Coverage Of Data Center Water Permitting",
            "Jornal Local",
            "local_media",
            "local_media",
            "pt",
            "Brazil",
            "Residents and environmental groups question water consumption and cooling impacts.",
        ),
        evidence_item(
            "C3",
            "Cloud Provider Announces Latin America AI Region Expansion",
            "ExampleCloud",
            "company_announcement",
            "company",
            "en",
            "Latin America",
            "A cloud provider announced AI-capable cloud regions in Chile and Brazil.",
        ),
        evidence_item(
            "C4",
            "Policy Brief On Sovereign AI And Cloud Dependency",
            "Regional Digital Policy Institute",
            "policy_report",
            "policy",
            "es",
            "Latin America",
            "Foreign cloud dependency can limit sovereign AI and government bargaining capacity.",
        ),
        evidence_item(
            "C5",
            "Public Comments On Data Center Electricity Prices",
            "Municipal Hearing Sample",
            "public_comment_sample",
            "public_opinion",
            "es",
            "Mexico",
            "Residents raised public opinion concerns at a hearing about electricity prices.",
        ),
    ]
    return {
        "request_id": "req_unit",
        "project_id": "ai_infra_geo_risk",
        "status": "complete",
        "query": "AI data center risk",
        "mode": "unit",
        "policy": "rag",
        "evidence_packets": [
            {
                "packet_id": "packet_req_unit",
                "supporting_evidence": evidence,
                "conflicting_evidence": [],
                "limitations": [],
                "evidence_scope_summary": {"private_corpus": 5, "external_source": 0},
            }
        ],
        "contract_validation": {"status": "pass"},
        "external_expansion": {"status": "planned", "executed": False},
        "frame_coverage": {"status": "pass"},
        "coverage_gaps": [],
    }


def evidence_item(evidence_id, title, source, source_type, provenance, language, jurisdiction, text):
    return {
        "evidence_id": evidence_id,
        "chunk_id": f"chunk:{evidence_id}",
        "document_id": f"doc:{evidence_id}",
        "title": title,
        "source_name": source,
        "source_type": source_type,
        "provenance": provenance,
        "language": language,
        "jurisdiction": jurisdiction,
        "text": text,
        "rank": int(evidence_id.removeprefix("C")),
        "evidence_scope": "private_corpus",
        "retrieval_stage": "private_corpus_retrieval",
    }


if __name__ == "__main__":
    unittest.main()
