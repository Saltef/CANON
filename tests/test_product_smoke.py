import unittest
from unittest.mock import patch

from canon.product.smoke import build_smoke_report, smoke_checks


class ProductSmokeTests(unittest.TestCase):
    def test_smoke_report_passes_when_product_workflows_return_required_fields(self):
        with patch("canon.product.smoke.service.health", return_value={"status": "ok"}):
            with patch(
                "canon.product.smoke.service.product_summary",
                return_value={"product": "CANON Evidence Workbench", "claim_boundaries": {"claim": "bounded"}},
            ):
                with patch("canon.product.smoke.service.answer", return_value=answer_payload()):
                    with patch("canon.product.smoke.service.compare_retrieval", return_value=compare_payload()):
                        with patch(
                            "canon.product.smoke.service.query_diagnostics",
                            return_value=query_diagnostics_payload(),
                        ):
                            with patch(
                                "canon.product.smoke.service.evidence_packets",
                                return_value=evidence_packets_payload(),
                            ):
                                report = build_smoke_report(mode="demo", write_report=False)
        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["result_summary"]["answer_citation_count"], 1)
        self.assertEqual(report["result_summary"]["evidence_packet_contract_status"], "pass")
        self.assertIn("POST /v1/query-diagnostics", report["endpoints"])
        self.assertIn("POST /v1/evidence-packets", report["endpoints"])

    def test_smoke_checks_fail_without_answer_citations(self):
        results = {
            "health": {"status": "ok"},
            "summary": {"claim_boundaries": {"claim": "bounded"}},
            "answer": {**answer_payload(), "citations": []},
            "compare": compare_payload(),
            "query_diagnostics": query_diagnostics_payload(),
            "evidence_packets": evidence_packets_payload(),
        }
        checks = smoke_checks(results)
        self.assertIn({"id": "answer_has_citations", "passed": False}, checks)

    def test_smoke_checks_fail_when_evidence_packet_contract_fails(self):
        results = {
            "health": {"status": "ok"},
            "summary": {"claim_boundaries": {"claim": "bounded"}},
            "answer": answer_payload(),
            "compare": compare_payload(),
            "query_diagnostics": query_diagnostics_payload(),
            "evidence_packets": {
                **evidence_packets_payload(),
                "contract_validation": {"status": "fail"},
            },
        }
        checks = smoke_checks(results)
        self.assertIn({"id": "evidence_packets_contract_passed", "passed": False}, checks)


def answer_payload():
    return {
        "query": "Do sanctions work?",
        "citations": [{"citation_id": "C1"}],
        "evidence": [{"citation_id": "C1"}],
        "support_assessment": {"support_level": "supported"},
        "query_diagnostics": query_diagnostics_payload(),
    }


def compare_payload():
    return {
        "policies": ["lexical", "rag"],
        "rank_table": [],
        "runs": [{"policy": "lexical", "results": []}, {"policy": "rag", "results": []}],
    }


def query_diagnostics_payload():
    return {
        "query_to_corpus": {"matched_terms": ["sanctions"], "weak_terms": ["work"]},
        "query_variants": [{"query": "economic sanctions"}],
        "stability": {"status": "sensitive"},
    }


def evidence_packets_payload():
    return {
        "report_id": "evidence_packet_response_v1",
        "status": "complete",
        "evidence_packets": [
            {
                "packet_id": "packet_smoke",
                "supporting_evidence": [{"evidence_id": "C1"}],
                "evidence_scope_summary": {"private_corpus": 1, "external_source": 0},
            }
        ],
        "external_expansion": {"status": "planned", "executed": False},
        "contract_validation": {"status": "pass"},
    }


if __name__ == "__main__":
    unittest.main()
