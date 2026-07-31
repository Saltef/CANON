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
                with patch("canon.product.smoke.service.production_status", return_value=production_status_payload()):
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
                                    with patch(
                                        "canon.product.smoke.service.production_evidence_workbench",
                                        return_value=production_workbench_payload(),
                                    ):
                                        report = build_smoke_report(mode="demo", write_report=False)
        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["result_summary"]["answer_citation_count"], 1)
        self.assertEqual(report["result_summary"]["evidence_packet_contract_status"], "pass")
        self.assertEqual(report["result_summary"]["evidence_packet_run_diagnosis_status"], "ready_for_user_inspection")
        self.assertEqual(report["result_summary"]["production_workbench_evidence_count"], 1)
        self.assertEqual(
            report["result_summary"]["production_workbench_run_diagnosis_status"],
            "ready_for_user_inspection",
        )
        self.assertIn("GET /v1/production/status", report["endpoints"])
        self.assertIn("POST /v1/production/evidence-workbench", report["endpoints"])
        self.assertIn("POST /v1/query-diagnostics", report["endpoints"])
        self.assertIn("POST /v1/evidence-packets", report["endpoints"])

    def test_smoke_checks_fail_without_answer_citations(self):
        results = {
            "health": {"status": "ok"},
            "summary": {"claim_boundaries": {"claim": "bounded"}},
            "production_status": production_status_payload(),
            "answer": {**answer_payload(), "citations": []},
            "compare": compare_payload(),
            "query_diagnostics": query_diagnostics_payload(),
            "evidence_packets": evidence_packets_payload(),
            "production_workbench": production_workbench_payload(),
        }
        checks = smoke_checks(results)
        self.assertIn({"id": "answer_has_citations", "passed": False}, checks)

    def test_smoke_checks_fail_when_evidence_packet_contract_fails(self):
        results = {
            "health": {"status": "ok"},
            "summary": {"claim_boundaries": {"claim": "bounded"}},
            "production_status": production_status_payload(),
            "answer": answer_payload(),
            "compare": compare_payload(),
            "query_diagnostics": query_diagnostics_payload(),
            "evidence_packets": {
                **evidence_packets_payload(),
                "contract_validation": {"status": "fail"},
            },
            "production_workbench": production_workbench_payload(),
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
        "run_diagnosis": {
            "report_id": "run_diagnosis_v1",
            "overall_status": "ready_for_user_inspection",
            "failure_class": "human_review_required",
        },
        "contract_validation": {"status": "pass"},
    }


def production_status_payload():
    return {
        "status": "ship_user_experience_ready",
        "shippable_without_human_review": ["corpus-backed evidence packets"],
        "blocked_without_human_review": ["autonomous factual conclusions"],
    }


def production_workbench_payload():
    return {
        "status": "ready_for_user_inspection",
        "claim_boundary": {"safe_to_use_for": ["finding evidence"]},
        "evidence_cards": [{"evidence_id": "C1"}],
        "run_diagnosis": {
            "report_id": "run_diagnosis_v1",
            "overall_status": "ready_for_user_inspection",
            "failure_class": "human_review_required",
        },
    }


if __name__ == "__main__":
    unittest.main()
