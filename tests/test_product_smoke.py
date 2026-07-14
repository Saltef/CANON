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
                            report = build_smoke_report(mode="demo", write_report=False)
        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["result_summary"]["answer_citation_count"], 1)
        self.assertIn("POST /v1/query-diagnostics", report["endpoints"])

    def test_smoke_checks_fail_without_answer_citations(self):
        results = {
            "health": {"status": "ok"},
            "summary": {"claim_boundaries": {"claim": "bounded"}},
            "answer": {**answer_payload(), "citations": []},
            "compare": compare_payload(),
            "query_diagnostics": query_diagnostics_payload(),
        }
        checks = smoke_checks(results)
        self.assertIn({"id": "answer_has_citations", "passed": False}, checks)


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


if __name__ == "__main__":
    unittest.main()
