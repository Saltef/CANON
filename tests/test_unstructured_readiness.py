import unittest

from canon.config import load_settings
from canon.corpus.build import build_named_corpus
from canon.eval.document_type_slices import evaluate_document_type_slices
from canon.eval.domain_slices import evaluate_domain_slices
from canon.eval.unstructured import evaluate_mixed_unstructured
from canon.eval.unstructured_readiness import DEFAULT_THRESHOLDS, readiness_checks, run_unstructured_readiness_gate
from canon.ingest.unstructured import ingest_unstructured_jsonl
from canon.reports.data_card import build_data_card


class UnstructuredReadinessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        settings = load_settings()
        ingest_unstructured_jsonl(
            settings.fixtures_dir / "unstructured_sample.jsonl",
            mode="unstructured_demo",
            chunk_tokens=24,
            overlap_tokens=4,
        )
        build_named_corpus("unstructured_demo_corpus", from_modes=["unstructured_demo"])
        evaluate_mixed_unstructured(write_report=True)
        evaluate_document_type_slices(
            "unstructured_demo_corpus",
            policies=["lexical", "balanced", "rag"],
            top_k=3,
            write_report=True,
        )
        build_data_card("unstructured_demo_corpus")
        ingest_unstructured_jsonl(
            settings.fixtures_dir / "mixed_domain_sample.jsonl",
            mode="mixed_domain_demo",
            chunk_tokens=32,
            overlap_tokens=4,
        )
        build_named_corpus("mixed_domain_demo_corpus", from_modes=["mixed_domain_demo"])
        evaluate_domain_slices(
            "mixed_domain_demo_corpus",
            policies=["lexical", "balanced", "rag"],
            top_k=4,
            write_report=True,
        )
        build_data_card("mixed_domain_demo_corpus")

    def test_readiness_gate_passes_generated_fixture_reports(self):
        report = run_unstructured_readiness_gate(write_report=False)
        self.assertEqual(report["report_id"], "heterogeneous_unstructured_readiness_v1")
        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["contract_validation"]["status"], "pass")
        self.assertTrue(any("Not allowed" in boundary for boundary in report["claim_boundaries"]))

    def test_readiness_checks_fail_when_required_artifacts_are_missing(self):
        checks = readiness_checks({}, DEFAULT_THRESHOLDS)
        self.assertTrue(any(not item["passed"] for item in checks))
        failed_ids = {item["id"] for item in checks if not item["passed"]}
        self.assertIn("mixed_unstructured_contract", failed_ids)
        self.assertIn("domain_slices_contract", failed_ids)


if __name__ == "__main__":
    unittest.main()
