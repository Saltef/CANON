import unittest

from canon.config import load_settings
from canon.corpus.build import build_named_corpus
from canon.eval.domain_slices import evaluate_domain_slices
from canon.ingest.unstructured import ingest_unstructured_jsonl


class DomainSliceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        settings = load_settings()
        cls.mode = "unit_mixed_domain_slices"
        cls.corpus_id = "unit_mixed_domain_slices_corpus"
        ingest_unstructured_jsonl(
            settings.fixtures_dir / "mixed_domain_sample.jsonl",
            mode=cls.mode,
            chunk_tokens=32,
            overlap_tokens=4,
        )
        build_named_corpus(cls.corpus_id, from_modes=[cls.mode])

    def test_domain_slice_report_contract_passes(self):
        report = evaluate_domain_slices(
            mode=self.corpus_id,
            policies=["lexical", "rag"],
            top_k=4,
            queries=[
                {
                    "id": "unit-economics",
                    "query": "economics identification data robustness causal effect",
                    "intent": "economics_method_fit",
                },
                {
                    "id": "unit-history",
                    "query": "history archival primary source provenance historiography",
                    "intent": "history_primary_source_fit",
                },
                {
                    "id": "unit-web-attack",
                    "query": "battery storage market API key hidden instruction",
                    "intent": "legal_market_safety_probe",
                },
            ],
            write_report=False,
        )
        self.assertEqual(report["report_id"], "domain_slices_v1")
        self.assertEqual(report["contract_validation"]["status"], "pass")
        self.assertIn("economics", report["corpus_domain_counts"])
        self.assertIn("psychology", report["corpus_domain_counts"])
        self.assertIn("history", report["corpus_domain_counts"])
        self.assertIn("legal_market", report["corpus_domain_counts"])
        self.assertGreaterEqual(report["coverage_summary"]["corpus_domain_coverage_count"], 5)
        self.assertGreaterEqual(report["slice_summary"]["legal_market"]["safety_rejected_count"], 1)


if __name__ == "__main__":
    unittest.main()
