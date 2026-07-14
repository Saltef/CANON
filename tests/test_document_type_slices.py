import unittest

from canon.config import load_settings
from canon.corpus.build import build_named_corpus
from canon.eval.document_type_slices import evaluate_document_type_slices, parse_queries
from canon.ingest.unstructured import ingest_unstructured_jsonl


class DocumentTypeSliceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        settings = load_settings()
        cls.mode = "unit_document_type_slices"
        cls.corpus_id = "unit_document_type_slices_corpus"
        ingest_unstructured_jsonl(
            settings.fixtures_dir / "unstructured_sample.jsonl",
            mode=cls.mode,
            chunk_tokens=24,
            overlap_tokens=4,
        )
        build_named_corpus(cls.corpus_id, from_modes=[cls.mode])

    def test_document_type_slice_report_contract_passes(self):
        report = evaluate_document_type_slices(
            mode=self.corpus_id,
            policies=["lexical", "rag"],
            top_k=3,
            queries=[
                {
                    "id": "unit-legal",
                    "query": "legal authority disclosure duty contract risk appellate court",
                    "intent": "legal_authority_lookup",
                },
                {
                    "id": "unit-web",
                    "query": "battery storage API key hidden instruction",
                    "intent": "open_web_safety_probe",
                },
            ],
            write_report=False,
        )
        self.assertEqual(report["report_id"], "document_type_slices_v1")
        self.assertEqual(report["contract_validation"]["status"], "pass")
        self.assertEqual(len(report["runs"]), 4)
        self.assertIn("legal_authority", report["corpus_document_type_counts"])
        self.assertIn("regulatory_filing", report["corpus_document_type_counts"])
        self.assertIn("web_page", report["corpus_document_type_counts"])
        self.assertIn("slice_summary", report)
        self.assertGreaterEqual(report["slice_summary"]["web_page"]["safety_rejected_count"], 1)
        self.assertIn("web_page", report["coverage_summary"]["rejected_document_type_coverage"])

    def test_inline_query_parser_accepts_separator(self):
        queries = parse_queries("first query || second query")
        self.assertEqual([query["id"] for query in queries], ["custom-1", "custom-2"])
        self.assertEqual(queries[1]["query"], "second query")


if __name__ == "__main__":
    unittest.main()
