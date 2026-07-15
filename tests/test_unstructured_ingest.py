import json
import tempfile
import unittest
from pathlib import Path

from canon.config import load_settings
from canon.ingest.unstructured import (
    ingest_unstructured_jsonl,
    load_jsonl,
    sections_from_record,
    work_from_record,
)


class UnstructuredIngestTests(unittest.TestCase):
    def test_load_jsonl_requires_text_or_sections(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.jsonl"
            path.write_text(json.dumps({"id": "x", "title": "No text"}) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_jsonl(path)

    def test_work_from_record_preserves_document_metadata(self):
        record = {
            "id": "filing-1",
            "title": "Company 10-K Annual Report",
            "year": "2026-02-01",
            "source_name": "SEC EDGAR",
            "document_type": "regulatory_filing",
            "provenance": "official",
            "domain": "legal_market",
            "text": "The filing discloses market risk.",
        }
        work = work_from_record(record, 1)
        self.assertEqual(work.id, "filing-1")
        self.assertEqual(work.year, 2026)
        self.assertEqual(work.raw["document_type"], "regulatory_filing")
        self.assertEqual(work.raw["provenance"], "official")

    def test_sections_from_record_handles_named_sections(self):
        sections = sections_from_record(
            {
                "sections": [
                    {"section": "risk factors", "text": "Risk text"},
                    {"title": "discussion", "content": "Discussion text"},
                ]
            }
        )
        self.assertEqual(sections, [("risk factors", "Risk text"), ("discussion", "Discussion text")])

    def test_ingest_unstructured_fixture_writes_processed_files(self):
        settings = load_settings()
        mode = "unit_unstructured_ingest"
        report = ingest_unstructured_jsonl(
            settings.fixtures_dir / "unstructured_sample.jsonl",
            mode=mode,
            chunk_tokens=24,
            overlap_tokens=4,
        )
        self.assertEqual(report["work_count"], 4)
        self.assertGreaterEqual(report["chunk_count"], 4)
        self.assertIn("document_type_profile", report)
        self.assertEqual(report["document_type_profile"]["document_type_counts"]["regulatory_filing"], 1)
        self.assertEqual(report["document_type_profile"]["document_type_counts"]["unknown_unstructured_text"], 1)
        works_path = settings.data_dir / "processed" / f"works_{mode}.json"
        chunks_path = settings.data_dir / "processed" / f"chunks_{mode}.json"
        self.assertTrue(works_path.exists())
        self.assertTrue(chunks_path.exists())
        works = json.loads(works_path.read_text(encoding="utf-8"))
        self.assertEqual(works[0]["raw"]["document_type"], "regulatory_filing")

    def test_ai_infra_geo_risk_fixture_has_mvp_coverage_metadata(self):
        settings = load_settings()
        mode = "unit_ai_infra_geo_risk"
        report = ingest_unstructured_jsonl(
            settings.fixtures_dir / "ai_infra_geo_risk_sample.jsonl",
            mode=mode,
            chunk_tokens=32,
            overlap_tokens=4,
        )

        self.assertEqual(report["work_count"], 5)
        profile = report["document_type_profile"]
        self.assertIn("news_article", profile["document_type_counts"])
        self.assertIn("policy_report", profile["document_type_counts"])
        self.assertIn("transcript", profile["document_type_counts"])
        works_path = settings.data_dir / "processed" / f"works_{mode}.json"
        works = json.loads(works_path.read_text(encoding="utf-8"))
        languages = {work["language"] for work in works}
        domains = {(work.get("raw") or {}).get("domain") for work in works}
        raw_types = {(work.get("raw") or {}).get("document_type") for work in works}
        self.assertGreaterEqual(len(languages), 3)
        self.assertEqual(domains, {"ai_infra_geo_risk"})
        self.assertIn("official_report", raw_types)
        self.assertIn("local_media", raw_types)
        self.assertIn("company_announcement", raw_types)


if __name__ == "__main__":
    unittest.main()
