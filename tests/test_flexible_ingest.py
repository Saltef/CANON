import json
import tempfile
import unittest
from pathlib import Path

from canon.config import load_settings
from canon.ingest.flexible import ingest_flexible_source, profile_source


class FlexibleIngestTests(unittest.TestCase):
    def test_profile_detects_crm_csv_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "crm.csv"
            path.write_text(
                "id,account_id,opportunity_stage,notes,owner\n"
                "n1,acme,evaluation,Customer needs better evidence review,Ada\n",
                encoding="utf-8",
            )

            profile = profile_source(path)

        self.assertEqual(profile["detected_format"], "csv")
        self.assertEqual(profile["source_shape"], "crm_record")
        self.assertEqual(profile["proposed_mapping"]["document_type"], "crm_record")
        self.assertEqual(profile["chunking_strategy"], "field_aware_record")

    def test_ingest_csv_normalizes_records_and_writes_processed_files(self):
        settings = load_settings()
        mode = "unit_flexible_csv"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "crm.csv"
            path.write_text(
                "id,account_id,opportunity_stage,notes,owner\n"
                "n1,acme,evaluation,Customer asked for renewal-risk evidence,Ada\n"
                "n2,bravo,proposal,,Ben\n",
                encoding="utf-8",
            )

            report = ingest_flexible_source(path, mode=mode, chunk_tokens=24, overlap_tokens=4)

        self.assertEqual(report["raw_record_count"], 2)
        self.assertEqual(report["normalized_record_count"], 1)
        self.assertEqual(report["skipped_reasons"]["missing_text"], 1)
        self.assertGreaterEqual(report["chunk_count"], 1)
        works_path = settings.data_dir / "processed" / f"works_{mode}.json"
        works = json.loads(works_path.read_text(encoding="utf-8"))
        self.assertEqual(works[0]["raw"]["document_type"], "crm_record")
        self.assertEqual(works[0]["raw"]["source_shape"], "crm_record")

    def test_ingest_text_folder_profiles_mixed_source(self):
        settings = load_settings()
        mode = "unit_flexible_folder"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "note.md").write_text("# Topic\n\nInterconnection delays affect delivery.", encoding="utf-8")
            (root / "memo.txt").write_text("Financing costs may delay storage projects.", encoding="utf-8")

            report = ingest_flexible_source(root, mode=mode, chunk_tokens=24, overlap_tokens=4)

        self.assertEqual(report["profile"]["source_shape"], "mixed_source")
        self.assertEqual(report["normalized_record_count"], 2)
        self.assertGreaterEqual(report["chunk_count"], 2)
        raw_path = settings.data_dir / "raw" / f"unstructured_{mode}.json"
        self.assertTrue(raw_path.exists())

    def test_profile_research_json_records(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "papers.json"
            path.write_text(
                json.dumps(
                    {
                        "records": [
                            {
                                "id": "paper-1",
                                "title": "Evidence Systems",
                                "authors": ["A. Scholar"],
                                "doi": "10.123/example",
                                "abstract": "We study evidence retrieval systems.",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            profile = profile_source(path)

        self.assertEqual(profile["source_shape"], "research_paper")
        self.assertEqual(profile["proposed_mapping"]["document_type"], "academic_article")
        self.assertEqual(profile["proposed_mapping"]["confidence"], "high")


if __name__ == "__main__":
    unittest.main()
