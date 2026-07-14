import json
import tempfile
import unittest
from pathlib import Path

from canon.corpus.build import cli_summary, merge_modes, mode_files_exist, raw_mode_path, raw_source_kind, resumed_topic_run, validation_warnings


class CorpusBuildTests(unittest.TestCase):
    def test_merge_modes_deduplicates_and_preserves_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory)
            (data_dir / "raw").mkdir()
            (data_dir / "processed").mkdir()
            for mode in ["a", "b"]:
                (data_dir / "raw" / f"openalex_{mode}.json").write_text(
                    json.dumps([{"id": "W1", "title": "Title", "publication_year": 2020}]),
                    encoding="utf-8",
                )
                (data_dir / "processed" / f"works_{mode}.json").write_text(
                    json.dumps([{"id": "W1", "title": "Title", "year": 2020}]),
                    encoding="utf-8",
                )
                (data_dir / "processed" / f"chunks_{mode}.json").write_text(
                    json.dumps([{"id": "C1", "work_id": "W1", "text": "Text"}]),
                    encoding="utf-8",
                )
            raw, works, chunks, provenance = merge_modes(data_dir, ["a", "b"])
        self.assertEqual(len(raw), 1)
        self.assertEqual(len(works), 1)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(works[0]["corpus_source_modes"], ["a", "b"])
        self.assertEqual(provenance["W1"]["source_modes"], ["a", "b"])

    def test_validation_warnings_detect_low_coverage(self):
        warnings = validation_warnings(
            {"policy_summary": {"rag": {"average_claim_citation_count": 0, "average_distinct_clusters": 1}}},
            {"claim_count": 0},
            {"conflict_count": 0},
        )
        self.assertGreaterEqual(len(warnings), 4)

    def test_mode_files_exist_requires_raw_works_and_chunks(self):
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory)
            (data_dir / "raw").mkdir()
            (data_dir / "processed").mkdir()
            (data_dir / "raw" / "openalex_m.json").write_text("[]", encoding="utf-8")
            (data_dir / "processed" / "works_m.json").write_text("[]", encoding="utf-8")
            self.assertFalse(mode_files_exist(data_dir, "m"))
            (data_dir / "processed" / "chunks_m.json").write_text("[]", encoding="utf-8")
            self.assertTrue(mode_files_exist(data_dir, "m"))

    def test_mode_files_exist_accepts_unstructured_raw_source(self):
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory)
            (data_dir / "raw").mkdir()
            (data_dir / "processed").mkdir()
            (data_dir / "raw" / "unstructured_m.json").write_text("[]", encoding="utf-8")
            (data_dir / "processed" / "works_m.json").write_text("[]", encoding="utf-8")
            (data_dir / "processed" / "chunks_m.json").write_text("[]", encoding="utf-8")
            self.assertTrue(mode_files_exist(data_dir, "m"))
            self.assertEqual(raw_mode_path(data_dir, "m").name, "unstructured_m.json")
            self.assertEqual(raw_source_kind(raw_mode_path(data_dir, "m")), "unstructured_jsonl")

    def test_merge_modes_accepts_unstructured_raw_source(self):
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory)
            (data_dir / "raw").mkdir()
            (data_dir / "processed").mkdir()
            (data_dir / "raw" / "unstructured_m.json").write_text(
                json.dumps([{"id": "U1", "title": "Filing", "document_type": "regulatory_filing"}]),
                encoding="utf-8",
            )
            (data_dir / "processed" / "works_m.json").write_text(
                json.dumps([{"id": "U1", "title": "Filing", "year": 2026}]),
                encoding="utf-8",
            )
            (data_dir / "processed" / "chunks_m.json").write_text(
                json.dumps([{"id": "C1", "work_id": "U1", "text": "Text"}]),
                encoding="utf-8",
            )
            raw, works, chunks, provenance = merge_modes(data_dir, ["m"])
        self.assertEqual(len(raw), 1)
        self.assertEqual(works[0]["corpus_source_modes"], ["m"])
        self.assertEqual(provenance["U1"]["raw_source_kind"], "unstructured_jsonl")

    def test_resumed_topic_run_marks_existing_report(self):
        with tempfile.TemporaryDirectory() as directory:
            reports_dir = Path(directory)
            (reports_dir / "phase1_ingest_m.json").write_text(
                json.dumps({"mode": "m", "work_count": 10}),
                encoding="utf-8",
            )
            report = resumed_topic_run(reports_dir, "m", "topic.toml")
        self.assertTrue(report["resumed"])
        self.assertEqual(report["work_count"], 10)

    def test_cli_summary_omits_full_provenance(self):
        summary = cli_summary(
            {
                "corpus": {
                    "corpus_id": "c",
                    "work_count": 2,
                    "chunk_count": 3,
                    "provenance": {"w1": {}, "w2": {}},
                },
                "validation": {"status": "corpus_only"},
            }
        )
        self.assertNotIn("provenance", summary["corpus"])
        self.assertEqual(summary["corpus"]["provenance_work_count"], 2)


if __name__ == "__main__":
    unittest.main()
