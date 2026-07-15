import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from canon.product.mounted_corpus import run_mounted_corpus, supported_file_summary


class MountedCorpusTests(unittest.TestCase):
    def test_supported_file_summary_separates_supported_and_unsupported_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "note.md").write_text("AI infrastructure risk", encoding="utf-8")
            (root / "scan.pdf").write_text("not parsed by flexible ingest", encoding="utf-8")

            summary = supported_file_summary(root)

        self.assertEqual(summary["supported_count"], 1)
        self.assertEqual(summary["unsupported_count"], 1)
        self.assertIn(".md", summary["supported_extensions"])
        self.assertIn(".pdf", summary["unsupported_extensions"])

    def test_run_mounted_corpus_profiles_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "memo.txt").write_text("Data centers can affect energy demand.", encoding="utf-8")

            report = run_mounted_corpus(root, mode="unit_mount", profile_only=True)

        self.assertEqual(report["status"], "profiled")
        self.assertEqual(report["supported_files"]["supported_count"], 1)
        self.assertEqual(report["security_boundary"]["drive_access"], "filesystem_mount_only")

    def test_run_mounted_corpus_ingests_and_builds_corpus(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "memo.txt").write_text("Data centers can affect water and energy demand.", encoding="utf-8")
            with patch("canon.product.mounted_corpus.run_phase16") as build:
                build.return_value = {
                    "corpus": {
                        "corpus_id": "unit_mount_corpus",
                        "work_count": 1,
                        "chunk_count": 1,
                    },
                    "validation": {"status": "corpus_only"},
                }

                report = run_mounted_corpus(
                    root,
                    mode="unit_mount",
                    corpus_id="unit_mount_corpus",
                    chunk_tokens=20,
                    overlap_tokens=0,
                )

        build.assert_called_once()
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["corpus"]["corpus_id"], "unit_mount_corpus")
        self.assertGreaterEqual(report["ingest"]["chunk_count"], 1)


if __name__ == "__main__":
    unittest.main()
