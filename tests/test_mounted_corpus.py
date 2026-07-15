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
            (root / "scan.pdf").write_text("parsed when it contains extractable PDF text", encoding="utf-8")
            (root / "scan.png").write_text("image placeholder", encoding="utf-8")
            (root / "model.py").write_text("print('risk model')", encoding="utf-8")
            (root / "Dockerfile").write_text("FROM python:3.12-slim", encoding="utf-8")
            (root / "slides.pptx").write_text("presentation placeholder", encoding="utf-8")
            (root / "legacy.ppt").write_text("legacy presentation placeholder", encoding="utf-8")
            (root / ".git").mkdir()
            (root / ".git" / "config").write_text("ignored git metadata", encoding="utf-8")
            (root / "node_modules").mkdir()
            (root / "node_modules" / "dependency.js").write_text("ignored dependency", encoding="utf-8")
            (root / "archive.bin").write_text("unsupported binary placeholder", encoding="utf-8")

            summary = supported_file_summary(root)

        self.assertEqual(summary["supported_count"], 7)
        self.assertEqual(summary["unsupported_count"], 1)
        self.assertIn(".md", summary["supported_extensions"])
        self.assertIn(".pdf", summary["supported_extensions"])
        self.assertIn(".png", summary["supported_extensions"])
        self.assertIn(".py", summary["supported_extensions"])
        self.assertIn(".pptx", summary["supported_extensions"])
        self.assertIn(".ppt", summary["supported_extensions"])
        self.assertIn("<none>", summary["supported_extensions"])
        self.assertIn(".bin", summary["unsupported_extensions"])
        self.assertNotIn(str(root / ".git" / "config"), summary["sample_supported_files"])

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
