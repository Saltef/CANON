import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from canon.corpus.sync import (
    SourceManifestOptions,
    build_source_manifest,
    diff_manifests,
    load_previous_manifest,
    write_source_manifest,
)


class CorpusSyncTests(unittest.TestCase):
    def test_manifest_counts_supported_files_and_detects_changes(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            docs = root / "docs"
            docs.mkdir()
            (docs / "a.md").write_text("alpha evidence", encoding="utf-8")
            (docs / "b.bin").write_bytes(b"\x00\x01")

            first = build_source_manifest(SourceManifestOptions(mode="m", input_path=docs))
            (docs / "a.md").write_text("alpha evidence updated", encoding="utf-8")
            (docs / "c.txt").write_text("new evidence", encoding="utf-8")
            second = build_source_manifest(SourceManifestOptions(mode="m", input_path=docs))
            diff = diff_manifests(first, second)

        self.assertEqual(first["supported_file_count"], 1)
        self.assertEqual(first["unsupported_file_count"], 1)
        self.assertEqual(diff["status"], "changed")
        self.assertEqual(diff["changed"], ["a.md"])
        self.assertEqual(diff["added"], ["c.txt"])
        self.assertEqual(diff["removed"], [])

    def test_write_and_load_previous_manifest(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            docs = root / "docs"
            reports = root / "reports"
            docs.mkdir()
            (docs / "note.txt").write_text("source", encoding="utf-8")
            manifest = build_source_manifest(SourceManifestOptions(mode="my_mode", input_path=docs))
            path = write_source_manifest(manifest, reports_dir=reports)
            loaded = load_previous_manifest("my_mode", reports_dir=reports)

            self.assertTrue(path.exists())
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["content_digest"], manifest["content_digest"])
            self.assertEqual(loaded["content_digest"], manifest["content_digest"])


if __name__ == "__main__":
    unittest.main()
