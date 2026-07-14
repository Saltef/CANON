import unittest
import json
import tempfile
from pathlib import Path

from canon.graph.build import build_edges, connected_components, load_raw_works, matched_reference_fraction, raw_mode_path


class GraphTests(unittest.TestCase):
    def test_build_edges_keeps_only_in_corpus_references(self):
        raw = [
            {"id": "A", "referenced_works": ["B", "X"]},
            {"id": "B", "referenced_works": []},
        ]
        self.assertEqual(build_edges(raw), [("A", "B")])

    def test_matched_reference_fraction(self):
        raw = [
            {"id": "A", "referenced_works": ["B", "X"]},
            {"id": "B", "referenced_works": []},
        ]
        self.assertEqual(matched_reference_fraction(raw), 0.5)

    def test_connected_components(self):
        components = connected_components(["A", "B", "C"], [("A", "B")])
        self.assertEqual(components, [["A", "B"], ["C"]])

    def test_load_raw_works_accepts_neutral_raw_corpus(self):
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory)
            (data_dir / "raw").mkdir()
            path = data_dir / "raw" / "raw_social_demo_corpus.json"
            path.write_text(json.dumps([{"id": "social:1", "source_name": "Forum"}]), encoding="utf-8")

            self.assertEqual(raw_mode_path(data_dir, "social_demo_corpus"), path)
            self.assertEqual(load_raw_works(data_dir, "social_demo_corpus")[0]["id"], "social:1")


if __name__ == "__main__":
    unittest.main()
