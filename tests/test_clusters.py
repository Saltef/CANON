import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
import json

from canon.retrieval.clusters import load_cluster_assignments


class ClusterTests(unittest.TestCase):
    def test_load_cluster_assignments_from_raw_openalex(self):
        with TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            raw_dir = data_dir / "raw"
            raw_dir.mkdir()
            (raw_dir / "openalex_demo.json").write_text(
                json.dumps(
                    [
                        {"id": "A", "referenced_works": ["B"]},
                        {"id": "B", "referenced_works": []},
                        {"id": "C", "referenced_works": []},
                    ]
                ),
                encoding="utf-8",
            )
            assignments = load_cluster_assignments(data_dir, "demo")
            self.assertEqual(assignments["A"], assignments["B"])
            self.assertNotEqual(assignments["A"], assignments["C"])


if __name__ == "__main__":
    unittest.main()
