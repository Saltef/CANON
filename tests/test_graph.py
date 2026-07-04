import unittest

from canon.graph.build import build_edges, connected_components, matched_reference_fraction


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


if __name__ == "__main__":
    unittest.main()
