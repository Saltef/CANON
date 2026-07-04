import unittest

from canon.eval.harness import average_components, average_rank_delta, rank_overlap


class EvalHarnessTests(unittest.TestCase):
    def test_rank_overlap_uses_jaccard(self):
        baseline = [{"chunk_id": "a"}, {"chunk_id": "b"}]
        candidate = [{"chunk_id": "b"}, {"chunk_id": "c"}]
        self.assertEqual(rank_overlap(baseline, candidate), 0.333333)

    def test_average_components(self):
        averages = average_components(
            [
                {"components": {"relevance": 1.0, "author_score": 0.5}},
                {"components": {"relevance": 0.0, "author_score": 1.0}},
            ]
        )
        self.assertEqual(averages["relevance"], 0.5)
        self.assertEqual(averages["author_score"], 0.75)

    def test_average_rank_delta(self):
        baseline = [{"chunk_id": "a", "rank": 1}, {"chunk_id": "b", "rank": 2}]
        candidate = [{"chunk_id": "b", "rank": 1}, {"chunk_id": "a", "rank": 2}]
        self.assertEqual(average_rank_delta(baseline, candidate), 1.0)


if __name__ == "__main__":
    unittest.main()
