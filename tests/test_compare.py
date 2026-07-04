import unittest

from canon.retrieval.compare import build_rank_table


class CompareTests(unittest.TestCase):
    def test_build_rank_table_merges_policy_ranks(self):
        table = build_rank_table(
            [
                {
                    "policy": "lexical",
                    "results": [
                        {
                            "chunk_id": "c1",
                            "work_id": "w1",
                            "title": "A",
                            "year": 2020,
                            "cluster_id": 1,
                            "rank": 1,
                            "final_score": 0.9,
                        }
                    ],
                },
                {
                    "policy": "balanced",
                    "results": [
                        {
                            "chunk_id": "c1",
                            "work_id": "w1",
                            "title": "A",
                            "year": 2020,
                            "cluster_id": 1,
                            "rank": 2,
                            "final_score": 0.7,
                        }
                    ],
                },
            ]
        )
        self.assertEqual(table[0]["ranks"], {"lexical": 1, "balanced": 2})
        self.assertEqual(table[0]["cluster_id"], 1)


if __name__ == "__main__":
    unittest.main()
