import tempfile
import unittest
from pathlib import Path

from canon.claims.conflict import (
    candidate_pair_indices,
    classify_conflict,
    cli_summary,
    detect_conflicts_from_claims,
    shared_topic_tokens,
    write_json,
)


class ConflictTests(unittest.TestCase):
    def test_shared_topic_tokens(self):
        left = {"topic_tokens": ["democratic", "peace"]}
        right = {"topic_tokens": ["peace", "war"]}
        self.assertEqual(shared_topic_tokens(left, right), ["peace"])

    def test_classify_cross_cluster_tension(self):
        left = {"stance": "supportive", "cluster_id": 1}
        right = {"stance": "critical", "cluster_id": 2}
        classified = classify_conflict(left, right, ["democratic", "peace"])
        self.assertIsNotNone(classified)
        self.assertEqual(classified[0], "cross_cluster_tension")

    def test_detect_conflicts_skips_same_work(self):
        claims = [
            {
                "id": "a",
                "work_id": "w1",
                "title": "A",
                "stance": "supportive",
                "cluster_id": 1,
                "topic_tokens": ["democratic", "peace"],
            },
            {
                "id": "b",
                "work_id": "w1",
                "title": "B",
                "stance": "critical",
                "cluster_id": 2,
                "topic_tokens": ["democratic", "peace"],
            },
        ]
        self.assertEqual(detect_conflicts_from_claims(claims), [])

    def test_detect_conflicts_finds_candidate(self):
        claims = [
            {
                "id": "a",
                "work_id": "w1",
                "title": "A",
                "stance": "supportive",
                "cluster_id": 1,
                "topic_tokens": ["democratic", "peace"],
            },
            {
                "id": "b",
                "work_id": "w2",
                "title": "B",
                "stance": "critical",
                "cluster_id": 2,
                "topic_tokens": ["democratic", "peace"],
            },
        ]
        self.assertEqual(len(detect_conflicts_from_claims(claims)), 1)

    def test_candidate_pair_indices_uses_shared_topic_buckets(self):
        claims = [
            {"topic_tokens": ["democratic", "peace"]},
            {"topic_tokens": ["democratic", "peace"]},
            {"topic_tokens": ["sanctions"]},
        ]
        self.assertEqual(candidate_pair_indices(claims), [(0, 1)])

    def test_write_json_creates_parent_and_cleans_process_temp(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "conflicts.json"
            write_json(path, {"status": "ok"})
            self.assertTrue(path.exists())
            self.assertFalse(list(path.parent.glob("*.tmp")))

    def test_cli_summary_omits_full_conflicts(self):
        summary = cli_summary(
            {
                "mode": "demo",
                "claim_count": 2,
                "conflict_count": 1,
                "conflicts": [
                    {
                        "conflict_type": "cross_cluster_tension",
                        "shared_topic_tokens": ["democratic", "peace"],
                    }
                ],
            }
        )
        self.assertNotIn("conflicts", summary)
        self.assertEqual(summary["conflict_type_counts"]["cross_cluster_tension"], 1)


if __name__ == "__main__":
    unittest.main()
