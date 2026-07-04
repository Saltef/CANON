import unittest
from pathlib import Path

from canon.corpus.packs import topic_manifest_row
from canon.ingest.pipeline import topic_scope


class CorpusPackTests(unittest.TestCase):
    def test_topic_scope_supports_original_scope_format(self):
        ids, terms = topic_scope(
            {"scope": {"openalex_concept_ids": ["C1"], "seed_terms": ["democratic peace"]}}
        )
        self.assertEqual(ids, ["C1"])
        self.assertEqual(terms, ["democratic peace"])

    def test_topic_scope_supports_pack_format(self):
        ids, terms = topic_scope(
            {
                "topic": {"query": "fallback query"},
                "openalex": {"search": "sanctions effectiveness"},
            }
        )
        self.assertEqual(ids, [])
        self.assertEqual(terms, ["sanctions effectiveness"])

    def test_topic_manifest_row_supports_original_scope_format(self):
        row = topic_manifest_row(
            {"scope": {"label": "IR", "seed_terms": ["a", "b"]}},
            Path("conf/topics/demo.toml"),
            "conf/topics/demo.toml",
            10,
        )
        self.assertEqual(row["topic_id"], "demo")
        self.assertEqual(row["name"], "IR")
        self.assertIn("--topic-file conf/topics/demo.toml", row["recommended_command"])


if __name__ == "__main__":
    unittest.main()
