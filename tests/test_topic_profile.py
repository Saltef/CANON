import tempfile
import unittest
from pathlib import Path

from canon.retrieval.corpus import RetrievalDocument
from canon.retrieval.topic_profile import (
    build_topic_profile,
    topic_query_variants,
    ensure_topic_profile,
    topic_profile_path,
)


class TopicProfileTests(unittest.TestCase):
    def test_topic_profile_builds_terms_and_related_variants(self):
        profile = build_topic_profile(
            "unit",
            [
                doc("c1", "fava beans contain vicine convicine and can trigger favism"),
                doc("c2", "fava beans are also known as vicia faba"),
            ],
        )

        self.assertEqual(profile["report_id"], "corpus_topic_profile_v1")
        self.assertTrue(profile["top_terms"])
        variants = topic_query_variants("fava beans", profile)
        self.assertTrue(any("vicine" in variant or "vicia" in variant for variant in variants))

    def test_ensure_topic_profile_writes_reusable_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory)
            profile = ensure_topic_profile(data_dir, "unit", [doc("c1", "energy grid water cooling")])

            self.assertTrue(topic_profile_path(data_dir, "unit").exists())
            self.assertEqual(profile["mode"], "unit")


def doc(chunk_id: str, text: str) -> RetrievalDocument:
    return RetrievalDocument(
        chunk_id=chunk_id,
        work_id=f"work:{chunk_id}",
        title=f"Title {chunk_id}",
        year=2024,
        source_name="Fixture",
        section="abstract",
        text=text,
        chunk_importance={},
        work_signals={"domain": "unit"},
    )


if __name__ == "__main__":
    unittest.main()
