import tempfile
import unittest
from pathlib import Path

from canon.claims.extract import (
    claim_confidence,
    cli_summary,
    extract_claims_from_document,
    split_sentences,
    stance_for_text,
    topic_tokens_for_text,
    write_json,
)
from canon.retrieval.corpus import RetrievalDocument


class ClaimsTests(unittest.TestCase):
    def test_stance_detection(self):
        self.assertEqual(stance_for_text("The evidence shows a significant effect."), "supportive")
        self.assertEqual(stance_for_text("Critics argue the effect is weak."), "critical")

    def test_claim_confidence_increases_with_markers(self):
        self.assertGreater(
            claim_confidence("We find evidence of a significant effect."),
            claim_confidence("Background context only."),
        )

    def test_topic_tokens(self):
        self.assertIn("democratic", topic_tokens_for_text("democratic peace democratic theory"))

    def test_split_sentences_handles_single_abstract(self):
        self.assertEqual(split_sentences("One sentence only"), ["One sentence only"])

    def test_extract_claims_from_document(self):
        document = RetrievalDocument(
            chunk_id="c1",
            work_id="w1",
            title="Demo",
            year=2020,
            source_name="Journal",
            section="abstract",
            text="We find evidence of a significant democratic peace effect.",
            chunk_importance={},
            work_signals={},
            cluster_id=1,
        )
        claims = extract_claims_from_document(document)
        self.assertEqual(len(claims), 1)
        self.assertEqual(claims[0].cluster_id, 1)
        self.assertEqual(claims[0].model_id, "claim-model-baseline-v1")
        self.assertIn("claim_marker_count", claims[0].features)

    def test_write_json_creates_parent_and_cleans_process_temp(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "claims.json"
            write_json(path, {"status": "ok"})
            self.assertTrue(path.exists())
            self.assertFalse(list(path.parent.glob("*.tmp")))

    def test_cli_summary_omits_full_claims(self):
        summary = cli_summary(
            {
                "mode": "demo",
                "model_id": "m",
                "claim_count": 1,
                "work_count": 1,
                "claims": [{"id": "claim:1", "text": "long"}],
            }
        )
        self.assertNotIn("claims", summary)
        self.assertEqual(summary["sample_claim_ids"], ["claim:1"])


if __name__ == "__main__":
    unittest.main()
