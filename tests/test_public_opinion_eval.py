import unittest

from canon.config import load_settings
from canon.corpus.build import build_named_corpus
from canon.eval.public_opinion import analyze_public_opinion, stance_label
from canon.ingest.social_media import ingest_social_media


class PublicOpinionAnalysisTests(unittest.TestCase):
    def test_stance_label_detects_conditional_support(self):
        self.assertEqual(
            stance_label("I support the project if the company publishes safety commitments."),
            "mixed_or_conditional",
        )
        self.assertEqual(stance_label("Residents worry about fire risk."), "concern_or_opposition")

    def test_public_opinion_analysis_tracks_safety_allowed_themes(self):
        settings = load_settings()
        ingest_social_media(
            settings.fixtures_dir / "social_media_public_opinion_sample.jsonl",
            mode="unit_public_opinion_analysis",
            input_format="jsonl",
            chunk_tokens=32,
            overlap_tokens=4,
        )
        build_named_corpus(
            corpus_id="unit_public_opinion_analysis_corpus",
            from_modes=["unit_public_opinion_analysis"],
            harvest=False,
        )

        report = analyze_public_opinion("unit_public_opinion_analysis_corpus")

        self.assertEqual(report["work_count"], 4)
        self.assertEqual(report["platform_counts"], {"Forum": 1, "MicroBlog": 3})
        self.assertEqual(report["safety_decision_counts"]["block"], 1)
        self.assertEqual(report["aggregation_readiness"]["status"], "fixture_or_insufficient_for_inference")
        self.assertEqual(report["collection_audit"]["status"], "fixture_only_or_needs_collection_metadata")
        self.assertIn("terms_review_share", report["collection_audit"]["metadata_coverage"])
        self.assertIn("safety_risk", report["theme_counts_safety_allowed"])
        self.assertIn("mixed_or_conditional", report["stance_counts_safety_allowed"])
        self.assertTrue(any("lexical probes" in item for item in report["limitations"]))
        self.assertTrue(any("Collection metadata" in item for item in report["limitations"]))


if __name__ == "__main__":
    unittest.main()
