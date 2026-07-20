import unittest

from canon.product.frame_enrichment import enrich_frame, merge_frames, sanitize_frame


class FrameEnrichmentTests(unittest.TestCase):
    def test_merge_frames_keeps_schema_and_merges_lists(self):
        merged = merge_frames(
            {"topic": "AI infrastructure", "subdomains": ["energy"], "regions": ["Brazil"]},
            {
                "subdomains": ["water", "energy"],
                "regions": ["Chile"],
                "unknown": "drop me",
            },
        )

        self.assertEqual(merged["subdomains"], ["energy", "water"])
        self.assertEqual(merged["regions"], ["Brazil", "Chile"])
        self.assertNotIn("unknown", merged)

    def test_sanitize_frame_coerces_lists_and_drops_unknown_fields(self):
        sanitized = sanitize_frame({"languages": "English, Spanish", "topic": "Risk", "bad": "x"})

        self.assertEqual(sanitized["languages"], ["English", "Spanish"])
        self.assertEqual(sanitized["topic"], "Risk")
        self.assertNotIn("bad", sanitized)

    def test_enrich_frame_deterministic_only_is_noop(self):
        frame = {"topic": "Risk", "subdomains": ["energy"]}

        result = enrich_frame("question", frame, provider=None)

        self.assertEqual(result["status"], "deterministic_only")
        self.assertEqual(result["research_frame"], frame)


if __name__ == "__main__":
    unittest.main()
