import unittest

from canon.retrieval.semantic import HashedSemanticEncoder, cosine, semantic_scores, token_features


class SemanticTests(unittest.TestCase):
    def test_token_features_include_bigrams(self):
        features = token_features("democratic peace theory")
        self.assertIn("tok:democratic", features)
        self.assertIn("bi:democratic_peace", features)

    def test_encoder_is_deterministic(self):
        encoder = HashedSemanticEncoder(dimensions=64)
        first = encoder.encode("democratic peace")
        second = encoder.encode("democratic peace")
        self.assertEqual(first.values, second.values)

    def test_cosine_scores_related_text_higher(self):
        scores = semantic_scores(
            "democratic peace",
            ["democratic peace research", "economic sanctions policy"],
            dimensions=128,
        )
        self.assertGreater(scores[0], scores[1])

    def test_empty_similarity(self):
        encoder = HashedSemanticEncoder(dimensions=64)
        self.assertEqual(cosine(encoder.encode(""), encoder.encode("text")), 0.0)


if __name__ == "__main__":
    unittest.main()
