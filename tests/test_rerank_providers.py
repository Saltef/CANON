import unittest
from unittest.mock import patch

from canon.rerank.providers import get_rerank_provider


class RerankProviderTests(unittest.TestCase):
    def test_heuristic_reranker_prefers_overlap(self):
        provider = get_rerank_provider("heuristic")
        rows = provider.rerank(
            "grid risks around AI data center expansion",
            [
                "civil war mediation and sanctions",
                "AI data center grid interconnection pressure",
            ],
            top_n=2,
        )

        self.assertEqual(rows[0].index, 1)
        self.assertGreater(rows[0].score, rows[1].score)

    def test_cohere_reranker_requires_key(self):
        with patch.dict("os.environ", {}, clear=True):
            with patch("canon.rerank.providers.load_local_env"):
                with self.assertRaises(RuntimeError):
                    get_rerank_provider("cohere")

    def test_openrouter_reranker_requires_key(self):
        with patch.dict("os.environ", {}, clear=True):
            with patch("canon.rerank.providers.load_local_env"):
                with self.assertRaises(RuntimeError):
                    get_rerank_provider("openrouter")

    def test_unknown_reranker_fails(self):
        with self.assertRaises(ValueError):
            get_rerank_provider("missing")


if __name__ == "__main__":
    unittest.main()
