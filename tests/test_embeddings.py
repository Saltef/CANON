import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from canon.embeddings.providers import (
    CohereEmbeddingProvider,
    HashedEmbeddingProvider,
    OpenRouterEmbeddingProvider,
    embed_queries,
    get_embedding_provider,
    prepare_openrouter_query,
)
from canon.embeddings.store import artifact_key, embedding_store_path, load_embedding_records, write_jsonl


class EmbeddingTests(unittest.TestCase):
    def test_hashed_embedding_provider_returns_dense_vector(self):
        provider = HashedEmbeddingProvider(dimensions=32)
        result = provider.embed(["democratic peace"])[0]
        self.assertEqual(result.provider, "local")
        self.assertEqual(result.dimensions, 32)
        self.assertEqual(len(result.vector), 32)
        self.assertGreater(sum(abs(value) for value in result.vector), 0)

    def test_get_unknown_provider_fails(self):
        with self.assertRaises(ValueError):
            get_embedding_provider("missing")

    def test_openai_provider_requires_key(self):
        with patch.dict("os.environ", {}, clear=True):
            with patch("canon.embeddings.providers.load_local_env"):
                with self.assertRaises(RuntimeError):
                    get_embedding_provider("openai")

    def test_openrouter_provider_uses_openrouter_key(self):
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}, clear=True):
            with patch("canon.embeddings.providers.load_local_env"):
                provider = get_embedding_provider("openrouter")
        self.assertEqual(provider.provider, "openrouter")
        self.assertEqual(provider.model, "openai/text-embedding-3-small")

    def test_cohere_query_embeddings_use_search_query_input_type(self):
        with patch.dict("os.environ", {"COHERE_API_KEY": "test-key"}, clear=True):
            with patch("canon.embeddings.providers.load_local_env"):
                provider = CohereEmbeddingProvider()
            with patch.object(provider, "embed_with_input_type", return_value=[]) as embed:
                embed_queries(provider, ["what is relevant?"])
        embed.assert_called_once_with(["what is relevant?"], "search_query")

    def test_qwen_openrouter_queries_get_retrieval_instruction(self):
        prepared = prepare_openrouter_query("qwen/qwen3-embedding-8b", "What supports this claim?")
        self.assertIn("Instruct:", prepared)
        self.assertIn("Query: What supports this claim?", prepared)

    def test_openrouter_non_qwen_queries_stay_raw(self):
        self.assertEqual(
            prepare_openrouter_query("baai/bge-m3", "What supports this claim?"),
            "What supports this claim?",
        )

    def test_jsonl_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "embeddings.jsonl"
            write_jsonl(path, [{"chunk_id": "c1"}, {"chunk_id": "c2"}])
            records = load_embedding_records(path)
        self.assertEqual([record["chunk_id"] for record in records], ["c1", "c2"])

    def test_embedding_store_paths_include_model_for_openrouter_variants(self):
        path = embedding_store_path(Path("data"), "mode", "openrouter", "qwen/qwen3-embedding-8b")
        self.assertEqual(path.name, "embeddings_mode_openrouter_qwen-qwen3-embedding-8b.jsonl")
        self.assertEqual(artifact_key("cohere", "embed-v4.0"), "cohere_embed-v4-0")


if __name__ == "__main__":
    unittest.main()
