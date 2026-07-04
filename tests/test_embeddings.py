import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from canon.embeddings.providers import HashedEmbeddingProvider, get_embedding_provider
from canon.embeddings.store import load_embedding_records, write_jsonl


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
            with self.assertRaises(RuntimeError):
                get_embedding_provider("openai")

    def test_jsonl_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "embeddings.jsonl"
            write_jsonl(path, [{"chunk_id": "c1"}, {"chunk_id": "c2"}])
            records = load_embedding_records(path)
        self.assertEqual([record["chunk_id"] for record in records], ["c1", "c2"])


if __name__ == "__main__":
    unittest.main()
