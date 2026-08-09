import unittest

from canon.embeddings.index import index_documents
from canon.embeddings.providers import EmbeddingResult
from canon.retrieval.corpus import RetrievalDocument
from canon.vectorstores import MemoryVectorStore, VectorPoint


class VectorIndexTests(unittest.TestCase):
    def test_index_documents_upserts_and_deletes_stale_points(self):
        store = MemoryVectorStore()
        store.ensure_collection("unit", 2)
        store.upsert("unit", [VectorPoint("stale", [1.0, 0.0], {"mode": "m", "chunk_id": "old"})])
        provider = FakeEmbeddingProvider()

        report = index_documents(
            documents=[doc("c1", "grid risk"), doc("c2", "water risk")],
            mode="m",
            provider=provider,
            vector_store=store,
            vector_backend="memory",
            collection="unit",
            batch_size=2,
            previous_manifest={"point_ids": ["stale"]},
            delete_stale=True,
        )

        self.assertEqual(report["status"], "indexed")
        self.assertEqual(report["indexed_count"], 2)
        self.assertEqual(report["deleted_stale_count"], 1)
        self.assertEqual(report["manifest"]["point_count"], 2)
        results = store.search("unit", [1.0, 0.0], top_k=3, filters={"mode": "m"})
        self.assertEqual({row.payload["chunk_id"] for row in results}, {"c1", "c2"})
        self.assertNotIn("stale", store.collections["unit"]["points"])


class FakeEmbeddingProvider:
    provider = "fake"
    model = "fake-model"

    def embed(self, texts):
        return [
            EmbeddingResult(provider=self.provider, model=self.model, dimensions=2, vector=[1.0, float(index)])
            for index, _text in enumerate(texts)
        ]


def doc(chunk_id: str, text: str) -> RetrievalDocument:
    return RetrievalDocument(
        chunk_id=chunk_id,
        work_id=f"work:{chunk_id}",
        title=f"Title {chunk_id}",
        year=2026,
        source_name="Fixture",
        section="body",
        text=text,
        chunk_importance={},
        work_signals={},
    )


if __name__ == "__main__":
    unittest.main()
