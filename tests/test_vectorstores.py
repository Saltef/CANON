import unittest
from types import SimpleNamespace

from canon.vectorstores import MemoryVectorStore, VectorPoint, point_id_for_chunk, vector_collection_name
from canon.vectorstores.qdrant import QdrantVectorStore


class VectorStoreTests(unittest.TestCase):
    def test_memory_vector_store_upserts_searches_and_deletes(self):
        store = MemoryVectorStore()
        store.ensure_collection("unit", vector_size=2)
        store.upsert(
            "unit",
            [
                VectorPoint("p1", [1.0, 0.0], {"mode": "m", "chunk_id": "c1"}),
                VectorPoint("p2", [0.0, 1.0], {"mode": "m", "chunk_id": "c2"}),
                VectorPoint("p3", [1.0, 0.0], {"mode": "other", "chunk_id": "c3"}),
            ],
        )

        results = store.search("unit", [1.0, 0.0], top_k=3, filters={"mode": "m"})

        self.assertEqual([row.payload["chunk_id"] for row in results], ["c1", "c2"])
        store.delete_points("unit", ["p1"])
        results = store.search("unit", [1.0, 0.0], top_k=3, filters={"mode": "m"})
        self.assertEqual([row.payload["chunk_id"] for row in results], ["c2"])

    def test_collection_and_point_ids_are_stable(self):
        collection = vector_collection_name("My Corpus", "openrouter", "qwen/qwen3-embedding-8b")

        self.assertEqual(collection, "canon_my-corpus_openrouter_qwen-qwen3-embedding-8b")
        self.assertEqual(point_id_for_chunk(collection, "chunk:1"), point_id_for_chunk(collection, "chunk:1"))
        self.assertNotEqual(point_id_for_chunk(collection, "chunk:1"), point_id_for_chunk(collection, "chunk:2"))

    def test_qdrant_search_indexes_filter_payload_keys(self):
        client = FakeQdrantClient()
        store = QdrantVectorStore(client=client)

        results = store.search("unit", [1.0, 0.0], top_k=1, filters={"mode": "m"})

        self.assertEqual(client.indexed_fields, [("unit", "mode")])
        self.assertEqual(results[0].payload["chunk_id"], "c1")


class FakeQdrantClient:
    def __init__(self):
        self.indexed_fields = []

    def create_payload_index(self, collection_name, field_name, field_schema):
        self.indexed_fields.append((collection_name, field_name))

    def query_points(self, **_kwargs):
        return SimpleNamespace(
            points=[
                SimpleNamespace(
                    id="point-1",
                    score=0.9,
                    payload={"mode": "m", "chunk_id": "c1"},
                )
            ]
        )


if __name__ == "__main__":
    unittest.main()
