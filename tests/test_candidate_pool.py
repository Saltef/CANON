from __future__ import annotations

import json
import unittest

from canon.embeddings.providers import HashedEmbeddingProvider
from canon.retrieval.candidates import CandidateHit, candidate_pool_from_documents, expand_query_variants, sort_hits
from canon.retrieval.corpus import RetrievalDocument


class CandidatePoolTests(unittest.TestCase):
    def test_candidate_pool_dedupes_lexical_and_vector_sources(self):
        documents = [
            doc("c1", "economic sanctions compliance pressure"),
            doc("c2", "civil war mediation dataset"),
        ]
        with self.subTest("pooled"):
            hits = candidate_pool_from_documents(
                query="economic sanctions compliance",
                documents=documents,
                embeddings_path=missing_path(),
                lexical_k=2,
                vector_k=0,
            )

        self.assertEqual(hits[0].document.chunk_id, "c1")
        self.assertEqual(hits[0].retrieval_sources, ("lexical",))
        self.assertIn("lexical", hits[0].scores)

    def test_candidate_pool_uses_embedding_records_for_vector_candidates(self):
        import tempfile
        from pathlib import Path

        documents = [
            doc("c1", "economic sanctions compliance pressure"),
            doc("c2", "civil war mediation dataset"),
        ]
        provider = HashedEmbeddingProvider()
        vectors = provider.embed([document.text for document in documents])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "embeddings.jsonl"
            write_jsonl(
                path,
                [
                    embedding("c1", vectors[0].vector),
                    embedding("c2", vectors[1].vector),
                ],
            )
            hits = candidate_pool_from_documents(
                query="economic sanctions compliance",
                documents=documents,
                embeddings_path=path,
                lexical_k=2,
                vector_k=2,
                provider="local",
            )

        by_id = {hit.document.chunk_id: hit for hit in hits}
        self.assertIn("lexical", by_id["c1"].retrieval_sources)
        self.assertIn("vector:local", by_id["c1"].retrieval_sources)

    def test_candidate_pool_supports_rrf_fusion(self):
        documents = [
            doc("c1", "alpha beta"),
            doc("c2", "gamma delta"),
        ]
        provider = HashedEmbeddingProvider()
        vectors = provider.embed([documents[1].text])
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "embeddings.jsonl"
            write_jsonl(path, [embedding("c2", vectors[0].vector)])
            hits = candidate_pool_from_documents(
                query="alpha beta",
                documents=documents,
                embeddings_path=path,
                lexical_k=1,
                vector_k=1,
                provider="local",
                fusion="rrf",
            )

        self.assertEqual({hit.document.chunk_id for hit in hits}, {"c1", "c2"})

    def test_auto_query_expansion_adds_alias_hits(self):
        documents = [
            doc("c1", "vicia faba favism vicine convicine"),
            doc("c2", "civil war mediation dataset"),
        ]
        hits = candidate_pool_from_documents(
            query="fava beans",
            documents=documents,
            embeddings_path=missing_path(),
            lexical_k=1,
            vector_k=0,
            auto_query_expansion=True,
            benchmark_oracle_expansion=True,
        )

        self.assertEqual(hits[0].document.chunk_id, "c1")
        self.assertIn("lexical:variant:1", hits[0].retrieval_sources)

    def test_auto_query_expansion_does_not_use_oracle_aliases_by_default(self):
        documents = [
            doc("c1", "vicia faba favism vicine convicine"),
            doc("c2", "civil war mediation dataset"),
        ]
        hits = candidate_pool_from_documents(
            query="fava beans",
            documents=documents,
            embeddings_path=missing_path(),
            lexical_k=1,
            vector_k=0,
            auto_query_expansion=True,
        )

        self.assertEqual(hits, [])

    def test_expand_query_variants_knows_biomedical_aliases(self):
        self.assertIn("hearing loss auditory impairment", expand_query_variants("deafness"))

    def test_weighted_fusion_normalizes_bm25_and_dense_scores(self):
        lexical_hit = CandidateHit(
            document=doc("c1", "alpha lexical"),
            retrieval_sources=("lexical",),
            best_rank=1,
            best_score=100.0,
            scores={"lexical": 100.0},
            ranks={"lexical": 1},
        )
        dense_hit = CandidateHit(
            document=doc("c2", "alpha dense"),
            retrieval_sources=("vector:local",),
            best_rank=2,
            best_score=0.9,
            scores={"vector:local": 0.9},
            ranks={"vector:local": 2},
        )

        ranked = sort_hits([lexical_hit, dense_hit], fusion="weighted_bm25_dense")

        self.assertEqual(ranked[0].document.chunk_id, "c2")

    def test_parent_expansion_adds_sibling_chunks(self):
        documents = [
            doc("c1", "alpha query evidence", work_id="work:a"),
            doc("c2", "sibling paragraph with different wording", work_id="work:a"),
            doc("c3", "unrelated beta", work_id="work:b"),
        ]
        hits = candidate_pool_from_documents(
            query="alpha query",
            documents=documents,
            embeddings_path=missing_path(),
            lexical_k=1,
            vector_k=0,
            parent_expansion_limit=1,
        )

        by_id = {hit.document.chunk_id: hit for hit in hits}
        self.assertIn("c2", by_id)
        self.assertIn("parent_neighborhood", by_id["c2"].retrieval_sources)


def embedding(chunk_id: str, vector: list[float]) -> dict:
    return {
        "chunk_id": chunk_id,
        "work_id": f"work:{chunk_id}",
        "provider": "test",
        "model": "test",
        "dimensions": len(vector),
        "vector": vector,
    }


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def missing_path():
    from pathlib import Path

    return Path("__missing_embeddings__.jsonl")


def doc(chunk_id: str, text: str, work_id: str | None = None) -> RetrievalDocument:
    return RetrievalDocument(
        chunk_id=chunk_id,
        work_id=work_id or f"work:{chunk_id}",
        title=f"Title {chunk_id}",
        year=2024,
        source_name="Fixture",
        section="abstract",
        text=text,
        chunk_importance={},
        work_signals={},
    )


if __name__ == "__main__":
    unittest.main()
