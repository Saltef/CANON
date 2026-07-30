import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from canon.eval.rerank_evaluation import (
    RerankResult,
    diversify_by_parent,
    document_text,
    effective_qrels,
    evaluate_rerankers,
    get_rerank_provider,
)


class RerankEvaluationTests(unittest.TestCase):
    @patch("canon.eval.rerank_evaluation.run_retrieval")
    def test_evaluate_heuristic_reranker_reports_metrics(self, run_retrieval):
        run_retrieval.return_value = {
            "results": [
                result("chunk:a", "Economic Sanctions", "economic sanctions affect compliance"),
                result("chunk:b", "Civil War", "civil war mediation dataset"),
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            qrels_path = Path(directory) / "qrels.json"
            qrels_path.write_text(
                json.dumps(
                    {
                        "benchmark_id": "unit",
                        "queries": [
                            {
                                "id": "q1",
                                "query": "economic sanctions compliance",
                                "relevant": {"chunk:a": 3},
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            report = evaluate_rerankers(
                mode="unit",
                qrels_path=qrels_path,
                rerankers=["heuristic"],
                candidate_k=2,
                k=2,
            )

        self.assertEqual(report["report_id"], "rerank_evaluation_v1")
        self.assertEqual(report["rerankers"][0]["status"], "ok")
        self.assertEqual(report["leaderboard"][0]["provider"], "heuristic")
        self.assertGreater(report["rerankers"][0]["summary"]["nDCG@2"], 0.0)
        query = report["rerankers"][0]["queries"][0]
        self.assertEqual(query["candidate_recall"]["candidate_recall"], 1.0)
        self.assertEqual(query["score_observability"]["scored_candidate_count"], 2)
        self.assertGreater(query["score_observability"]["relevant_nonrelevant_score_gap"], 0.0)

    @patch.dict("os.environ", {}, clear=True)
    def test_cohere_reranker_is_unavailable_without_key(self):
        with patch("canon.eval.rerank_evaluation.load_local_env"):
            with tempfile.TemporaryDirectory() as directory:
                qrels_path = Path(directory) / "qrels.json"
                qrels_path.write_text(
                    json.dumps(
                        {
                            "benchmark_id": "unit",
                            "queries": [
                                {
                                    "id": "q1",
                                    "query": "economic sanctions compliance",
                                    "relevant": {"chunk:a": 3},
                                }
                            ],
                        }
                    ),
                    encoding="utf-8",
                )
                report = evaluate_rerankers(
                    mode="unit",
                    qrels_path=qrels_path,
                    rerankers=["cohere"],
                    candidate_k=2,
                    k=2,
                )

        self.assertEqual(report["rerankers"][0]["status"], "unavailable")
        self.assertIn("COHERE_API_KEY", report["rerankers"][0]["reason"])

    def test_openrouter_rerank_provider_uses_openrouter_key(self):
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}, clear=True):
            with patch("canon.eval.rerank_evaluation.load_local_env"):
                provider = get_rerank_provider("openrouter")
        self.assertEqual(provider.provider, "openrouter")
        self.assertEqual(provider.model, "cohere/rerank-v3.5")

    @patch("canon.eval.rerank_evaluation.build_pooled_candidate_context", return_value=None)
    @patch("canon.eval.rerank_evaluation.build_candidate_pool")
    def test_evaluate_reranker_can_use_pooled_candidates(
        self,
        build_candidate_pool,
        build_pooled_candidate_context,
    ):
        build_candidate_pool.return_value = {
            "source_counts": {"lexical": 1, "vector:local": 1},
            "candidates": [
                {
                    "chunk_id": "chunk:a",
                    "work_id": "work:a",
                    "title": "Economic Sanctions",
                    "source_name": "Journal",
                    "year": 2024,
                    "preview": "economic sanctions affect compliance",
                    "scores": {"lexical": 1.0, "vector:local": 0.8},
                    "ranks": {"lexical": 1, "vector:local": 1},
                    "retrieval_sources": ["lexical", "vector:local"],
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            qrels_path = Path(directory) / "qrels.json"
            qrels_path.write_text(
                json.dumps(
                    {
                        "benchmark_id": "unit",
                        "queries": [
                            {
                                "id": "q1",
                                "query": "economic sanctions compliance",
                                "relevant": {"chunk:a": 3},
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            report = evaluate_rerankers(
                mode="unit",
                qrels_path=qrels_path,
                rerankers=["heuristic"],
                candidate_k=2,
                k=1,
                pooled_candidates=True,
                query_variants=["sanctions pressure"],
                fusion="rrf",
                document_format="structured",
            )

        query = report["rerankers"][0]["queries"][0]
        self.assertEqual(report["candidate_generation"]["mode"], "pooled_lexical_vector")
        self.assertEqual(report["candidate_generation"]["query_variant_count"], 1)
        self.assertEqual(query["candidate_generation"]["mode"], "pooled_lexical_vector")
        self.assertEqual(query["candidate_recall"]["candidate_recall"], 1.0)
        self.assertEqual(build_candidate_pool.call_args.kwargs["query_variants"], ["sanctions pressure"])
        self.assertEqual(build_candidate_pool.call_args.kwargs["fusion"], "rrf")

    def test_structured_document_text_includes_retrieval_metadata(self):
        text = document_text(
            {
                "title": "Economic Sanctions",
                "source_name": "Journal",
                "year": 2024,
                "section": "abstract",
                "preview": "economic sanctions affect compliance",
                "retrieval_sources": ["lexical", "vector:local"],
                "candidate_scores": {"lexical": 2.0, "vector:local": 0.8},
                "candidate_ranks": {"lexical": 1, "vector:local": 3},
                "best_rank": 1,
            },
            "structured",
        )

        self.assertIn("section: abstract", text)
        self.assertIn("bm25_rank: 1", text)
        self.assertIn("dense_rank: 3", text)

    def test_diversify_by_parent_moves_duplicate_chunks_after_breadth(self):
        ranked = [
            ({"chunk_id": "chunk:a1", "work_id": "work:a"}, RerankResult(index=0, score=0.99)),
            ({"chunk_id": "chunk:a2", "work_id": "work:a"}, RerankResult(index=1, score=0.98)),
            ({"chunk_id": "chunk:b1", "work_id": "work:b"}, RerankResult(index=2, score=0.50)),
        ]

        diversified = diversify_by_parent(ranked, max_chunks_per_parent=1)

        self.assertEqual([row[0]["chunk_id"] for row in diversified], ["chunk:a1", "chunk:b1", "chunk:a2"])

    def test_effective_qrels_expands_document_labels_from_full_corpus_map(self):
        qrels = {"chunk:gold": 1.0}
        effective = effective_qrels(
            qrels=qrels,
            chunk_to_work={
                "chunk:gold": "work:a",
                "chunk:sibling": "work:a",
                "chunk:other": "work:b",
            },
            parent_qrels=True,
        )

        self.assertEqual(effective["chunk:sibling"], 1.0)
        self.assertNotIn("chunk:other", effective)

    def test_effective_qrels_is_not_candidate_dependent(self):
        qrels = {"chunk:gold": 1.0}
        effective = effective_qrels(
            qrels=qrels,
            chunk_to_work={"chunk:gold": "work:a"},
            parent_qrels=True,
        )

        self.assertNotIn("chunk:candidate_only", effective)

    @patch("canon.eval.rerank_evaluation.run_retrieval")
    def test_reranker_reuses_query_cache(self, run_retrieval):
        run_retrieval.return_value = {
            "results": [
                result("chunk:a", "Economic Sanctions", "economic sanctions affect compliance"),
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            qrels_path = root / "qrels.json"
            cache_dir = root / "cache"
            qrels_path.write_text(
                json.dumps(
                    {
                        "benchmark_id": "unit",
                        "queries": [
                            {
                                "id": "q1",
                                "query": "economic sanctions compliance",
                                "relevant": {"chunk:a": 3},
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            first = evaluate_rerankers(
                mode="unit",
                qrels_path=qrels_path,
                rerankers=["heuristic"],
                candidate_k=1,
                k=1,
                query_cache_dir=cache_dir,
            )
            second = evaluate_rerankers(
                mode="unit",
                qrels_path=qrels_path,
                rerankers=["heuristic"],
                candidate_k=1,
                k=1,
                query_cache_dir=cache_dir,
            )

        self.assertEqual(run_retrieval.call_count, 1)
        self.assertFalse(first["rerankers"][0]["queries"][0]["cache_hit"])
        self.assertTrue(second["rerankers"][0]["queries"][0]["cache_hit"])
        self.assertEqual(second["rerankers"][0]["cache"]["hit_count"], 1)

    @patch("canon.eval.rerank_evaluation.build_candidate_pool")
    def test_complete_query_cache_skips_pooled_context_build(self, build_candidate_pool):
        build_candidate_pool.return_value = {
            "source_counts": {"lexical": 1, "vector:local": 1},
            "candidates": [
                {
                    "chunk_id": "chunk:a",
                    "work_id": "work:a",
                    "title": "Economic Sanctions",
                    "source_name": "Journal",
                    "year": 2024,
                    "preview": "economic sanctions affect compliance",
                    "scores": {"lexical": 1.0, "vector:local": 0.8},
                    "ranks": {"lexical": 1, "vector:local": 1},
                    "retrieval_sources": ["lexical", "vector:local"],
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            qrels_path = root / "qrels.json"
            cache_dir = root / "cache"
            qrels_path.write_text(
                json.dumps(
                    {
                        "benchmark_id": "unit",
                        "queries": [
                            {
                                "id": "q1",
                                "query": "economic sanctions compliance",
                                "relevant": {"chunk:a": 3},
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with patch("canon.eval.rerank_evaluation.build_pooled_candidate_context", return_value=None):
                first = evaluate_rerankers(
                    mode="unit",
                    qrels_path=qrels_path,
                    rerankers=["heuristic"],
                    candidate_k=1,
                    k=1,
                    pooled_candidates=True,
                    query_cache_dir=cache_dir,
                )
            with patch(
                "canon.eval.rerank_evaluation.build_pooled_candidate_context",
                side_effect=AssertionError("pooled context should not be rebuilt"),
            ):
                second = evaluate_rerankers(
                    mode="unit",
                    qrels_path=qrels_path,
                    rerankers=["heuristic"],
                    candidate_k=1,
                    k=1,
                    pooled_candidates=True,
                    query_cache_dir=cache_dir,
                )

        self.assertFalse(first["rerankers"][0]["queries"][0]["cache_hit"])
        self.assertTrue(second["rerankers"][0]["queries"][0]["cache_hit"])
        self.assertEqual(build_candidate_pool.call_count, 1)


def result(chunk_id, title, preview):
    return {
        "chunk_id": chunk_id,
        "title": title,
        "source_name": "Journal",
        "year": 2024,
        "preview": preview,
    }


if __name__ == "__main__":
    unittest.main()
