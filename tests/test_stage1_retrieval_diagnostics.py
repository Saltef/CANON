import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from canon.product.stage1_retrieval_diagnostics import run_stage1_retrieval_diagnostics


class Stage1RetrievalDiagnosticsTests(unittest.TestCase):
    def test_diagnostics_reports_query_failure_modes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_dir = root / "data"
            reports_dir = root / "reports"
            processed = data_dir / "processed"
            gold = root / "gold"
            processed.mkdir(parents=True)
            gold.mkdir()
            chunks = [
                {"id": "chunk:a", "work_id": "work:a"},
                {"id": "chunk:b", "work_id": "work:b"},
            ]
            (processed / "chunks_unit.json").write_text(json.dumps(chunks), encoding="utf-8")
            qrels = {
                "benchmark_id": "unit",
                "queries": [
                    {"id": "q1", "query": "alpha", "relevant": {"chunk:a": 1}},
                    {"id": "q2", "query": "beta", "relevant": {"chunk:b": 1}},
                ],
            }
            qrels_path = gold / "qrels.json"
            qrels_path.write_text(json.dumps(qrels), encoding="utf-8")
            rerank_path = reports_dir / "rerank.json"
            reports_dir.mkdir()
            rerank_path.write_text(
                json.dumps(
                    {
                        "rerankers": [
                            {
                                "status": "ok",
                                "provider_id": "heuristic",
                                "queries": [
                                    {
                                        "id": "q1",
                                        "query": "alpha",
                                        "candidate_count": 2,
                                        "candidate_recall": {"hit_ids": ["chunk:a"]},
                                        "ranked_chunk_ids": ["chunk:z"],
                                        "scored_candidates": [
                                            {"chunk_id": "chunk:a", "rerank_score": 0.9, "relevance": 1.0},
                                            {"chunk_id": "chunk:z", "rerank_score": 0.1, "relevance": 0.0},
                                        ],
                                    },
                                    {
                                        "id": "q2",
                                        "query": "beta",
                                        "candidate_count": 2,
                                        "candidate_recall": {"hit_ids": []},
                                        "ranked_chunk_ids": [],
                                        "scored_candidates": [
                                            {"chunk_id": "chunk:x", "rerank_score": 0.5, "relevance": 0.0},
                                        ],
                                    },
                                ],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            with patch.dict("os.environ", {"CANON_DATA_DIR": str(data_dir), "CANON_REPORTS_DIR": str(reports_dir)}):
                report = run_stage1_retrieval_diagnostics(
                    rerank_report=rerank_path,
                    qrels_path=qrels_path,
                    mode="unit",
                    write_report=False,
                )

        self.assertEqual(report["aggregate"]["query_count"], 2)
        self.assertEqual(report["failure_modes"]["candidate_pool_has_evidence_but_top10_misses"], 1)
        self.assertEqual(report["failure_modes"]["no_relevant_chunks_in_pool"], 1)
        self.assertEqual(report["oracle_reranking"]["status"], "ok")
        self.assertEqual(report["oracle_reranking"]["interpretation"], "strong_discrimination")
        self.assertIn("qrels_granularity", report["qrels_semantics"])
        self.assertEqual(report["parent_dominance"]["queries_with_parent_duplication_at_10"], 0)


if __name__ == "__main__":
    unittest.main()
