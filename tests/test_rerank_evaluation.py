import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from canon.eval.rerank_evaluation import evaluate_rerankers


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

    @patch.dict("os.environ", {}, clear=True)
    def test_cohere_reranker_is_unavailable_without_key(self):
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
