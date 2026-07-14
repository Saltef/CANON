import json
import tempfile
import unittest
from pathlib import Path

from canon.config import load_settings
from canon.eval.model_evaluation import (
    dense_cosine,
    evaluate_semantic_models,
    query_slices,
)
from canon.ingest.unstructured import ingest_unstructured_jsonl


class SemanticModelEvaluationTests(unittest.TestCase):
    def test_dense_cosine_handles_normal_and_bad_vectors(self):
        self.assertEqual(round(dense_cosine([1.0, 0.0], [1.0, 0.0]), 6), 1.0)
        self.assertEqual(dense_cosine([1.0], [1.0, 0.0]), 0.0)
        self.assertEqual(dense_cosine([], []), 0.0)

    def test_query_slices_detect_relational_and_domain_terms(self):
        slices = query_slices("How does interconnection backlog affect project delivery risk?")
        self.assertIn("relational_semantics", slices)
        self.assertIn("domain_terminology", slices)

    def test_evaluate_semantic_models_reports_metrics_and_unavailable_provider(self):
        settings = load_settings()
        mode = "unit_model_eval"
        ingest_unstructured_jsonl(
            settings.fixtures_dir / "unstructured_sample.jsonl",
            mode=mode,
            chunk_tokens=24,
            overlap_tokens=4,
        )
        chunks = json.loads((settings.data_dir / "processed" / f"chunks_{mode}.json").read_text(encoding="utf-8"))
        target_chunk = next(chunk for chunk in chunks if "battery storage market risk" in chunk["text"].lower())
        with tempfile.TemporaryDirectory() as directory:
            qrels_path = Path(directory) / "qrels.json"
            qrels_path.write_text(
                json.dumps(
                    {
                        "benchmark_id": "unit_model_eval",
                        "queries": [
                            {
                                "id": "q1",
                                "query": "battery storage market risk",
                                "relevant": {target_chunk["id"]: 3},
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            report = evaluate_semantic_models(
                mode=mode,
                qrels_path=qrels_path,
                providers=["local", "openai"],
                k=5,
            )

        self.assertEqual(report["report_id"], "semantic_model_evaluation_v1")
        self.assertEqual(report["providers"][0]["status"], "ok")
        self.assertEqual(report["providers"][0]["summary"]["labeled_query_count"], 1)
        self.assertIn("nDCG@5", report["providers"][0]["summary"])
        self.assertEqual(report["providers"][1]["status"], "unavailable")
        self.assertEqual(report["leaderboard"][0]["provider"], "local")
        self.assertTrue(report["recommendation"]["human_review_required"])


if __name__ == "__main__":
    unittest.main()
