import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from canon.eval.qrels_review import (
    build_qrels_review_packet,
    qrels_from_review_csv,
)


class QrelsReviewTests(unittest.TestCase):
    @patch("canon.eval.qrels_review.write_json")
    @patch("canon.eval.qrels_review.export_review_csv")
    @patch("canon.eval.qrels_review.Path.write_text")
    @patch("canon.eval.qrels_review.load_gold")
    @patch("canon.eval.qrels_review.run_retrieval")
    def test_build_qrels_review_packet_pools_candidates_across_policies(
        self,
        run_retrieval,
        load_gold,
        write_text,
        export_review_csv,
        write_json,
    ):
        load_gold.return_value = [{"id": "q1", "query": "battery risk", "type": "synthesis"}]
        run_retrieval.side_effect = [
            retrieval("lexical", [result("chunk:a", 1, 0.9), result("chunk:b", 2, 0.7)]),
            retrieval("semantic", [result("chunk:b", 1, 0.8), result("chunk:c", 2, 0.6)]),
        ]

        report = build_qrels_review_packet(
            mode="unit",
            queries_path=None,
            policies=["lexical", "semantic"],
            top_k=2,
            write_report=True,
        )

        first_record = report["records"][0]
        self.assertEqual(report["status"], "ready_for_human_relevance_review")
        self.assertEqual(first_record["candidate_count"], 3)
        pooled = {candidate["chunk_id"]: candidate for candidate in first_record["candidates"]}
        self.assertEqual(pooled["chunk:b"]["retrieved_by_policies"], ["lexical", "semantic"])
        self.assertEqual(pooled["chunk:b"]["best_rank"], 1)
        self.assertEqual(pooled["chunk:b"]["policy_ranks"]["semantic"], 1)
        write_json.assert_called_once()
        export_review_csv.assert_called_once()
        write_text.assert_called_once()

    def test_qrels_from_review_csv_writes_positive_relevance_only(self):
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "review.csv"
            output_path = Path(directory) / "qrels.json"
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["query_id", "query", "chunk_id", "relevance"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "query_id": "q1",
                        "query": "battery risk",
                        "chunk_id": "chunk:a",
                        "relevance": "3",
                    }
                )
                writer.writerow(
                    {
                        "query_id": "q1",
                        "query": "battery risk",
                        "chunk_id": "chunk:b",
                        "relevance": "0",
                    }
                )

            result = qrels_from_review_csv(
                csv_path=csv_path,
                benchmark_id="unit_qrels",
                output_path=output_path,
            )

        self.assertEqual(result["status"], "qrels_written")
        self.assertEqual(result["judgment_count"], 1)
        self.assertEqual(result["qrels"]["queries"][0]["relevant"], {"chunk:a": 3.0})

    def test_qrels_from_review_csv_rejects_invalid_relevance(self):
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "review.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["query_id", "query", "chunk_id", "relevance"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "query_id": "q1",
                        "query": "battery risk",
                        "chunk_id": "chunk:a",
                        "relevance": "5",
                    }
                )

            result = qrels_from_review_csv(csv_path=csv_path, benchmark_id="unit_qrels")

        self.assertEqual(result["status"], "failed_validation")
        self.assertTrue(result["validation_errors"])

    def test_qrels_from_review_csv_can_drop_empty_provisional_queries(self):
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "review.csv"
            output_path = Path(directory) / "qrels.json"
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["query_id", "query", "chunk_id", "relevance"],
                )
                writer.writeheader()
                writer.writerow({"query_id": "q1", "query": "battery risk", "chunk_id": "chunk:a", "relevance": "0"})
                writer.writerow({"query_id": "q2", "query": "grid risk", "chunk_id": "chunk:b", "relevance": "2"})

            strict = qrels_from_review_csv(csv_path=csv_path, benchmark_id="unit_qrels")
            relaxed = qrels_from_review_csv(
                csv_path=csv_path,
                benchmark_id="unit_qrels",
                output_path=output_path,
                benchmark_kind="llm_judged_qrels",
                allow_missing_relevance=True,
            )

        self.assertEqual(strict["status"], "failed_validation")
        self.assertEqual(relaxed["status"], "qrels_written")
        self.assertEqual(relaxed["dropped_query_count"], 1)
        self.assertEqual(relaxed["qrels"]["queries"][0]["id"], "q2")


def retrieval(policy, results):
    return {
        "policy": policy,
        "top_k": len(results),
        "results": results,
    }


def result(chunk_id, rank, score):
    return {
        "chunk_id": chunk_id,
        "work_id": f"work:{chunk_id}",
        "title": f"Title {chunk_id}",
        "source_name": "Journal",
        "year": 2024,
        "cluster_id": rank,
        "rank": rank,
        "final_score": score,
        "preview": "candidate preview",
    }


if __name__ == "__main__":
    unittest.main()
