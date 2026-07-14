import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from canon.eval.llm_judge import (
    judge_answer_review_packet,
    judge_qrels_csv,
    parse_json_object,
)


class LlmJudgeTests(unittest.TestCase):
    def test_judge_qrels_csv_adds_provisional_labels_without_overwriting(self):
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "qrels_review.csv"
            output_path = Path(directory) / "judged.csv"
            write_qrels_csv(
                input_path,
                [
                    {
                        "query_id": "q1",
                        "query": "economic sanctions effects",
                        "chunk_id": "chunk:a",
                        "title": "Economic Sanctions and Policy Effects",
                        "preview": "Economic sanctions can affect compliance and distributional outcomes.",
                        "best_rank": "1",
                        "relevance": "",
                    },
                    {
                        "query_id": "q1",
                        "query": "economic sanctions effects",
                        "chunk_id": "chunk:b",
                        "title": "Civil War Mediation",
                        "preview": "Mediation can affect civil war duration.",
                        "best_rank": "2",
                        "relevance": "0",
                    },
                ],
            )

            report = judge_qrels_csv(input_path, output_path, provider_name="heuristic")
            rows = read_csv(output_path)

        self.assertEqual(report["status"], "judged")
        self.assertEqual(rows[0]["judge_relevance"], "3")
        self.assertEqual(rows[0]["relevance"], "")
        self.assertEqual(rows[1]["relevance"], "0")
        self.assertIn(rows[1]["human_audit_priority"], {"low", "high", "medium"})

    def test_judge_qrels_csv_can_fill_relevance_when_requested(self):
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "qrels_review.csv"
            output_path = Path(directory) / "judged.csv"
            write_qrels_csv(
                input_path,
                [
                    {
                        "query_id": "q1",
                        "query": "economic sanctions effects",
                        "chunk_id": "chunk:a",
                        "title": "Economic Sanctions and Policy Effects",
                        "preview": "Economic sanctions can affect compliance.",
                        "best_rank": "1",
                        "relevance": "",
                    }
                ],
            )

            judge_qrels_csv(input_path, output_path, provider_name="heuristic", fill_relevance=True)
            rows = read_csv(output_path)

        self.assertEqual(rows[0]["relevance"], "3")

    @patch("canon.eval.llm_judge.load_records")
    def test_judge_answer_review_packet_marks_high_risk_categories_for_audit(self, load_records):
        load_records.return_value = [
            {
                "id": "q28",
                "category": "overclaim_check",
                "query": "Overclaim check",
                "answer_response": {
                    "answer": "A cautious answer",
                    "citations": [{"citation_id": "C1"}],
                    "evidence": [{"citation_id": "C1"}, {"citation_id": "C2"}, {"citation_id": "C3"}],
                    "support_assessment": {"support_level": "mixed"},
                    "query_diagnostics": {"stability": {"status": "sensitive"}},
                },
            }
        ]
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "answer_judge.csv"
            report = judge_answer_review_packet(Path("records.json"), output_path)
            rows = read_csv(output_path)

        self.assertEqual(report["status"], "judged")
        self.assertEqual(rows[0]["human_audit_priority"], "high")
        self.assertEqual(rows[0]["judge_source"], "heuristic_judge")

    def test_parse_json_object_accepts_fenced_json(self):
        parsed = parse_json_object(
            """```json
{"judge_relevance": 2, "judge_confidence": 0.7, "judge_rationale": "Partial."}
```"""
        )

        self.assertEqual(parsed["judge_relevance"], 2)


def write_qrels_csv(path, rows):
    fieldnames = ["query_id", "query", "chunk_id", "title", "preview", "best_rank", "relevance"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_csv(path):
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


if __name__ == "__main__":
    unittest.main()
