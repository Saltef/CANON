import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from canon.product.human_review_scaffold import build_human_review_scaffold


class HumanReviewScaffoldTests(unittest.TestCase):
    def test_scaffold_builds_retrieval_and_synthesis_review_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reports_dir = root / "reports"
            suite_dir = root / "conf" / "benchmark_suites"
            gold_dir = root / "gold"
            reports_dir.mkdir()
            suite_dir.mkdir(parents=True)
            gold_dir.mkdir()
            (gold_dir / "qrels.json").write_text(
                json.dumps(
                    {
                        "benchmark_id": "unit_qrels",
                        "benchmark_kind": "public_qrels",
                        "queries": [
                            {
                                "id": "q1",
                                "query": "alpha",
                                "relevant": {"chunk:1": 2, "chunk:2": 1},
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            suite_path = suite_dir / "suite.json"
            suite_path.write_text(
                json.dumps(
                    {
                        "suite_id": "unit_suite",
                        "benchmarks": [{"id": "unit", "mode": "unit_mode", "qrels": "gold/qrels.json"}],
                    }
                ),
                encoding="utf-8",
            )
            disagreement_path = gold_dir / "disagreement.json"
            disagreement_path.write_text(
                json.dumps(
                    {
                        "benchmark_id": "unit_disagreement",
                        "cases": [
                            {
                                "id": "c1",
                                "question": "Where is evidence contested?",
                                "evidence": [
                                    {"evidence_id": "C1", "expected_stance": "supports", "text": "Supports."},
                                    {"evidence_id": "C2", "expected_stance": "contradicts", "text": "Does not support."},
                                ],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            settings = SimpleNamespace(root=root, reports_dir=reports_dir)
            with patch("canon.product.human_review_scaffold.load_settings", return_value=settings):
                report = build_human_review_scaffold(
                    suite_path=suite_path,
                    disagreement_benchmark_path=disagreement_path,
                    retrieval_target=2,
                    synthesis_target=2,
                    write_report=False,
                )

        self.assertEqual(report["status"], "review_scaffold_ready_human_labels_required")
        self.assertEqual(report["retrieval_review"]["row_count"], 2)
        self.assertEqual(report["synthesis_review"]["row_count"], 2)
        self.assertFalse(report["completion_status"]["retrieval_target_met"])
        self.assertFalse(report["completion_status"]["synthesis_target_met"])
        self.assertIn("reviewer_id", report["retrieval_review"]["rows"][0])
        self.assertIn("reviewer_id", report["synthesis_review"]["rows"][0])
        self.assertIn("reviewer_provenance", report["label_schemas"])
        self.assertIn("retrieval_qrels", report["label_schemas"])
        self.assertIn("synthesis_disagreement", report["label_schemas"])


if __name__ == "__main__":
    unittest.main()
