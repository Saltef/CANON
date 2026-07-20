import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from canon.product.stage1_gate import run_stage1_gate


class Stage1GateTests(unittest.TestCase):
    def test_gate_blocks_when_scale_is_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            report_path = Path(directory) / "optimizer.json"
            report_path.write_text(
                json.dumps(
                    {
                        "status": "automated_pass_human_review_required",
                        "trials": [
                            {
                                "benchmark_id": "unit",
                                "query_count": 5,
                                "status": "ok",
                            }
                        ],
                        "best_trial": {
                            "candidate_recall_mean": 1.0,
                            "Recall@k": 1.0,
                            "nDCG@k": 1.0,
                        },
                    }
                ),
                encoding="utf-8",
            )
            with patch.dict("os.environ", {"CANON_REPORTS_DIR": directory}):
                gate = run_stage1_gate(optimizer_report=report_path, write_report=False)

        self.assertEqual(gate["status"], "blocked")
        self.assertIn("query_scale", {failure["id"] for failure in gate["failures"]})
        self.assertIn("corpus_scale", {failure["id"] for failure in gate["failures"]})

    def test_gate_passes_when_scale_and_metrics_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            report_path = Path(directory) / "optimizer.json"
            report_path.write_text(
                json.dumps(
                    {
                        "status": "automated_pass_human_review_required",
                        "trials": [
                            {
                                "benchmark_id": "scifact",
                                "query_count": 30,
                                "status": "ok",
                            },
                            {
                                "benchmark_id": "nfcorpus",
                                "query_count": 30,
                                "status": "ok",
                            },
                        ],
                        "best_trial": {
                            "candidate_recall_mean": 1.0,
                            "Recall@k": 0.9,
                            "nDCG@k": 0.85,
                        },
                    }
                ),
                encoding="utf-8",
            )
            with patch.dict("os.environ", {"CANON_REPORTS_DIR": directory}):
                gate = run_stage1_gate(optimizer_report=report_path, write_report=False)

        self.assertEqual(gate["status"], "pass")


if __name__ == "__main__":
    unittest.main()
