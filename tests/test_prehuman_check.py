import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from canon.product.prehuman_check import run_prehuman_check


class PrehumanCheckTests(unittest.TestCase):
    def test_prehuman_check_passes_automated_gates_but_blocks_human_claims(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reports_dir = root / "reports"
            reports_dir.mkdir()
            settings = SimpleNamespace(root=root, reports_dir=reports_dir)
            with patch("canon.product.prehuman_check.load_settings", return_value=settings):
                with patch("canon.product.prehuman_check.build_qrels_review_packet") as packet:
                    with patch("canon.product.prehuman_check.judge_qrels_csv") as judge:
                        with patch("canon.product.prehuman_check.qrels_from_review_csv") as qrels:
                            with patch("canon.product.prehuman_check.evaluate_semantic_models") as semantic:
                                with patch("canon.product.prehuman_check.evaluate_rerankers") as rerank:
                                    with patch("canon.product.prehuman_check.evaluate_source_diversity") as diversity:
                                        with patch("canon.product.prehuman_check.smoke.build_smoke_report") as smoke:
                                            with patch("canon.product.prehuman_check.readiness.build_readiness_report") as readiness:
                                                packet.return_value = {
                                                    "status": "ready_for_human_relevance_review",
                                                    "query_count": 2,
                                                    "candidate_count": 10,
                                                }
                                                judge.return_value = {
                                                    "status": "judged",
                                                    "row_count": 10,
                                                    "judged_count": 10,
                                                    "fill_relevance": True,
                                                }
                                                qrels.return_value = {
                                                    "status": "qrels_written",
                                                    "query_count": 2,
                                                    "judgment_count": 6,
                                                    "qrels": {"benchmark_kind": "llm_judged_qrels"},
                                                }
                                                semantic.return_value = {
                                                    "providers": [{"provider": "local", "status": "ok"}],
                                                    "leaderboard": [{"provider": "local", "primary_score": 0.5}],
                                                    "recommendation": {"human_review_required": True},
                                                }
                                                rerank.return_value = {
                                                    "rerankers": [{"provider": "heuristic", "status": "ok"}],
                                                    "leaderboard": [{"provider": "heuristic", "primary_score": 0.4}],
                                                }
                                                diversity.return_value = {"status": "pass", "summary": {"query_pass_rate": 1.0}}
                                                smoke.return_value = {"status": "pass", "checks": []}
                                                readiness.return_value = {"status": "pass", "checks": []}

                                                report = run_prehuman_check(
                                                    mode="unit",
                                                    benchmark_id="llm_judged_unit",
                                                    model_providers=["local"],
                                                    rerankers=["heuristic"],
                                                    write_report=False,
                                                )

        self.assertEqual(report["status"], "automated_pass_human_review_required")
        self.assertEqual(report["qrels_path"], "gold/llm_judged_unit.json")
        self.assertIn("human-reviewed qrels", report["human_review_boundary"]["next_test"])
        human_limit = next(item for item in report["components"] if item["id"] == "human_review_limit")
        self.assertFalse(human_limit["passed"])


if __name__ == "__main__":
    unittest.main()
