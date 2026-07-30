import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from canon.product.publishable_package import build_publishable_package, package_status


class PublishablePackageTests(unittest.TestCase):
    def test_package_blocks_pilot_qrels_and_missing_review_artifacts(self):
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
                        "queries": [{"id": "q1", "query": "alpha", "relevant": {"c1": 1}}],
                    }
                ),
                encoding="utf-8",
            )
            suite_path = suite_dir / "suite.json"
            suite_path.write_text(
                json.dumps(
                    {
                        "suite_id": "unit_suite",
                        "thresholds": {"min_candidate_recall": 0.8},
                        "fusion_methods": ["union", "rrf"],
                        "benchmarks": [
                            {
                                "id": "unit",
                                "mode": "unit_mode",
                                "qrels": "gold/qrels.json",
                                "publishable_query_target": 3,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            settings = SimpleNamespace(root=root, reports_dir=reports_dir)
            with patch("canon.product.publishable_package.load_settings", return_value=settings):
                report = build_publishable_package(
                    suite_path=suite_path,
                    required_query_count=3,
                    write_report=False,
                )

        self.assertEqual(report["status"], "publication_blocked_evidence_incomplete")
        self.assertEqual(report["public_benchmark_coverage"]["status"], "pilot_only")
        self.assertIn("Complete human-reviewed qrels", report["next_required_work"][-1])
        blocked = {row["id"] for row in report["checks"] if not row["passed"]}
        self.assertIn("public_benchmark_coverage", blocked)
        self.assertIn("human_review_complete", blocked)

    def test_package_ready_when_required_artifacts_and_coverage_exist(self):
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
                            {"id": "q1", "query": "alpha", "relevant": {"c1": 1}},
                            {"id": "q2", "query": "beta", "relevant": {"c2": 1}},
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
                        "thresholds": {"min_candidate_recall": 0.8},
                        "fusion_methods": ["union", "rrf"],
                        "benchmarks": [
                            {
                                "id": "unit",
                                "mode": "unit_mode",
                                "qrels": "gold/qrels.json",
                                "publishable_query_target": 2,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            for name in ["automated_benchmark_suite_unit_suite.json"]:
                (reports_dir / name).write_text(
                    json.dumps({"suite_id": "unit_suite"}),
                    encoding="utf-8",
                )
            write_stage1_optimizer_report(reports_dir, suite_id="unit_suite", fusions=["union", "rrf"])
            (gold_dir / "disagreement_preservation_publishable.json").write_text(
                json.dumps({"benchmark_id": "unit_disagreement", "cases": []}),
                encoding="utf-8",
            )
            write_disagreement_report(reports_dir)
            write_scaffold_report(reports_dir, retrieval_target_met=True, synthesis_target_met=True)
            write_review_csvs(reports_dir)
            (reports_dir / "publishable_benchmark_card_v1.json").write_text(
                json.dumps(
                    {
                        "report_id": "publishable_benchmark_card_v1",
                        "suite_id": "unit_suite",
                        "status": "benchmark_card_ready_with_automated_failures",
                        "aggregate_diagnostics": {"status": "automated_fail"},
                        "claim_boundary": {"human_review_required": True},
                    }
                ),
                encoding="utf-8",
            )
            (reports_dir / "publishable_human_review_status_v1.json").write_text(
                json.dumps(
                    {
                        "report_id": "publishable_human_review_status_v1",
                        "suite_id": "unit_suite",
                        "status": "complete",
                        "retrieval_review": {"target_met": True},
                        "synthesis_review": {"target_met": True},
                        "validation_error_count": 0,
                    }
                ),
                encoding="utf-8",
            )
            settings = SimpleNamespace(root=root, reports_dir=reports_dir)
            with patch("canon.product.publishable_package.load_settings", return_value=settings):
                report = build_publishable_package(
                    suite_path=suite_path,
                    required_query_count=2,
                    write_report=False,
                )

        self.assertEqual(report["status"], "publishable_package_ready_pending_claim_review")
        self.assertTrue(all(row["passed"] for row in report["checks"]))
        artifact_paths = {row["path"] for row in report["artifact_manifest"]["artifacts"]}
        self.assertEqual(report["artifact_manifest"]["artifact_count"], 11)
        self.assertIn("gold/qrels.json", artifact_paths)
        self.assertIn("gold/disagreement_preservation_publishable.json", artifact_paths)
        self.assertIn("reports/disagreement_preservation_unit_disagreement.json", artifact_paths)
        self.assertIn("reports/publishable_retrieval_review_scaffold.csv", artifact_paths)
        self.assertIn("reports/publishable_synthesis_review_scaffold.csv", artifact_paths)

    def test_package_blocks_review_scaffold_below_configured_targets(self):
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
                            {"id": "q1", "query": "alpha", "relevant": {"c1": 1}},
                            {"id": "q2", "query": "beta", "relevant": {"c2": 1}},
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
                        "thresholds": {"min_candidate_recall": 0.8},
                        "fusion_methods": ["union", "rrf"],
                        "benchmarks": [
                            {
                                "id": "unit",
                                "mode": "unit_mode",
                                "qrels": "gold/qrels.json",
                                "publishable_query_target": 2,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (reports_dir / "automated_benchmark_suite_unit_suite.json").write_text(
                json.dumps({"suite_id": "unit_suite"}),
                encoding="utf-8",
            )
            write_stage1_optimizer_report(reports_dir, suite_id="unit_suite", fusions=["union", "rrf"])
            (gold_dir / "disagreement_preservation_publishable.json").write_text(
                json.dumps({"benchmark_id": "unit_disagreement", "cases": []}),
                encoding="utf-8",
            )
            write_disagreement_report(reports_dir)
            write_scaffold_report(reports_dir, retrieval_target_met=True, synthesis_target_met=False)
            write_review_csvs(reports_dir)
            (reports_dir / "publishable_benchmark_card_v1.json").write_text(
                json.dumps(
                    {
                        "report_id": "publishable_benchmark_card_v1",
                        "suite_id": "unit_suite",
                        "status": "benchmark_card_ready_with_automated_failures",
                        "aggregate_diagnostics": {"status": "automated_fail"},
                        "claim_boundary": {"human_review_required": True},
                    }
                ),
                encoding="utf-8",
            )
            (reports_dir / "publishable_human_review_status_v1.json").write_text(
                json.dumps(
                    {
                        "report_id": "publishable_human_review_status_v1",
                        "suite_id": "unit_suite",
                        "status": "complete",
                        "retrieval_review": {"target_met": True},
                        "synthesis_review": {"target_met": True},
                        "validation_error_count": 0,
                    }
                ),
                encoding="utf-8",
            )
            settings = SimpleNamespace(root=root, reports_dir=reports_dir)
            with patch("canon.product.publishable_package.load_settings", return_value=settings):
                report = build_publishable_package(
                    suite_path=suite_path,
                    required_query_count=2,
                    write_report=False,
                )

        blocked = {row["id"] for row in report["checks"] if not row["passed"]}
        self.assertNotIn("human_review_retrieval_scaffold_target", blocked)
        self.assertIn("human_review_synthesis_scaffold_target", blocked)
        self.assertIn("Expand the disagreement-preservation benchmark", report["next_required_work"][0])

    def test_package_blocks_missing_benchmark_card(self):
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
                        "queries": [{"id": "q1", "query": "alpha", "relevant": {"c1": 1}}],
                    }
                ),
                encoding="utf-8",
            )
            suite_path = suite_dir / "suite.json"
            suite_path.write_text(
                json.dumps(
                    {
                        "suite_id": "unit_suite",
                        "thresholds": {"min_candidate_recall": 0.8},
                        "fusion_methods": ["union", "rrf"],
                        "benchmarks": [
                            {
                                "id": "unit",
                                "mode": "unit_mode",
                                "qrels": "gold/qrels.json",
                                "publishable_query_target": 1,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            for name in ["automated_benchmark_suite_unit_suite.json"]:
                (reports_dir / name).write_text(
                    json.dumps({"suite_id": "unit_suite"}),
                    encoding="utf-8",
                )
            write_stage1_optimizer_report(reports_dir, suite_id="unit_suite", fusions=["union", "rrf"])
            (reports_dir / "disagreement_preservation_unit.json").write_text(
                json.dumps({"status": "automated_pass_human_review_required"}),
                encoding="utf-8",
            )
            (reports_dir / "publishable_human_review_scaffold_v1.json").write_text(
                json.dumps({"status": "review_scaffold_ready_human_labels_required"}),
                encoding="utf-8",
            )
            (reports_dir / "publishable_human_review_status_v1.json").write_text(
                json.dumps(
                    {
                        "report_id": "publishable_human_review_status_v1",
                        "suite_id": "unit_suite",
                        "status": "complete",
                        "retrieval_review": {"target_met": True},
                        "synthesis_review": {"target_met": True},
                        "validation_error_count": 0,
                    }
                ),
                encoding="utf-8",
            )
            settings = SimpleNamespace(root=root, reports_dir=reports_dir)
            with patch("canon.product.publishable_package.load_settings", return_value=settings):
                report = build_publishable_package(
                    suite_path=suite_path,
                    required_query_count=1,
                    write_report=False,
                )

        blocked = {row["id"] for row in report["checks"] if not row["passed"]}
        self.assertIn("benchmark_card_report", blocked)

    def test_package_does_not_accept_disagreement_report_from_different_benchmark(self):
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
                        "queries": [{"id": "q1", "query": "alpha", "relevant": {"c1": 1}}],
                    }
                ),
                encoding="utf-8",
            )
            suite_path = suite_dir / "suite.json"
            suite_path.write_text(
                json.dumps(
                    {
                        "suite_id": "unit_suite",
                        "thresholds": {"min_candidate_recall": 0.8},
                        "fusion_methods": ["union", "rrf"],
                        "benchmarks": [
                            {
                                "id": "unit",
                                "mode": "unit_mode",
                                "qrels": "gold/qrels.json",
                                "publishable_query_target": 1,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (gold_dir / "disagreement_preservation_publishable.json").write_text(
                json.dumps({"benchmark_id": "unit_disagreement", "cases": []}),
                encoding="utf-8",
            )
            (reports_dir / "disagreement_preservation_unit_disagreement.json").write_text(
                json.dumps(
                    {
                        "benchmark_id": "other_disagreement",
                        "status": "automated_pass_human_review_required",
                    }
                ),
                encoding="utf-8",
            )
            settings = SimpleNamespace(root=root, reports_dir=reports_dir)
            with patch("canon.product.publishable_package.load_settings", return_value=settings):
                report = build_publishable_package(
                    suite_path=suite_path,
                    required_query_count=1,
                    write_report=False,
                )

        blocked = {row["id"] for row in report["checks"] if not row["passed"]}
        self.assertIn("disagreement_preservation_report", blocked)

    def test_package_does_not_accept_legacy_human_review_status(self):
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
                        "queries": [{"id": "q1", "query": "alpha", "relevant": {"c1": 1}}],
                    }
                ),
                encoding="utf-8",
            )
            suite_path = suite_dir / "suite.json"
            suite_path.write_text(
                json.dumps(
                    {
                        "suite_id": "unit_suite",
                        "thresholds": {"min_candidate_recall": 0.8},
                        "fusion_methods": ["union", "rrf"],
                        "benchmarks": [
                            {
                                "id": "unit",
                                "mode": "unit_mode",
                                "qrels": "gold/qrels.json",
                                "publishable_query_target": 1,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            for name in ["automated_benchmark_suite_unit_suite.json"]:
                (reports_dir / name).write_text(
                    json.dumps({"suite_id": "unit_suite"}),
                    encoding="utf-8",
                )
            write_stage1_optimizer_report(reports_dir, suite_id="unit_suite", fusions=["union", "rrf"])
            (reports_dir / "disagreement_preservation_unit.json").write_text(
                json.dumps({"status": "automated_pass_human_review_required"}),
                encoding="utf-8",
            )
            (reports_dir / "publishable_human_review_scaffold_v1.json").write_text(
                json.dumps({"status": "review_scaffold_ready_human_labels_required"}),
                encoding="utf-8",
            )
            (reports_dir / "human_review_status_v1.json").write_text(
                json.dumps({"status": "complete"}),
                encoding="utf-8",
            )
            settings = SimpleNamespace(root=root, reports_dir=reports_dir)
            with patch("canon.product.publishable_package.load_settings", return_value=settings):
                report = build_publishable_package(
                    suite_path=suite_path,
                    required_query_count=1,
                    write_report=False,
                )

        blocked = {row["id"] for row in report["checks"] if not row["passed"]}
        self.assertIn("human_review_complete", blocked)

    def test_package_blocks_reports_from_different_suite(self):
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
                        "queries": [{"id": "q1", "query": "alpha", "relevant": {"c1": 1}}],
                    }
                ),
                encoding="utf-8",
            )
            suite_path = suite_dir / "suite.json"
            suite_path.write_text(
                json.dumps(
                    {
                        "suite_id": "unit_suite",
                        "thresholds": {"min_candidate_recall": 0.8},
                        "fusion_methods": ["union", "rrf"],
                        "benchmarks": [
                            {
                                "id": "unit",
                                "mode": "unit_mode",
                                "qrels": "gold/qrels.json",
                                "publishable_query_target": 1,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (reports_dir / "stage1_optimizer_v1.json").write_text(
                json.dumps({"suite_id": "other_suite"}),
                encoding="utf-8",
            )
            (reports_dir / "automated_benchmark_suite_unit_suite.json").write_text(
                json.dumps({"suite_id": "other_suite"}),
                encoding="utf-8",
            )
            settings = SimpleNamespace(root=root, reports_dir=reports_dir)
            with patch("canon.product.publishable_package.load_settings", return_value=settings):
                report = build_publishable_package(
                    suite_path=suite_path,
                    required_query_count=1,
                    write_report=False,
                )

        blocked = {row["id"] for row in report["checks"] if not row["passed"]}
        self.assertIn("resumable_stage1_execution", blocked)
        self.assertIn("automated_suite_report", blocked)

    def test_package_blocks_optimizer_missing_declared_fusion_trials(self):
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
                        "queries": [{"id": "q1", "query": "alpha", "relevant": {"c1": 1}}],
                    }
                ),
                encoding="utf-8",
            )
            suite_path = suite_dir / "suite.json"
            suite_path.write_text(
                json.dumps(
                    {
                        "suite_id": "unit_suite",
                        "thresholds": {"min_candidate_recall": 0.8},
                        "fusion_methods": ["union", "rrf"],
                        "benchmarks": [
                            {
                                "id": "unit",
                                "mode": "unit_mode",
                                "qrels": "gold/qrels.json",
                                "publishable_query_target": 1,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (reports_dir / "stage1_optimizer_v1.json").write_text(
                json.dumps(
                    {
                        "suite_id": "unit_suite",
                        "fusion_methods": ["union"],
                        "trials": [{"fusion": "union", "status": "ok"}],
                    }
                ),
                encoding="utf-8",
            )
            settings = SimpleNamespace(root=root, reports_dir=reports_dir)
            with patch("canon.product.publishable_package.load_settings", return_value=settings):
                report = build_publishable_package(
                    suite_path=suite_path,
                    required_query_count=1,
                    write_report=False,
                )

        blocked = {row["id"] for row in report["checks"] if not row["passed"]}
        self.assertIn("optimizer_fusion_coverage", blocked)

    def test_package_status_blocks_any_failed_check(self):
        self.assertEqual(
            package_status([{"passed": True}, {"passed": False}]),
            "publication_blocked_evidence_incomplete",
        )


def write_review_csvs(reports_dir: Path) -> None:
    (reports_dir / "publishable_retrieval_review_scaffold.csv").write_text(
        "review_set,benchmark_id,query_id,candidate_id\n",
        encoding="utf-8",
    )
    (reports_dir / "publishable_synthesis_review_scaffold.csv").write_text(
        "case_id,evidence_id,claim_id\n",
        encoding="utf-8",
    )


def write_scaffold_report(
    reports_dir: Path,
    *,
    retrieval_target_met: bool,
    synthesis_target_met: bool,
) -> None:
    (reports_dir / "publishable_human_review_scaffold_v1.json").write_text(
        json.dumps(
            {
                "status": "review_scaffold_ready_human_labels_required",
                "completion_status": {
                    "retrieval_query_count": 2 if retrieval_target_met else 1,
                    "retrieval_target": 2,
                    "retrieval_target_met": retrieval_target_met,
                    "synthesis_case_count": 2 if synthesis_target_met else 1,
                    "synthesis_target": 2,
                    "synthesis_target_met": synthesis_target_met,
                },
            }
        ),
        encoding="utf-8",
    )


def write_disagreement_report(reports_dir: Path) -> None:
    (reports_dir / "disagreement_preservation_unit_disagreement.json").write_text(
        json.dumps(
            {
                "benchmark_id": "unit_disagreement",
                "status": "automated_pass_human_review_required",
            }
        ),
        encoding="utf-8",
    )


def write_stage1_optimizer_report(reports_dir: Path, suite_id: str, fusions: list[str]) -> None:
    (reports_dir / "stage1_optimizer_v1.json").write_text(
        json.dumps(
            {
                "suite_id": suite_id,
                "fusion_methods": fusions,
                "trials": [{"fusion": fusion, "status": "ok"} for fusion in fusions],
            }
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    unittest.main()
