import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from canon.product.publishable_workflow import run_publishable_workflow


class PublishableWorkflowTests(unittest.TestCase):
    def test_workflow_freezes_from_existing_benchmark_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reports = root / "reports"
            suite_path = root / "conf" / "benchmark_suites" / "suite.json"
            reports.mkdir()
            suite_path.parent.mkdir(parents=True)
            write_suite(suite_path)
            (reports / "automated_benchmark_suite_unit_suite.json").write_text("{}", encoding="utf-8")
            (reports / "stage1_optimizer_v1.json").write_text("{}", encoding="utf-8")
            settings = SimpleNamespace(root=root, reports_dir=reports)
            with patch("canon.product.publishable_workflow.load_settings", return_value=settings):
                with patched_light_stages() as calls:
                    report = run_publishable_workflow(
                        suite_path=suite_path,
                        run_benchmarks=False,
                        write_report=False,
                    )

        self.assertEqual(report["status"], "workflow_complete_human_review_blocked")
        self.assertEqual(report["suite_id"], "unit_suite")
        stages = {stage["id"]: stage["status"] for stage in report["stages"]}
        self.assertEqual(stages["automated_benchmark_suite"], "existing_artifact")
        self.assertEqual(stages["stage1_optimizer"], "existing_artifact")
        self.assertEqual(stages["publishable_benchmark_card"], "ok")
        self.assertEqual(calls["benchmark_suite"].call_count, 0)
        self.assertEqual(calls["optimizer"].call_count, 0)
        self.assertEqual(
            report["next_required_work"],
            ["Expand the disagreement-preservation benchmark and synthesis review scaffold."],
        )

    def test_workflow_reports_missing_benchmark_artifacts_without_crashing_card(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reports = root / "reports"
            suite_path = root / "conf" / "benchmark_suites" / "suite.json"
            reports.mkdir()
            suite_path.parent.mkdir(parents=True)
            write_suite(suite_path)
            settings = SimpleNamespace(root=root, reports_dir=reports)
            with patch("canon.product.publishable_workflow.load_settings", return_value=settings):
                with patched_light_stages() as calls:
                    report = run_publishable_workflow(
                        suite_path=suite_path,
                        run_benchmarks=False,
                        write_report=False,
                    )

        self.assertEqual(report["status"], "workflow_blocked_missing_benchmark_artifact")
        stages = {stage["id"]: stage["status"] for stage in report["stages"]}
        self.assertEqual(stages["automated_benchmark_suite"], "missing_required_artifact")
        self.assertEqual(stages["stage1_optimizer"], "missing_required_artifact")
        self.assertEqual(stages["publishable_benchmark_card"], "skipped")
        self.assertEqual(calls["benchmark_card"].call_count, 0)

    def test_workflow_can_run_resumable_benchmark_stages_when_requested(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reports = root / "reports"
            suite_path = root / "conf" / "benchmark_suites" / "suite.json"
            reports.mkdir()
            suite_path.parent.mkdir(parents=True)
            write_suite(suite_path)
            settings = SimpleNamespace(root=root, reports_dir=reports)
            with patch("canon.product.publishable_workflow.load_settings", return_value=settings):
                with patched_light_stages() as calls:
                    report = run_publishable_workflow(
                        suite_path=suite_path,
                        run_benchmarks=True,
                        model_providers=["local"],
                        rerankers=["heuristic"],
                        candidate_values=[50],
                        write_report=False,
                    )

        self.assertEqual(report["status"], "workflow_complete_human_review_blocked")
        self.assertEqual(calls["benchmark_suite"].call_count, 1)
        self.assertEqual(calls["optimizer"].call_count, 1)
        benchmark_kwargs = calls["benchmark_suite"].call_args.kwargs
        optimizer_kwargs = calls["optimizer"].call_args.kwargs
        self.assertTrue(benchmark_kwargs["resume"])
        self.assertTrue(optimizer_kwargs["resume"])
        self.assertEqual(benchmark_kwargs["model_providers"], ["local"])
        self.assertEqual(optimizer_kwargs["fusion_methods"], ["union", "rrf"])


class patched_light_stages:
    def __enter__(self):
        self.patches = {
            "benchmark_suite": patch(
                "canon.product.publishable_workflow.run_automated_benchmark_suite",
                return_value={"report_id": "automated_benchmark_suite_v1", "status": "automated_fail", "suite_id": "unit_suite"},
            ),
            "optimizer": patch(
                "canon.product.publishable_workflow.run_stage1_optimizer",
                return_value={"report_id": "stage1_optimizer_v1", "status": "automated_fail", "suite_id": "unit_suite"},
            ),
            "disagreement": patch(
                "canon.product.publishable_workflow.evaluate_disagreement_preservation",
                return_value={
                    "report_id": "disagreement_preservation_benchmark_v1",
                    "status": "automated_pass_human_review_required",
                    "benchmark_id": "unit_disagreement",
                    "case_count": 1,
                },
            ),
            "scaffold": patch(
                "canon.product.publishable_workflow.build_human_review_scaffold",
                return_value={
                    "report_id": "publishable_human_review_scaffold_v1",
                    "status": "review_scaffold_ready_human_labels_required",
                    "suite_id": "unit_suite",
                    "completion_status": {"status": "labels_required"},
                },
            ),
            "review": patch(
                "canon.product.publishable_workflow.build_publishable_review_status",
                return_value={
                    "report_id": "publishable_human_review_status_v1",
                    "status": "incomplete",
                    "suite_id": "unit_suite",
                    "reviewed_label_count": 0,
                    "validation_error_count": 0,
                },
            ),
            "benchmark_card": patch(
                "canon.product.publishable_workflow.build_publishable_benchmark_card",
                return_value={
                    "report_id": "publishable_benchmark_card_v1",
                    "status": "benchmark_card_ready_with_automated_failures",
                    "suite_id": "unit_suite",
                    "aggregate_diagnostics": {"status": "automated_fail"},
                },
            ),
            "package": patch(
                "canon.product.publishable_workflow.build_publishable_package",
                return_value={
                    "report_id": "publishable_evaluation_package_v1",
                    "status": "publication_blocked_evidence_incomplete",
                    "suite": {"suite_id": "unit_suite"},
                    "artifact_manifest": {"artifact_count": 1},
                    "next_required_work": [
                        "Expand the disagreement-preservation benchmark and synthesis review scaffold.",
                    ],
                },
            ),
        }
        self.mocks = {key: patcher.start() for key, patcher in self.patches.items()}
        return self.mocks

    def __exit__(self, exc_type, exc_value, traceback):
        for patcher in reversed(list(self.patches.values())):
            patcher.stop()


def write_suite(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "suite_id": "unit_suite",
                "fusion_methods": ["union", "rrf"],
                "benchmarks": [
                    {
                        "id": "unit",
                        "mode": "unit_mode",
                        "qrels": "gold/unit_qrels.json",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    unittest.main()
