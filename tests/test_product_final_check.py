import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from canon.product.final_check import run_final_check, should_exit_nonzero


class ProductFinalCheckTests(unittest.TestCase):
    def test_final_check_passes_when_release_audit_passes(self):
        report = run_final_check(steps=steps_with_audit(audit_report("pass", all_pass=True)), write_report=False)

        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["blocking_reason"], "none")
        self.assertIn("summary", report["steps"][0])
        self.assertNotIn("report", report["steps"][0])

    def test_final_check_blocks_without_failing_for_missing_human_review(self):
        report = run_final_check(
            steps=steps_with_audit(
                audit_report(
                    "blocked",
                    human_review_passed=False,
                    industry_pilot_passed=False,
                    next_action="Complete human review labels for 30 remaining acceptance questions.",
                )
            ),
            write_report=False,
        )

        self.assertEqual(report["status"], "blocked_human_review")
        self.assertEqual(report["blocking_reason"], "human_review_required")
        self.assertIn("Complete human review", report["next_required_action"])

    def test_final_check_fails_for_automated_gate_failure(self):
        report = run_final_check(
            steps=steps_with_audit(
                audit_report(
                    "blocked",
                    smoke_passed=False,
                    next_action="Run and fix python -m canon.product.smoke.",
                )
            ),
            write_report=False,
        )

        self.assertEqual(report["status"], "fail")
        self.assertEqual(report["blocking_reason"], "product_smoke")

    def test_final_check_exit_policy_fails_blocked_release_by_default(self):
        self.assertFalse(should_exit_nonzero("pass"))
        self.assertTrue(should_exit_nonzero("fail"))
        self.assertTrue(should_exit_nonzero("blocked_human_review"))
        self.assertFalse(should_exit_nonzero("blocked_human_review", no_fail=True))

    def test_final_check_records_source_artifact_fingerprints(self):
        with TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            reports_dir = Path(temp_dir) / "reports"
            reports_dir.mkdir()
            mode = "social_science_ir_v1_harvest10"
            records_path = Path("reports") / "human_review_tasks_v1.json"
            (workspace / records_path).write_text('{"records": []}\n', encoding="utf-8")
            (reports_dir / f"product_smoke_{mode}.json").write_text('{"status": "pass"}\n', encoding="utf-8")

            report = run_final_check(
                steps=steps_with_audit(audit_report("pass", all_pass=True)),
                records_path=records_path,
                write_report=False,
                reports_dir=reports_dir,
            )

        first = report["source_artifacts"][0]
        second = report["source_artifacts"][1]
        missing = report["source_artifacts"][2]
        self.assertEqual(first["path"], "reports\\human_review_tasks_v1.json")
        self.assertTrue(first["exists"])
        self.assertEqual(len(first["sha256"]), 64)
        self.assertEqual(second["path"], f"reports\\product_smoke_{mode}.json")
        self.assertTrue(second["exists"])
        self.assertFalse(missing["exists"])
        self.assertIsNone(missing["sha256"])

    def test_final_check_fails_when_review_task_packet_is_missing(self):
        with TemporaryDirectory() as temp_dir:
            reports_dir = Path(temp_dir) / "reports"
            reports_dir.mkdir()

            report = run_final_check(
                steps=steps_with_audit(audit_report("pass", all_pass=True)),
                records_path=Path("reports") / "missing_tasks.json",
                write_report=False,
                reports_dir=reports_dir,
            )

        self.assertEqual(report["status"], "fail")
        self.assertEqual(report["blocking_reason"], "review_task_packet_integrity")
        self.assertEqual(report["integrity_errors"][0]["field"], "path")

    def test_final_check_fails_when_review_task_packet_is_malformed(self):
        with TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            reports_dir = Path(temp_dir) / "reports"
            reports_dir.mkdir()
            records_path = Path("reports") / "human_review_tasks_v1.json"
            (workspace / records_path).write_text("", encoding="utf-8")

            report = run_final_check(
                steps=steps_with_audit(audit_report("pass", all_pass=True)),
                records_path=records_path,
                write_report=False,
                reports_dir=reports_dir,
            )

        self.assertEqual(report["status"], "fail")
        self.assertEqual(report["blocking_reason"], "review_task_packet_integrity")
        self.assertEqual(report["integrity_errors"][0]["field"], "json")


def steps_with_audit(audit):
    return {
        "product_smoke": lambda: {"status": "pass"},
        "human_review_status": lambda: {"status": "complete"},
        "industry_pilot": lambda: {"status": "pass"},
        "product_readiness": lambda: {"status": "pass"},
        "product_release_audit": lambda: audit,
    }


def audit_report(
    status,
    all_pass=False,
    source_integrity_passed=True,
    smoke_passed=True,
    readiness_passed=True,
    human_review_passed=True,
    industry_pilot_passed=True,
    next_action="No required action remains.",
):
    if all_pass:
        source_integrity_passed = True
        smoke_passed = True
        readiness_passed = True
        human_review_passed = True
        industry_pilot_passed = True
    return {
        "status": status,
        "next_required_action": next_action,
        "components": [
            {"id": "source_report_integrity", "passed": source_integrity_passed},
            {"id": "product_smoke", "passed": smoke_passed},
            {"id": "product_readiness", "passed": readiness_passed},
            {"id": "human_review_status", "passed": human_review_passed},
            {"id": "industry_pilot", "passed": industry_pilot_passed},
        ],
    }


if __name__ == "__main__":
    unittest.main()
