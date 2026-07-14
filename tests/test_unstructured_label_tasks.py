import unittest

from canon.eval.contracts import validate_report
from canon.labeling.unstructured_tasks import (
    LABEL_FAMILY_TO_FIELDS,
    build_unstructured_label_tasks,
    coverage_summary,
)


class UnstructuredLabelTaskTests(unittest.TestCase):
    def test_builds_stratified_unstructured_label_task_pack(self):
        report = build_unstructured_label_tasks(write_report=False, max_tasks=40)

        self.assertEqual(report["report_id"], "unstructured_human_label_tasks_v1")
        self.assertGreater(report["task_count"], 0)
        self.assertTrue(report["coverage_summary"]["ready_for_annotation"])
        self.assertEqual(report["contract_validation"]["status"], "pass")
        for family in LABEL_FAMILY_TO_FIELDS:
            self.assertIn(family, report["coverage_summary"]["covered_label_families"])

    def test_includes_chunking_variant_review_tasks(self):
        report = build_unstructured_label_tasks(write_report=False, max_tasks=40)
        task_families = {task["task_family"] for task in report["tasks"]}
        chunking_tasks = [
            task for task in report["tasks"]
            if task["task_family"] == "chunking_variant_review"
        ]

        self.assertIn("chunking_variant_review", task_families)
        self.assertTrue(chunking_tasks)
        self.assertIn("chunking_quality", chunking_tasks[0]["label_families"])
        self.assertIn("preferred_chunking_variant", chunking_tasks[0]["human_labels"])

    def test_contract_rejects_missing_human_label_slot(self):
        report = build_unstructured_label_tasks(write_report=False, max_tasks=20)
        first = report["tasks"][0]
        first_family = first["label_families"][0]
        first_field = LABEL_FAMILY_TO_FIELDS[first_family][0]
        del first["human_labels"][first_field]

        validation = validate_report(report, "unstructured_label_tasks")

        self.assertEqual(validation["status"], "fail")
        self.assertTrue(any(first_field in error for error in validation["errors"]))

    def test_coverage_summary_blocks_missing_required_label_family(self):
        summary = coverage_summary(
            [
                {
                    "task_family": "chunk_evidence_review",
                    "domain": "economics",
                    "document_type": "academic_article",
                    "label_families": ["document_type"],
                    "deterministic_signals": {"safety": {"decision": "allow"}},
                }
            ]
        )

        self.assertFalse(summary["ready_for_annotation"])
        self.assertIn("answer_usefulness", summary["missing_label_families"])


if __name__ == "__main__":
    unittest.main()
