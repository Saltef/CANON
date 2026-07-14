import copy
import unittest

from canon.eval.contracts import validate_report
from canon.labeling.unstructured_calibration import calibrate_unstructured_label_tasks
from canon.labeling.unstructured_tasks import build_unstructured_label_tasks


class UnstructuredLabelCalibrationTests(unittest.TestCase):
    def test_empty_label_pack_reports_needs_labels(self):
        tasks = build_unstructured_label_tasks(write_report=False, max_tasks=20)
        report = calibrate_unstructured_label_tasks(tasks)

        self.assertEqual(report["status"], "needs_labels")
        self.assertEqual(report["completed_label_count"], 0)
        self.assertEqual(report["agreement"]["overall_comparison_count"], 0)
        self.assertEqual(report["contract_validation"]["status"], "pass")

    def test_completed_label_pack_reports_agreement(self):
        tasks = build_unstructured_label_tasks(write_report=False, max_tasks=20)
        filled = copy.deepcopy(tasks)
        first = filled["tasks"][0]
        first["human_labels"]["safety_disposition"] = "allow"

        report = calibrate_unstructured_label_tasks(filled)

        self.assertEqual(report["status"], "calibrated")
        self.assertEqual(report["completed_label_count"], 1)
        self.assertEqual(report["agreement"]["overall_comparison_count"], 1)

    def test_chunking_quality_labels_are_calibrated(self):
        tasks = build_unstructured_label_tasks(write_report=False, max_tasks=40)
        filled = copy.deepcopy(tasks)
        chunking_task = next(task for task in filled["tasks"] if "chunking_quality" in task["label_families"])
        chunking_task["human_labels"]["preferred_chunking_variant"] = "boundary_aware"

        report = calibrate_unstructured_label_tasks(filled)

        self.assertEqual(report["status"], "calibrated")
        self.assertEqual(report["completed_label_count"], 1)
        self.assertEqual(report["agreement"]["by_label"]["preferred_chunking_variant"]["comparison_count"], 1)

    def test_contract_rejects_calibrated_report_without_completed_labels(self):
        tasks = build_unstructured_label_tasks(write_report=False, max_tasks=20)
        report = calibrate_unstructured_label_tasks(tasks)
        report["status"] = "calibrated"

        validation = validate_report(report, "unstructured_label_calibration")

        self.assertEqual(validation["status"], "fail")
        self.assertTrue(any("calibrated report must contain completed labels" in error for error in validation["errors"]))


if __name__ == "__main__":
    unittest.main()
