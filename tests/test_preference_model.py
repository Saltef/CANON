import unittest

from canon.modeling.preference_model import (
    fit_pairwise_model,
    leave_one_query_out_pairs,
    pairwise_examples,
    preference_report,
)
from canon.modeling.calibration_model import CalibrationExample


class PreferenceModelTests(unittest.TestCase):
    def test_pairwise_examples_include_forward_and_reverse_rows(self):
        pairs = pairwise_examples(sample_examples(), min_margin=0.01)
        self.assertEqual(len(pairs), 4)
        self.assertTrue(any(pair.target == 1.0 for pair in pairs))
        self.assertTrue(any(pair.target == 0.0 for pair in pairs))
        self.assertGreater(pairs[0].margin, 0.0)

    def test_pairwise_model_predicts_bounded_preference_probability(self):
        pairs = pairwise_examples(sample_examples(), min_margin=0.01)
        model = fit_pairwise_model(pairs, ridge_lambda=0.5)
        prediction = model.preference_probability(pairs[0].delta_features)
        self.assertGreaterEqual(prediction, 0.0)
        self.assertLessEqual(prediction, 1.0)
        self.assertIn("citation_support_score", model.weights)

    def test_leave_one_query_out_pairs_reports_accuracy(self):
        pairs = pairwise_examples(sample_examples(), min_margin=0.01)
        report = leave_one_query_out_pairs(pairs, ridge_lambda=0.5)
        self.assertIn("accuracy", report["metrics"])
        self.assertEqual(len(report["rows"]), len(pairs))

    def test_preference_report_contains_policy_summary(self):
        report = preference_report(
            examples=sample_examples(),
            mode="unit",
            source_report="technical_calibration",
            ridge_lambda=0.5,
            min_margin=0.01,
            output_name="unit_preference_model_test.json",
        )
        self.assertEqual(report["example_count"], len(sample_examples()))
        self.assertEqual(report["pair_count"], 4)
        self.assertIn("policy_target_summary", report)


if __name__ == "__main__":
    unittest.main()


def sample_examples():
    feature_names = [
        "citation_support_score",
        "answer_faithfulness_score",
        "importance_fit_score",
        "paradigm_coverage_score",
        "evidence_role_fit_score",
        "source_trust_fit_score",
        "safety_handling_score",
        "corroboration_handling_score",
        "citation_count",
        "claim_citation_count",
        "conflict_note_count",
        "distinct_sources",
        "distinct_clusters",
        "relevance",
        "semantic_similarity",
        "source_quality",
        "method_signal",
        "limitation_signal",
        "evidence_specificity",
        "uncertainty_signal",
    ]
    low = {name: 0.0 for name in feature_names}
    high = {name: 1.0 for name in feature_names}
    medium = {name: 0.5 for name in feature_names}
    return [
        CalibrationExample("q1::bad", "q1", "bad", low, 0.125, "technical_proxy", 0.125),
        CalibrationExample("q1::good", "q1", "good", high, 0.875, "technical_proxy", 0.875),
        CalibrationExample("q2::mid", "q2", "mid", medium, 0.5625, "technical_proxy", 0.5625),
        CalibrationExample("q2::good", "q2", "good", high, 0.875, "technical_proxy", 0.875),
    ]
