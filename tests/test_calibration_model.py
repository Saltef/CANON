import unittest

from canon.modeling.calibration_model import (
    CalibrationExample,
    examples_from_technical_report,
    fit_ridge_model,
    leave_one_query_out,
    modeling_report,
)


class CalibrationModelTests(unittest.TestCase):
    def test_examples_from_technical_report_extract_features_and_targets(self):
        examples = examples_from_technical_report(
            {
                "rows": [
                    {
                        "task_id": "q1::rag",
                        "query_id": "q1",
                        "policy": "rag",
                        "dimension_scores": {"citation_support": 1.0},
                        "technical_calibration_score": 0.75,
                        "metrics": {"citation_count": 3, "distinct_sources": 2},
                        "average_components": {"relevance": 0.8, "source_quality": 0.6},
                    }
                ]
            }
        )
        self.assertEqual(len(examples), 1)
        self.assertEqual(examples[0].target, 0.75)
        self.assertGreater(examples[0].features["citation_count"], 0.0)
        self.assertEqual(examples[0].features["relevance"], 0.8)

    def test_ridge_model_predicts_bounded_scores(self):
        examples = sample_examples()
        model = fit_ridge_model(examples, ridge_lambda=0.5)
        prediction = model.predict(examples[0].features)
        self.assertGreaterEqual(prediction, 0.0)
        self.assertLessEqual(prediction, 1.0)
        self.assertIn("citation_support_score", model.weights)

    def test_leave_one_query_out_reports_model_and_proxy_errors(self):
        report = leave_one_query_out(sample_examples(), ridge_lambda=0.5)
        self.assertIn("model_mae", report["metrics"])
        self.assertIn("proxy_mae", report["metrics"])
        self.assertEqual(len(report["rows"]), len(sample_examples()))

    def test_modeling_report_contains_feature_rankings(self):
        report = modeling_report(
            examples=sample_examples(),
            mode="unit",
            source_report="technical_calibration",
            ridge_lambda=0.5,
            output_name="unit_calibration_model_test.json",
        )
        self.assertEqual(report["example_count"], len(sample_examples()))
        self.assertIn("model", report)
        self.assertIn("leave_one_query_out_metrics", report)


def sample_examples():
    return [
        CalibrationExample(
            task_id="q1::bad",
            query_id="q1",
            policy="bad",
            features={
                "citation_support_score": 0.0,
                "answer_faithfulness_score": 0.0,
                "importance_fit_score": 0.0,
                "paradigm_coverage_score": 0.0,
                "evidence_role_fit_score": 0.0,
                "source_trust_fit_score": 0.0,
                "safety_handling_score": 1.0,
                "corroboration_handling_score": 0.0,
                "citation_count": 0.0,
                "claim_citation_count": 0.0,
                "conflict_note_count": 0.0,
                "distinct_sources": 0.0,
                "distinct_clusters": 0.0,
                "relevance": 0.0,
                "semantic_similarity": 0.0,
                "source_quality": 0.0,
                "method_signal": 0.0,
                "limitation_signal": 0.0,
                "evidence_specificity": 0.0,
                "uncertainty_signal": 0.0,
            },
            target=0.125,
            target_source="technical_proxy",
            proxy_baseline=0.125,
        ),
        CalibrationExample(
            task_id="q2::good",
            query_id="q2",
            policy="good",
            features={
                "citation_support_score": 1.0,
                "answer_faithfulness_score": 1.0,
                "importance_fit_score": 1.0,
                "paradigm_coverage_score": 0.5,
                "evidence_role_fit_score": 1.0,
                "source_trust_fit_score": 1.0,
                "safety_handling_score": 1.0,
                "corroboration_handling_score": 0.5,
                "citation_count": 0.6,
                "claim_citation_count": 0.4,
                "conflict_note_count": 0.3,
                "distinct_sources": 0.4,
                "distinct_clusters": 0.4,
                "relevance": 0.8,
                "semantic_similarity": 0.8,
                "source_quality": 0.7,
                "method_signal": 0.2,
                "limitation_signal": 0.1,
                "evidence_specificity": 0.3,
                "uncertainty_signal": 0.0,
            },
            target=0.875,
            target_source="technical_proxy",
            proxy_baseline=0.875,
        ),
        CalibrationExample(
            task_id="q3::mid",
            query_id="q3",
            policy="mid",
            features={
                "citation_support_score": 0.5,
                "answer_faithfulness_score": 0.5,
                "importance_fit_score": 1.0,
                "paradigm_coverage_score": 0.0,
                "evidence_role_fit_score": 1.0,
                "source_trust_fit_score": 0.5,
                "safety_handling_score": 1.0,
                "corroboration_handling_score": 0.0,
                "citation_count": 0.2,
                "claim_citation_count": 0.2,
                "conflict_note_count": 0.0,
                "distinct_sources": 0.2,
                "distinct_clusters": 0.2,
                "relevance": 0.5,
                "semantic_similarity": 0.7,
                "source_quality": 0.5,
                "method_signal": 0.0,
                "limitation_signal": 0.0,
                "evidence_specificity": 0.0,
                "uncertainty_signal": 0.0,
            },
            target=0.5625,
            target_source="technical_proxy",
            proxy_baseline=0.5625,
        ),
    ]


if __name__ == "__main__":
    unittest.main()
