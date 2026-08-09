import json
import unittest
from unittest.mock import patch

from canon.model_tasks import run_evidence_model_review, validate_openrouter_model


class ModelTaskTests(unittest.TestCase):
    def test_openrouter_model_review_sanitizes_to_known_evidence_ids(self):
        payload = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "stance_assessments": [
                                    {
                                        "evidence_id": "C1",
                                        "stance": "supports",
                                        "claim": "Grid risk increases.",
                                        "confidence": 0.82,
                                        "excerpt": "Grid risk increases.",
                                        "rationale": "Directly addresses the query.",
                                    },
                                    {"evidence_id": "C9", "stance": "supports", "claim": "Invalid."},
                                ],
                                "extracted_dimensions": [
                                    {
                                        "evidence_id": "C1",
                                        "dimension": "measurement",
                                        "value": "peak load",
                                        "excerpt": "peak load",
                                    }
                                ],
                                "disagreement_diagnosis": {
                                    "axis": "measurement_difference",
                                    "confidence": 0.7,
                                    "summary": "Sources use different grid measurements.",
                                    "evidence_ids": ["C1", "C9"],
                                },
                                "draft_plan": {
                                    "safe_to_draft": True,
                                    "missing_evidence": ["utility filings"],
                                    "suggested_next_queries": ["AI data center utility filings"],
                                },
                            }
                        )
                    }
                }
            ]
        }

        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "key"}, clear=True):
            with patch("canon.model_tasks.load_local_env"):
                with patch("canon.model_tasks.post_json", return_value=payload):
                    report = run_evidence_model_review(
                        query="grid risk",
                        evidence=[{"evidence_id": "C1", "text": "Grid risk increases.", "title": "Grid"}],
                        allow_external_data=True,
                    )

        self.assertEqual(report["status"], "model_review_ready")
        self.assertEqual([row["evidence_id"] for row in report["stance_assessments"]], ["C1"])
        self.assertEqual(report["disagreement_diagnosis"]["evidence_ids"], ["C1"])
        self.assertTrue(report["human_review_required"])

    def test_model_review_rejects_gemini_and_cohere_models(self):
        with self.assertRaises(ValueError):
            validate_openrouter_model("google/gemini-2.5-flash")
        with self.assertRaises(ValueError):
            validate_openrouter_model("cohere/command-r")


if __name__ == "__main__":
    unittest.main()
