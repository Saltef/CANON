import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from canon.claims.model import ClaimModel, load_claim_model, write_model_report


class ClaimModelTests(unittest.TestCase):
    def test_load_claim_model(self):
        model = load_claim_model()
        self.assertEqual(model.model_id, "claim-model-baseline-v1")

    def test_predict_claim(self):
        model = load_claim_model()
        prediction = model.predict("We find robust support for the democratic peace.")
        self.assertTrue(prediction.is_claim)
        self.assertEqual(prediction.stance, "supportive")
        self.assertIn("claim_marker_count", prediction.features)

    def test_predict_non_claim(self):
        model = load_claim_model()
        prediction = model.predict("This article is about politics.")
        self.assertFalse(prediction.is_claim)

    def test_model_can_be_constructed_from_artifact(self):
        model = ClaimModel(
            {
                "model_id": "demo",
                "model_type": "demo",
                "claim_threshold": 0.5,
                "confidence_intercept": 0.0,
                "feature_weights": {"claim_marker_count": 1.0},
                "positive_markers": ["support"],
                "negative_markers": [],
                "finding_markers": [],
                "evidence_markers": [],
                "method_markers": [],
                "hedge_markers": [],
            }
        )
        self.assertTrue(model.predict("support").is_claim)

    def test_write_model_report(self):
        with tempfile.TemporaryDirectory() as directory:
            reports_dir = Path(directory)
            with patch.dict("os.environ", {"CANON_REPORTS_DIR": str(reports_dir)}):
                report = write_model_report(
                    "unit",
                    ["We find evidence of a significant effect.", "Background context only."],
                )
            report_exists = (reports_dir / "claim_model_unit.json").exists()
        self.assertEqual(report["mode"], "unit")
        self.assertEqual(report["candidate_count"], 2)
        self.assertEqual(report["claim_count"], 1)
        self.assertTrue(report_exists)


if __name__ == "__main__":
    unittest.main()
