import unittest
from unittest.mock import patch

from canon.eval.diversity_gate import run_diversity_gate


AUDIT = {
    "aggregate": {
        "verdict_counts": {
            "useful_breadth": 10,
            "off_topic_breadth_risk": 1,
            "mixed_breadth": 10,
        },
        "average_noise_rate": 0.013333,
        "average_cluster_delta": 1.76,
        "average_breadth_precision": 0.387333,
    }
}


class DiversityGateTests(unittest.TestCase):
    def test_diversity_gate_passes_when_thresholds_are_met(self):
        with patch("canon.eval.diversity_gate.load_diversity_audit", return_value=AUDIT):
            report = run_diversity_gate(
                "ignored",
                thresholds={
                    "useful_breadth_count_min": 10,
                    "off_topic_breadth_risk_max": 1,
                    "noisy_breadth_count_max": 0,
                    "average_noise_rate_max": 0.02,
                    "average_cluster_delta_min": 1.5,
                    "average_breadth_precision_min": 0.35,
                },
            )
        self.assertEqual(report["status"], "pass")

    def test_diversity_gate_fails_when_useful_breadth_is_too_low(self):
        with patch("canon.eval.diversity_gate.load_diversity_audit", return_value=AUDIT):
            report = run_diversity_gate(
                "ignored",
                thresholds={
                    "useful_breadth_count_min": 11,
                    "off_topic_breadth_risk_max": 1,
                    "noisy_breadth_count_max": 0,
                    "average_noise_rate_max": 0.02,
                    "average_cluster_delta_min": 1.5,
                    "average_breadth_precision_min": 0.35,
                },
            )
        self.assertEqual(report["status"], "fail")


if __name__ == "__main__":
    unittest.main()
