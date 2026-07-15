import unittest
from unittest.mock import patch

from canon.product.acceptance_scenario import acceptance_checks, run_acceptance_scenario


class AcceptanceScenarioTests(unittest.TestCase):
    def test_acceptance_checks_pass_with_review_ready_artifacts(self):
        checks = acceptance_checks(
            project_config={"status": "ready"},
            frame_coverage=frame_coverage(),
            brief=brief(),
            quality_gate=quality_gate(),
            alert_digest={"status": "ready_for_human_review"},
            review_packet={"records": [{"id": "q1"}]},
        )

        self.assertTrue(all(check["passed"] for check in checks))
        self.assertIn("grounded_claims_and_citations", [check["id"] for check in checks])
        self.assertIn("public_opinion_or_gap_visible", [check["id"] for check in checks])

    def test_run_acceptance_scenario_orchestrates_product_checks(self):
        with patch("canon.product.acceptance_scenario.build_project_config", return_value={"status": "ready"}):
            with patch("canon.product.acceptance_scenario.service.frame_coverage_report", return_value=frame_coverage()):
                with patch("canon.product.acceptance_scenario.service.intelligence_brief", return_value=brief()):
                    with patch("canon.product.acceptance_scenario.service.report_quality_gate", return_value=quality_gate()):
                        with patch("canon.product.acceptance_scenario.service.alert_digest", return_value={"status": "ready_for_human_review"}):
                            with patch("canon.product.acceptance_scenario.service.intelligence_review_prepare", return_value={"records": [{"id": "q1"}]}):
                                report = run_acceptance_scenario(write_report=False)

        self.assertEqual(report["report_id"], "flagship_acceptance_scenario_v1")
        self.assertEqual(report["status"], "automated_pass_human_review_required")
        self.assertTrue(report["human_review_boundary"]["required"])


def frame_coverage():
    return {
        "status": "coverage_gap_human_review_required",
        "frame_coverage": {
            "coverage": {
                "regions": {"covered": ["Brazil", "Chile"]},
                "subdomains": {"covered": ["public opinion"]},
            }
        },
        "coverage_gaps": [],
    }


def brief():
    return {
        "status": "ready_for_human_review",
        "agent_outputs": [
            {"claims": [{"issue_categories": ["energy"]}]},
            {"claims": [{"issue_categories": ["water"]}]},
            {"claims": [{"issue_categories": ["public opinion"]}]},
        ],
        "brief": {
            "contradictions_and_uncertainty": "Coverage remains provisional.",
            "next_signals_to_watch": ["local hearing"],
        },
    }


def quality_gate():
    return {
        "status": "pass_human_review_required",
        "grounding": {
            "unsupported_claim_count": 0,
            "grounded_claim_ratio": 1.0,
        },
    }


if __name__ == "__main__":
    unittest.main()
