import unittest

from canon.intelligence.alerts import alert_quality_report, build_alert_digest, duplicate_alert_report


class AlertDigestTests(unittest.TestCase):
    def test_build_alert_digest_creates_evidence_grounded_alerts(self):
        digest = build_alert_digest(brief_report())

        self.assertEqual(digest["report_id"], "evidence_alert_digest_v1")
        self.assertEqual(digest["status"], "ready_for_human_review")
        self.assertGreater(digest["alert_count"], 0)
        first = digest["alerts"][0]
        self.assertTrue(first["evidence_ids"])
        self.assertTrue(first["evidence_trigger"])
        self.assertTrue(first["recommended_follow_up"])

    def test_duplicate_alert_report_measures_duplicate_signatures(self):
        alerts = [
            {"issue": "water", "affected_regions": ["Brazil"], "evidence_ids": ["C1"]},
            {"issue": "water", "affected_regions": ["Brazil"], "evidence_ids": ["C1"]},
            {"issue": "grid", "affected_regions": ["Chile"], "evidence_ids": ["C2"]},
        ]

        report = duplicate_alert_report(alerts)

        self.assertEqual(report["duplicate_alert_count"], 1)
        self.assertAlmostEqual(report["duplicate_alert_rate"], 1 / 3, places=4)

    def test_alert_quality_requires_required_fields(self):
        duplicate = {"duplicate_alert_rate": 0.0}
        quality = alert_quality_report(
            [
                {
                    "alert_id": "alert_001",
                    "severity": "high",
                    "confidence": "medium",
                    "issue": "grid",
                    "evidence_trigger": "Regulator evidence",
                    "evidence_ids": ["C1"],
                    "uncertainty": "Needs review",
                    "recommended_follow_up": "Check official filings",
                }
            ],
            duplicate,
        )
        broken = alert_quality_report([{"alert_id": "alert_001", "issue": "grid"}], duplicate)

        self.assertEqual(quality["status"], "pass")
        self.assertEqual(broken["status"], "fail")


def brief_report():
    return {
        "project_id": "ai_infra_geo_risk",
        "mode": "unit",
        "policy": "rag",
        "question": "AI data center risk",
        "red_team": {"review_issue_count": 0, "objections": []},
        "agent_outputs": [
            {
                "claims": [
                    {
                        "claim": "Energy and Water Agent flags grid based on regulator evidence.",
                        "evidence_ids": ["C1"],
                        "issue_categories": ["grid"],
                        "confidence": "medium",
                    }
                ]
            }
        ],
        "brief": {
            "next_signals_to_watch": ["grid official filings"],
            "source_gaps": [
                {
                    "gap": "Limited Portuguese-language local-source coverage.",
                    "suggested_next_query": "Portuguese local data center water risk",
                }
            ],
            "citation_appendix": [
                {
                    "evidence_id": "C1",
                    "source_name": "Chile Energy Regulator",
                    "title": "Grid Note",
                    "jurisdiction": "Chile",
                    "text": "Large data center electricity requests can increase grid pressure.",
                }
            ],
        },
    }


if __name__ == "__main__":
    unittest.main()
