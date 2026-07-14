import unittest

from canon.evidence.corroboration import assess_corroboration
from canon.eval.corroboration import (
    echo_attack_fixture,
    evaluate_adversarial_corroboration,
)
from canon.synthesis.answer import support_assessment


class CorroborationTests(unittest.TestCase):
    def test_echo_attack_is_not_independent_corroboration(self):
        report = assess_corroboration(echo_attack_fixture())
        self.assertEqual(report["summary"]["overall_verdict"], "adversarial_echo_risk")
        self.assertLessEqual(report["summary"]["max_independent_support"], 1)
        self.assertGreater(report["summary"]["max_echo_risk"], 0.45)

    def test_independent_sources_raise_independent_support(self):
        evidence = [
            item("C1", "w1", "Journal A", "Study A", "The policy reduced costs.", "supportive", "a"),
            item("C2", "w2", "Journal B", "Study B", "The policy reduced costs.", "supportive", "b"),
        ]
        report = assess_corroboration(evidence)
        self.assertEqual(report["summary"]["overall_verdict"], "independently_corroborated")
        self.assertEqual(report["summary"]["max_independent_support"], 2)

    def test_support_assessment_downgrades_echo_risk(self):
        evidence = echo_attack_fixture()
        corroboration = assess_corroboration(evidence)
        results = [
            {
                "components": {
                    "relevance": 1.0,
                    "semantic_similarity": 1.0,
                }
            }
            for _item in evidence
        ]
        support = support_assessment(
            "Does the product have churn risk in enterprise accounts?",
            evidence,
            results,
            corroboration,
        )
        self.assertLessEqual(support["support_confidence"], 0.42)
        self.assertEqual(support["corroboration"]["overall_verdict"], "adversarial_echo_risk")

    def test_adversarial_corroboration_eval_passes(self):
        report = evaluate_adversarial_corroboration()
        self.assertEqual(report["status"], "pass")
        self.assertTrue(report["checks"]["echo_attack_flagged"])


def item(citation_id, work_id, source_name, title, claim_text, stance, chain):
    return {
        "citation_id": citation_id,
        "chunk_id": f"chunk:{work_id}",
        "work_id": work_id,
        "title": title,
        "source_name": source_name,
        "source_chain_id": chain,
        "claim": {"text": claim_text, "stance": stance, "confidence": 0.9},
        "preview": claim_text,
    }


if __name__ == "__main__":
    unittest.main()
