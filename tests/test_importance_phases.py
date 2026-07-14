import unittest

from canon.eval.anchors import build_anchor_report
from canon.eval.security import evaluate_adversarial_rag_security
from canon.evidence.signals import evidence_decision_from_result, evidence_signals_from_result


class ImportancePhaseTests(unittest.TestCase):
    def test_typed_signals_wrap_retrieval_result(self):
        result = {
            "chunk_id": "c1",
            "work_id": "w1",
            "year": 2024,
            "source_name": "Journal",
            "cluster_id": 1,
            "components": {
                "relevance": 0.9,
                "semantic_similarity": 0.8,
                "source_quality": 0.6,
                "method_signal": 0.4,
            },
            "decision": {
                "decision": "allow",
                "score": 0.75,
                "safety_decision": "allow",
                "safety_risk": 0.0,
                "stage": {
                    "signals": {
                        "query_text_relevance": {"candidate_rank": 1, "lexical_relevance": 0.9},
                        "evidence_role": {"method_signal": 0.4},
                        "source_trust": {"source_trust": 0.6},
                        "provenance": {"provenance": "academic"},
                    },
                    "safety_gate": {"safety": {"decision": "allow", "risk_score": 0.0}},
                },
            },
        }
        signals = evidence_signals_from_result(result)
        decision = evidence_decision_from_result(result)
        self.assertEqual(signals.query_text_relevance["candidate_rank"], 1)
        self.assertEqual(signals.evidence_role["method_signal"], 0.4)
        self.assertIn("generator_context", decision.allowed_uses)

    def test_adversarial_rag_security_benchmark_passes(self):
        report = evaluate_adversarial_rag_security()
        self.assertEqual(report["status"], "pass")
        self.assertTrue(report["checks"]["attacks_retrieved_as_candidates"])
        self.assertTrue(report["checks"]["no_attack_exposed_to_context"])
        self.assertGreaterEqual(report["metrics"]["benign_context_count"], 1)

    def test_anchor_registry_blocks_overclaiming(self):
        report = build_anchor_report()
        self.assertGreaterEqual(report["missing_or_limited_anchor_count"], 1)
        self.assertIn(
            "Do not claim validated source trust until human/expert trust labels exist.",
            report["claim_limits"],
        )


if __name__ == "__main__":
    unittest.main()
