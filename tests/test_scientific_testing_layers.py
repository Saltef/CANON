import unittest

from canon.eval.faithfulness import score_sentence
from canon.eval.perturbations import query_variants
from canon.eval.qrels import canonical_from_trec, validate_qrels
from canon.eval.regression_gate import run_regression_gate
from canon.eval.uncertainty import bootstrap_report


class ScientificTestingLayerTests(unittest.TestCase):
    def test_trec_qrels_convert_to_canonical_shape(self):
        qrels = canonical_from_trec(
            ["q1 0 doc-a 2", "q1 0 doc-b 1"],
            benchmark_id="toy",
            queries={"q1": "toy query"},
        )
        self.assertEqual(qrels["queries"][0]["query"], "toy query")
        self.assertEqual(qrels["queries"][0]["relevant"]["doc-a"], 2.0)
        self.assertEqual(validate_qrels(qrels), [])

    def test_validate_qrels_reports_missing_relevance(self):
        issues = validate_qrels({"benchmark_id": "bad", "queries": [{"id": "q1", "query": "x"}]})
        self.assertEqual(issues[0]["id"], "missing_relevance")

    def test_bootstrap_report_tracks_winner_probability(self):
        external = {
            "benchmark_id": "toy",
            "mode": "m",
            "method_set_id": "methods",
            "method_summary": {"a": {}, "b": {}},
            "queries": [
                {
                    "method_runs": [
                        {"method_id": "a", "metrics": {"nDCG@10": 1.0}},
                        {"method_id": "b", "metrics": {"nDCG@10": 0.0}},
                    ]
                }
            ],
        }
        report = bootstrap_report(external, metric="nDCG@10", samples=20, seed=1)
        self.assertEqual(report["topline"]["winner"], "a")
        self.assertEqual(report["intervals"]["a"]["winner_probability"], 1.0)

    def test_score_sentence_requires_valid_cited_evidence(self):
        citations = {"C1": {"citation_id": "C1"}}
        evidence = {
            "C1": {
                "preview": "Democratic dyads rarely fight wars in the corpus.",
                "title": "Democratic Peace",
                "claim": {"text": "Democratic dyads rarely fight wars."},
            }
        }
        scored = score_sentence("C1 says democratic dyads rarely fight wars.", ["C1"], citations, evidence)
        self.assertEqual(scored["citation_validity"], 1.0)
        self.assertEqual(scored["claim_backed"], 1.0)

    def test_query_variants_are_distinct(self):
        variants = query_variants("How effective are economic sanctions at changing state behavior?")
        self.assertGreaterEqual(len(variants), 3)
        self.assertTrue(all(variant["query"] for variant in variants))


if __name__ == "__main__":
    unittest.main()
