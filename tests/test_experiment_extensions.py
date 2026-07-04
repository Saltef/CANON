import tempfile
import unittest
from pathlib import Path

from canon.embeddings.pgvector import quote, render_pgvector_sql
from canon.eval.providers import parse_providers
from canon.eval.tuning import WEIGHT_VARIANTS
from canon.experiments.manifest import artifact_record
from canon.fulltext.grobid import safe_name
from canon.reports.dashboard import table
from canon.reports.scientific_audit import audit_status, build_checks, build_warnings, render_markdown


class ExperimentExtensionTests(unittest.TestCase):
    def test_parse_providers_defaults_and_splits(self):
        self.assertIn("local", parse_providers(None))
        self.assertEqual(parse_providers("local,cohere"), ["local", "cohere"])

    def test_pgvector_sql_renders_records(self):
        sql = render_pgvector_sql(
            [
                {
                    "chunk_id": "c1",
                    "work_id": "w1",
                    "provider": "local",
                    "model": "m",
                    "vector": [0.1, 0.2],
                }
            ],
            2,
        )
        self.assertIn("vector(2)", sql)
        self.assertIn("INSERT INTO chunk_embeddings", sql)

    def test_quote_escapes_single_quotes(self):
        self.assertEqual(quote("a'b"), "'a''b'")

    def test_safe_name_removes_path_characters(self):
        self.assertNotIn("/", safe_name("https://example.org/W1"))

    def test_dashboard_table_escapes_values(self):
        html = table(["name"], [{"name": "<script>"}])
        self.assertIn("&lt;script&gt;", html)

    def test_weight_variants_include_support_and_conflict(self):
        self.assertIn("support_heavy", WEIGHT_VARIANTS)
        self.assertIn("conflict_heavy", WEIGHT_VARIANTS)

    def test_artifact_record_hashes_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "artifact.json"
            path.write_text("{}", encoding="utf-8")
            record = artifact_record(path, root)
            self.assertEqual(record["path"], "artifact.json")
            self.assertEqual(record["bytes"], 2)

    def test_scientific_audit_checks_core_artifacts(self):
        artifacts = {
            "method_eval": {},
            "evaluation_suite": {"query_count": 25, "leaderboard": [1, 2, 3, 4, 5]},
            "batch_eval": {"batches": [1, 2, 3, 4]},
            "probe_eval": {"failure_summary": {"failure_count": 0}},
            "qrels_validation": {"status": "pass"},
            "public_qrels_validation": {"status": "pass"},
            "external_ir": {"query_count": 5},
            "bootstrap_ir": {},
            "paired_significance": {"comparison_count": 10},
            "faithfulness": {"query_count": 5},
            "perturbation_eval": {"query_count": 8},
            "data_card": {"work_count": 25},
            "corpus_expansion": {"profile_id": "social_science_product_v1"},
            "weight_tuning": {"variant_count": 5},
            "claim_decision": {"status": "pass"},
            "product_readiness": {"status": "pass"},
            "regression_gate": {"status": "pass"},
            "diversity_gate": {"status": "pass"},
            "dashboard": {},
        }
        checks = build_checks(artifacts)
        self.assertEqual(audit_status(checks), "pass")
        self.assertTrue(any(item["id"] == "diversity_gate==pass" for item in checks))

    def test_scientific_audit_warns_on_unstable_batches(self):
        warnings = build_warnings(
            {
                "batch_eval": {
                    "stability": {
                        "stable_top_method": None,
                        "minimum_top2_margin": 0.001,
                    }
                },
                "external_ir": {"benchmark_kind": "internal_qrels", "benchmark_id": "b", "query_count": 5},
                "data_card": {"limitations": ["small corpus"]},
                "corpus_expansion": {"readiness": {"status": "needs_expansion"}},
                "diversity_gate": {"status": "fail", "checks": [{"id": "useful_breadth_count"}]},
            }
        )
        self.assertEqual(
            {item["id"] for item in warnings},
            {
                "unstable_top_method",
                "narrow_top2_margin",
                "internal_qrels_only",
                "data_card_limitations",
                "corpus_expansion_needed",
                "diversity_gate_failed",
            },
        )

    def test_scientific_audit_markdown_includes_references(self):
        markdown = render_markdown(
            {
                "mode": "m",
                "status": "pass",
                "checks": [{"id": "x", "passed": True, "message": "ok"}],
                "warnings": [],
                "topline": {},
                "recommended_external_references": [
                    {"name": "BEIR", "url": "https://example.org", "project_relevance": "retrieval"}
                ],
                "next_scientific_steps": ["Add qrels."],
            }
        )
        self.assertIn("PASS", markdown)
        self.assertIn("[BEIR]", markdown)


if __name__ == "__main__":
    unittest.main()
