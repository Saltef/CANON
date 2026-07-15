import unittest
from unittest.mock import patch

from canon.product import service


class ProductServiceTests(unittest.TestCase):
    def test_optional_int_validates_positive_values(self):
        self.assertEqual(service.optional_int("5"), 5)
        with self.assertRaises(service.ProductError):
            service.optional_int("0")

    def test_require_text_rejects_empty_queries(self):
        with self.assertRaises(service.ProductError):
            service.require_text({"query": " "}, "query")

    def test_answer_returns_compact_evidence_with_explanations(self):
        with patch("canon.product.service.safe_claim_decision", return_value={"global_winner_claim": {}}):
            with patch("canon.product.service.build_query_diagnostics", return_value={"stability": {"status": "stable"}}):
                with patch("canon.product.service.synthesize") as synthesize:
                    synthesize.return_value = {
                        "query": "democratic peace",
                        "mode": "m",
                        "policy": "rag",
                        "top_k": 5,
                        "answer": "Answer.",
                        "citations": [{"citation_id": "C1"}],
                        "evidence": [
                            {
                                "citation_id": "C1",
                                "chunk_id": "c1",
                                "work_id": "w1",
                                "title": "A",
                                "source_name": "J",
                                "year": 2020,
                                "rank": 1,
                                "cluster_id": 1,
                                "final_score": 0.7,
                                "claim": {"id": "claim-a"},
                                "preview": "preview",
                                "explanation": {"reasons": ["high_lexical_relevance"]},
                            }
                        ],
                        "support_assessment": {"support_level": "strong"},
                        "limitations": [],
                        "conflict_notes": [],
                    }
                    report = service.answer({"query": "democratic peace", "mode": "m", "policy": "rag"})
        self.assertEqual(report["evidence"][0]["explanation"]["reasons"], ["high_lexical_relevance"])
        self.assertEqual(report["evidence"][0]["claim"]["id"], "claim-a")
        self.assertEqual(report["query_diagnostics"]["stability"]["status"], "stable")

    def test_query_diagnostics_endpoint_validates_and_returns_report(self):
        with patch("canon.product.service.build_query_diagnostics", return_value={"original_query": "q"}):
            report = service.query_diagnostics({"query": "q", "candidate_k": 5, "freedom_level": "strict"})
        self.assertEqual(report["original_query"], "q")

    def test_start_project_builds_scoped_config(self):
        report = service.start_project(
            {
                "project_name": "AI Infrastructure Geopolitical Risk",
                "domain": "AI infrastructure and geopolitical risk",
                "regions": ["Latin America"],
                "languages": "English,Spanish",
                "issue_categories": ["energy demand"],
                "desired_report_types": ["weekly_intelligence_brief"],
                "source_boundaries": ["G:/My Drive/CANON Corpus"],
                "write_report": "false",
            }
        )

        self.assertEqual(report["status"], "ready")
        self.assertTrue(report["source_plan"]["monitor_ready"])
        self.assertEqual(report["security_boundary"]["credentials_stored"], False)

    def test_start_project_requires_scope_fields(self):
        with self.assertRaises(service.ProductError):
            service.start_project(
                {
                    "project_name": "Missing fields",
                    "domain": "AI infrastructure",
                    "regions": [],
                    "languages": ["English"],
                    "issue_categories": ["energy demand"],
                    "desired_report_types": ["weekly_intelligence_brief"],
                }
            )

    def test_evidence_packets_wraps_synthesis_in_integration_contract(self):
        with patch("canon.product.service.build_query_diagnostics") as diagnostics:
            with patch("canon.product.service.synthesize") as synthesize:
                diagnostics.return_value = {
                    "query_to_corpus": {"matched_terms": ["water"], "weak_terms": ["permitting"]},
                    "result_neighborhood": {"field_phrases": ["grid stress"]},
                    "query_variants": [{"query": "water grid stress"}],
                    "stability": {"status": "needs_caution"},
                }
                synthesize.return_value = {
                    "query": "AI data center water risk",
                    "mode": "m",
                    "policy": "rag",
                    "top_k": 5,
                    "answer": "Answer.",
                    "citations": [{"citation_id": "C1"}],
                    "evidence": [
                        {
                            "citation_id": "C1",
                            "chunk_id": "chunk:1",
                            "work_id": "doc:1",
                            "title": "Grid Stress",
                            "source_name": "Local News",
                            "year": 2026,
                            "rank": 1,
                            "cluster_id": "energy",
                            "preview": "Water and grid stress are debated.",
                            "claim": {"id": "claim:1", "text": "Data centers can stress water supply."},
                        }
                    ],
                    "support_assessment": {"support_level": "moderate", "support_confidence": 0.51},
                    "limitations": ["Human review is still required."],
                    "conflict_notes": [{"id": "conflict:1"}],
                }

                report = service.evidence_packets(
                    {
                        "request_id": "req_001",
                        "project_id": "ai_infra_geo_risk",
                        "question": "AI data center water risk",
                        "mode": "m",
                        "research_frame": {
                            "subdomains": ["water", "energy"],
                            "regions": ["Latin America"],
                            "languages": ["English", "Spanish"],
                        },
                        "evidence_requirements": {
                            "top_k": 5,
                            "minimum_source_types": ["official", "local_media"],
                        },
                    }
                )

        packet = report["evidence_packets"][0]
        self.assertEqual(report["request_id"], "req_001")
        self.assertEqual(packet["support_level"], "mixed")
        self.assertEqual(packet["confidence"], "medium")
        self.assertEqual(packet["issue_categories"], ["water", "energy"])
        self.assertEqual(packet["supporting_evidence"][0]["evidence_id"], "C1")
        self.assertEqual(packet["conflicting_evidence"][0]["id"], "conflict:1")
        self.assertEqual(report["query_diagnostics"]["weak_terms"], ["permitting"])
        self.assertEqual(report["retrieval_metrics"]["source_diversity_status"], "pass")
        self.assertTrue(report["frame_coverage"]["human_review_required"])

    def test_frame_coverage_reports_requested_missing_dimensions(self):
        report = service.analyze_frame_coverage(
            research_frame={
                "subdomains": ["water and cooling", "sovereign AI"],
                "regions": ["Brazil", "Chile"],
                "languages": ["English", "Portuguese"],
                "representation_goals": ["language diversity"],
            },
            requirements={"minimum_source_types": ["official", "local_media"]},
            evidence=[
                {
                    "title": "Brazil data center water risk",
                    "preview": "Water and cooling concerns and local opposition around a data center.",
                    "source_name": "Local News",
                    "source_type": "local_media",
                    "language": "pt",
                    "jurisdiction": "Brazil",
                }
            ],
            diagnostics={"query_to_corpus": {"weak_terms": ["sovereign"]}},
        )

        self.assertEqual(report["status"], "partial")
        self.assertIn("water and cooling", report["coverage"]["subdomains"]["covered"])
        self.assertIn("sovereign AI", report["coverage"]["subdomains"]["missing"])
        self.assertIn("Chile", report["coverage"]["regions"]["missing"])
        self.assertIn("English", report["coverage"]["languages"]["missing"])
        self.assertIn("official", report["coverage"]["source_types"]["missing"])
        self.assertTrue(any(row["field"] == "query_terms" for row in report["diagnostics"]))

    def test_frame_coverage_report_focuses_packet_output(self):
        with patch("canon.product.service.evidence_packets") as packets:
            packets.return_value = {
                "request_id": "req_1",
                "project_id": "ai_infra_geo_risk",
                "query": "AI data center grid risk",
                "mode": "m",
                "policy": "rag",
                "research_frame": {"subdomains": ["energy"]},
                "frame_coverage": {
                    "status": "partial",
                    "diagnostics": [
                        {
                            "field": "regions",
                            "status": "missing",
                            "suggested_next_query": "Chile Brazil",
                        }
                    ],
                },
                "coverage_gaps": [{"gap": "Missing regions.", "suggested_next_query": "Chile Brazil"}],
                "query_diagnostics": {"weak_terms": ["grid"]},
                "retrieval_metrics": {"estimated_confidence": "medium"},
            }

            report = service.frame_coverage_report({"query": "AI data center grid risk", "write_report": "false"})

        self.assertEqual(report["report_id"], "research_frame_coverage_v1")
        self.assertEqual(report["status"], "coverage_gap_human_review_required")
        self.assertTrue(report["human_review_required"])
        self.assertEqual(report["next_actions"], ["Chile Brazil"])
        self.assertIn("human_review_boundary", report)
        packets.assert_called_once()

    def test_packet_evidence_item_exposes_metadata_for_downstream_agents(self):
        item = service.packet_evidence_item(
            {
                "citation_id": "C1",
                "chunk_id": "chunk:1",
                "work_id": "doc:1",
                "title": "Title",
                "source_name": "Regulator",
                "url": "https://example.test",
                "year": 2026,
                "rank": 1,
                "cluster_id": "energy",
                "preview": "Text",
                "language": "en",
                "source_type": "official_report",
                "provenance": "official",
                "domain": "ai_infra_geo_risk",
                "jurisdiction": "Chile",
            }
        )

        self.assertEqual(item["language"], "en")
        self.assertEqual(item["source_type"], "official_report")
        self.assertEqual(item["provenance"], "official")
        self.assertEqual(item["domain"], "ai_infra_geo_risk")
        self.assertEqual(item["jurisdiction"], "Chile")

    def test_evidence_packets_rejects_non_object_requirements(self):
        with self.assertRaises(service.ProductError):
            service.evidence_packets({"query": "q", "evidence_requirements": "top 10"})

    def test_intelligence_brief_wraps_runner(self):
        with patch("canon.intelligence.evidence_runner.run_intelligence_brief", return_value={"status": "ready_for_human_review"}) as runner:
            report = service.intelligence_brief(
                {
                    "query": "AI data center risk",
                    "mode": "m",
                    "policy": "rag",
                    "project_id": "ai_infra_geo_risk",
                    "write_report": "false",
                }
            )

        self.assertEqual(report["status"], "ready_for_human_review")
        runner.assert_called_once()
        self.assertFalse(runner.call_args.kwargs["write_report"])

    def test_report_quality_gate_compacts_brief_quality_signals(self):
        with patch("canon.product.service.intelligence_brief") as brief:
            brief.return_value = {
                "report_id": "mock_evidence_intelligence_brief_v1",
                "project_id": "ai_infra_geo_risk",
                "mode": "m",
                "policy": "rag",
                "question": "AI data center risk",
                "status": "ready_for_human_review",
                "report_quality": {
                    "status": "pass",
                    "checks": [{"id": "grounded_claim_ratio", "status": "pass"}],
                },
                "grounding_report": {
                    "status": "pass",
                    "claim_count": 3,
                    "supported_claim_count": 3,
                    "unsupported_claim_count": 0,
                    "grounded_claim_ratio": 1.0,
                    "unsupported_claims": [],
                },
                "duplicate_agent_output": {"duplicate_agent_output_rate": 0.0},
                "red_team": {"status": "pass_with_review", "blocking_issue_count": 0, "review_issue_count": 1, "objections": []},
            }

            report = service.report_quality_gate({"query": "AI data center risk", "write_report": "false"})

        self.assertEqual(report["report_id"], "intelligence_report_quality_gate_v1")
        self.assertEqual(report["status"], "pass_human_review_required")
        self.assertEqual(report["grounding"]["grounded_claim_ratio"], 1.0)
        self.assertTrue(report["human_review_required"])
        self.assertIn("human_review_boundary", report)
        brief.assert_called_once()

    def test_report_quality_gate_blocks_unsupported_claims(self):
        with patch("canon.product.service.intelligence_brief") as brief:
            brief.return_value = {
                "status": "blocked",
                "report_quality": {"status": "fail", "checks": [{"id": "grounding_pass", "status": "fail"}]},
                "grounding_report": {
                    "status": "fail",
                    "unsupported_claims": [{"claim_id": "c1"}],
                },
                "red_team": {
                    "blocking_issue_count": 1,
                    "objections": [{"severity": "blocker", "issue": "unsupported_claims"}],
                },
            }

            report = service.report_quality_gate({"query": "q", "write_report": "false"})

        self.assertEqual(report["status"], "blocked_grounding")
        self.assertTrue(any("Resolve or remove claims" in action for action in report["next_actions"]))

    def test_alert_digest_wraps_runner(self):
        with patch("canon.intelligence.alerts.run_alert_digest", return_value={"status": "ready_for_human_review"}) as runner:
            report = service.alert_digest(
                {
                    "query": "AI data center risk",
                    "mode": "m",
                    "policy": "rag",
                    "project_id": "ai_infra_geo_risk",
                    "write_report": "false",
                }
            )

        self.assertEqual(report["status"], "ready_for_human_review")
        runner.assert_called_once()
        self.assertEqual(runner.call_args.kwargs["question"], "AI data center risk")
        self.assertEqual(runner.call_args.kwargs["mode"], "m")
        self.assertFalse(runner.call_args.kwargs["write_report"])

    def test_flagship_handoff_wraps_runner_with_defaults(self):
        with patch(
            "canon.product.flagship_handoff.run_flagship_handoff",
            return_value={"status": "automated_pass_human_review_required"},
        ) as runner:
            report = service.flagship_handoff({"write_report": "false"})

        self.assertEqual(report["status"], "automated_pass_human_review_required")
        runner.assert_called_once()
        self.assertEqual(runner.call_args.kwargs["mode"], "ai_infra_geo_risk_demo")
        self.assertIn("AI data center expansion", runner.call_args.kwargs["question"])
        self.assertFalse(runner.call_args.kwargs["write_report"])

    def test_flagship_handoff_accepts_query_and_fixture_alias(self):
        with patch(
            "canon.product.flagship_handoff.run_flagship_handoff",
            return_value={"status": "automated_pass_human_review_required"},
        ) as runner:
            service.flagship_handoff(
                {
                    "query": "custom question",
                    "mode": "custom_mode",
                    "fixture": "data/fixtures/ai_infra_geo_risk_sample.jsonl",
                    "queries_path": "gold/ai_infra_geo_risk_seed_queries.json",
                    "write_report": "false",
                }
            )

        self.assertEqual(runner.call_args.kwargs["question"], "custom question")
        self.assertEqual(runner.call_args.kwargs["mode"], "custom_mode")
        self.assertEqual(str(runner.call_args.kwargs["fixture_path"]).replace("\\", "/"), "data/fixtures/ai_infra_geo_risk_sample.jsonl")
        self.assertEqual(str(runner.call_args.kwargs["queries_path"]).replace("\\", "/"), "gold/ai_infra_geo_risk_seed_queries.json")

    def test_acceptance_scenario_wraps_runner_with_defaults(self):
        with patch(
            "canon.product.acceptance_scenario.run_acceptance_scenario",
            return_value={"status": "automated_pass_human_review_required"},
        ) as runner:
            report = service.acceptance_scenario({"write_report": "false"})

        self.assertEqual(report["status"], "automated_pass_human_review_required")
        runner.assert_called_once()
        self.assertEqual(runner.call_args.kwargs["mode"], "ai_infra_geo_risk_demo")
        self.assertFalse(runner.call_args.kwargs["write_report"])

    def test_intelligence_review_prepare_wraps_review_tasks(self):
        with patch("canon.product.intelligence_review.build_review_tasks", return_value={"report_id": "intelligence_brief_review_tasks_v1"}) as builder:
            report = service.intelligence_review_prepare(
                {
                    "mode": "m",
                    "queries_path": "gold/ai_infra_geo_risk_seed_queries.json",
                    "records_path": "reports/review.json",
                    "reset_review_labels": "true",
                    "write_report": "false",
                }
            )

        self.assertEqual(report["report_id"], "intelligence_brief_review_tasks_v1")
        builder.assert_called_once()
        self.assertEqual(builder.call_args.kwargs["mode"], "m")
        self.assertFalse(builder.call_args.kwargs["preserve_existing_reviews"])
        self.assertFalse(builder.call_args.kwargs["write_report"])

    def test_intelligence_review_status_uses_mode_default_records_path(self):
        with patch("canon.product.intelligence_review.build_review_status_report", return_value={"status": "incomplete"}) as status:
            report = service.intelligence_review_status({"mode": "m", "write_report": "false"})

        self.assertEqual(report["status"], "incomplete")
        self.assertTrue(str(status.call_args.args[0]).replace("\\", "/").endswith("reports/intelligence_brief_review_tasks_m.json"))
        self.assertFalse(status.call_args.kwargs["write_report"])

    def test_intelligence_review_feedback_uses_mode_default_records_path(self):
        with patch("canon.product.intelligence_review.build_feedback_report", return_value={"status": "ready"}) as feedback:
            report = service.intelligence_review_feedback({"mode": "m", "write_report": "false"})

        self.assertEqual(report["status"], "ready")
        self.assertTrue(str(feedback.call_args.args[0]).replace("\\", "/").endswith("reports/intelligence_brief_review_tasks_m.json"))
        self.assertFalse(feedback.call_args.kwargs["write_report"])

    def test_intelligence_review_export_csv_returns_written_path(self):
        with patch("canon.product.intelligence_review.export_review_csv", return_value=service.Path("reports/review.csv")) as exporter:
            report = service.intelligence_review_export_csv(
                {
                    "records_path": "reports/review.json",
                    "output_path": "reports/review.csv",
                }
            )

        self.assertEqual(report["status"], "review_csv_written")
        self.assertEqual(report["output_path"], "reports/review.csv")
        exporter.assert_called_once()

    def test_intelligence_review_import_csv_requires_csv_path(self):
        with self.assertRaises(service.ProductError):
            service.intelligence_review_import_csv({"records_path": "reports/review.json"})

    def test_intelligence_review_import_csv_wraps_importer(self):
        with patch("canon.product.intelligence_review.import_review_csv", return_value={"status": "imported"}) as importer:
            report = service.intelligence_review_import_csv(
                {
                    "records_path": "reports/review.json",
                    "csv_path": "reports/review.csv",
                    "output_path": "reports/completed.json",
                }
            )

        self.assertEqual(report["status"], "imported")
        importer.assert_called_once()
        self.assertEqual(str(importer.call_args.args[0]).replace("\\", "/"), "reports/review.json")
        self.assertEqual(str(importer.call_args.args[1]).replace("\\", "/"), "reports/review.csv")

    def test_intelligence_brief_evaluation_wraps_eval_gate(self):
        with patch("canon.eval.intelligence_brief.evaluate_intelligence_briefs", return_value={"status": "pass"}) as evaluator:
            report = service.intelligence_brief_evaluation(
                {
                    "mode": "m",
                    "queries_path": "gold/ai_infra_geo_risk_seed_queries.json",
                    "policy": "rag",
                    "project_id": "ai_infra_geo_risk",
                    "write_report": "false",
                }
            )

        self.assertEqual(report["status"], "pass")
        evaluator.assert_called_once()
        self.assertEqual(evaluator.call_args.kwargs["mode"], "m")
        self.assertFalse(evaluator.call_args.kwargs["write_report"])

    def test_alert_digest_evaluation_wraps_eval_gate(self):
        with patch("canon.eval.alert_digest.evaluate_alert_digests", return_value={"status": "pass"}) as evaluator:
            report = service.alert_digest_evaluation(
                {
                    "mode": "m",
                    "queries_path": "gold/ai_infra_geo_risk_seed_queries.json",
                    "policy": "rag",
                    "project_id": "ai_infra_geo_risk",
                    "write_report": "false",
                }
            )

        self.assertEqual(report["status"], "pass")
        evaluator.assert_called_once()
        self.assertEqual(evaluator.call_args.kwargs["mode"], "m")
        self.assertFalse(evaluator.call_args.kwargs["write_report"])

    def test_invalid_freedom_level_raises_product_error(self):
        with self.assertRaises(service.ProductError):
            service.query_diagnostics({"query": "q", "freedom_level": "wild"})

    def test_source_profile_wraps_flexible_profiler(self):
        with patch("canon.product.service.profile_source", return_value={"source_shape": "crm_record"}) as profiler:
            report = service.source_profile({"input_path": "data/my_docs", "sample_size": "5"})

        profiler.assert_called_once()
        self.assertEqual(report["source_shape"], "crm_record")
        self.assertEqual(profiler.call_args.kwargs["sample_size"], 5)

    def test_source_profile_missing_path_raises_404(self):
        with patch("canon.product.service.profile_source", side_effect=FileNotFoundError("missing")):
            with self.assertRaises(service.ProductError) as context:
                service.source_profile({"input_path": "missing"})
        self.assertEqual(context.exception.status_code, 404)

    def test_source_ingest_wraps_flexible_ingest(self):
        with patch("canon.product.service.ingest_flexible_source", return_value={"mode": "m"}) as ingest:
            report = service.source_ingest(
                {
                    "input_path": "data/my_docs",
                    "mode": "m",
                    "domain": "energy",
                    "chunk_tokens": "50",
                    "overlap_tokens": "0",
                }
            )

        ingest.assert_called_once()
        self.assertEqual(report["mode"], "m")
        self.assertEqual(ingest.call_args.kwargs["chunk_tokens"], 50)
        self.assertEqual(ingest.call_args.kwargs["overlap_tokens"], 0)

    def test_corpus_build_requires_source_modes_and_wraps_phase16(self):
        with patch(
            "canon.product.service.run_phase16",
            return_value={"corpus": {"corpus_id": "c"}, "validation": {"status": "corpus_only"}},
        ) as build:
            report = service.corpus_build({"corpus_id": "c", "from_modes": "m1,m2", "corpus_only": "false"})

        self.assertEqual(report["corpus"]["corpus_id"], "c")
        self.assertFalse(build.call_args.kwargs["corpus_only"])
        self.assertEqual(build.call_args.kwargs["from_modes"], ["m1", "m2"])
        with self.assertRaises(service.ProductError):
            service.corpus_build({"corpus_id": "c"})

    def test_model_evaluation_requires_labels_and_wraps_eval(self):
        with patch("canon.product.service.evaluate_semantic_models", return_value={"report_id": "semantic_model_evaluation_v1"}) as evaluate:
            report = service.model_evaluation(
                {
                    "mode": "m",
                    "qrels_path": "gold/qrels.json",
                    "providers": ["local", "openai"],
                    "k": "5",
                }
            )

        self.assertEqual(report["report_id"], "semantic_model_evaluation_v1")
        self.assertEqual(evaluate.call_args.kwargs["providers"], ["local", "openai"])
        self.assertEqual(evaluate.call_args.kwargs["k"], 5)
        with self.assertRaises(service.ProductError):
            service.model_evaluation({"mode": "m"})

    def test_prehuman_check_wraps_release_boundary_gate(self):
        with patch("canon.product.prehuman_check.run_prehuman_check", return_value={"status": "automated_pass_human_review_required"}) as runner:
            report = service.prehuman_check(
                {
                    "mode": "m",
                    "benchmark_id": "llm_judged_m",
                    "model_providers": ["local"],
                    "rerankers": "heuristic",
                    "top_k": "5",
                    "candidate_k": "10",
                    "write_report": "false",
                }
            )

        self.assertEqual(report["status"], "automated_pass_human_review_required")
        runner.assert_called_once()
        self.assertEqual(runner.call_args.kwargs["mode"], "m")
        self.assertEqual(runner.call_args.kwargs["model_providers"], ["local"])
        self.assertEqual(runner.call_args.kwargs["rerankers"], ["heuristic"])
        self.assertFalse(runner.call_args.kwargs["write_report"])

    def test_optional_bool_parses_strings(self):
        self.assertTrue(service.optional_bool("true"))
        self.assertFalse(service.optional_bool("false"))
        with self.assertRaises(service.ProductError):
            service.optional_bool("maybe")

    def test_compact_answer_evidence_caps_long_lists(self):
        evidence = [
            {
                "citation_id": f"C{index}",
                "chunk_id": f"c{index}",
                "work_id": f"w{index}",
                "title": "A",
                "source_name": "J",
                "year": 2020,
                "rank": index,
                "cluster_id": 1,
                "final_score": 0.7,
                "explanation": {"reasons": ["reason"]},
            }
            for index in range(12)
        ]
        self.assertEqual(len(service.compact_answer_evidence(evidence)), 10)

    def test_product_summary_has_claim_boundaries(self):
        summary = service.product_summary("social_science_ir_v1_harvest10")
        self.assertEqual(summary["product"], "CANON Evidence Workbench")
        self.assertIn("claim_boundaries", summary)

    def test_unknown_report_raises_404(self):
        with self.assertRaises(service.ProductError) as context:
            service.report("missing", "social_science_ir_v1_harvest10")
        self.assertEqual(context.exception.status_code, 404)

    def test_diversity_gate_report_is_allowed(self):
        with patch("canon.product.service.load_report", return_value={"status": "pass"}):
            report = service.report("diversity-gate", "social_science_ir_10k")
        self.assertEqual(report["status"], "pass")

    def test_diversity_diagnostics_report_is_allowed(self):
        with patch("canon.product.service.load_report", return_value={"headline": {"finding": "conditional"}}):
            report = service.report("diversity-diagnostics", "social_science_ir_10k")
        self.assertEqual(report["headline"]["finding"], "conditional")

    def test_compare_retrieval_preserves_result_explanations(self):
        with patch("canon.product.service.compare") as compare:
            compare.return_value = {
                "query": "democratic peace",
                "mode": "m",
                "policies": ["rag"],
                "rank_table": [],
                "runs": [
                    {
                        "policy": "rag",
                        "top_k": 1,
                        "results": [
                            {
                                "chunk_id": "c1",
                                "explanation": {"reasons": ["high_lexical_relevance"]},
                            }
                        ],
                    }
                ],
            }
            report = service.compare_retrieval({"query": "democratic peace", "mode": "m", "policies": ["rag"]})
        self.assertEqual(report["runs"][0]["results"][0]["explanation"]["reasons"], ["high_lexical_relevance"])

    def test_compact_diversity_report_keeps_product_fields(self):
        compact = service.compact_diversity_report(
            {
                "mode": "m",
                "method_set_id": "methods",
                "diverse_method_id": "diverse",
                "baseline_method_id": "lexical",
                "query_count": 25,
                "aggregate": {"verdict_counts": {"useful_breadth": 10}},
                "queries": [{"large": "payload"}],
            }
        )
        self.assertNotIn("queries", compact)
        self.assertEqual(compact["aggregate"]["verdict_counts"]["useful_breadth"], 10)

    def test_diversity_audit_uses_defaults_and_returns_compact_report(self):
        with patch("canon.product.service.run_diversity_audit") as audit:
            audit.return_value = {
                "mode": service.DEFAULT_DIVERSITY_MODE,
                "method_set_id": "baseline_methods_v1",
                "diverse_method_id": service.DEFAULT_DIVERSE_METHOD_ID,
                "baseline_method_id": service.DEFAULT_DIVERSITY_BASELINE_METHOD_ID,
                "query_count": 25,
                "aggregate": {"verdict_counts": {"mixed_breadth": 1}},
                "queries": [{"large": "payload"}],
            }
            report = service.diversity_audit({})
        audit.assert_called_once_with(
            mode=service.DEFAULT_DIVERSITY_MODE,
            diverse_method_id=service.DEFAULT_DIVERSE_METHOD_ID,
            baseline_method_id=service.DEFAULT_DIVERSITY_BASELINE_METHOD_ID,
        )
        self.assertNotIn("queries", report)
        self.assertEqual(report["query_count"], 25)

    def test_compact_diversity_query_omits_added_and_removed_titles(self):
        compact = service.compact_diversity_query(
            {
                "query_id": "q1",
                "query_type": "synthesis",
                "query": "What changed?",
                "verdict": "useful_breadth",
                "cluster_delta": 2,
                "rank_overlap": 0.25,
                "added_titles": [{"title": "A"}],
                "removed_titles": [{"title": "B"}],
            }
        )
        self.assertEqual(compact["query_id"], "q1")
        self.assertNotIn("added_titles", compact)
        self.assertNotIn("removed_titles", compact)

    def test_diversity_queries_filters_loaded_report(self):
        payload = {
            "mode": "m",
            "method_set_id": "methods",
            "diverse_method_id": "diverse",
            "baseline_method_id": "lexical",
            "query_count": 2,
            "aggregate": {},
            "queries": [
                {"query_id": "q1", "query_type": "synthesis", "verdict": "useful_breadth"},
                {"query_id": "q2", "query_type": "off_topic", "verdict": "off_topic_breadth_risk"},
            ],
        }
        with patch("canon.product.service.load_diversity_report_from_params", return_value=payload):
            report = service.diversity_queries({"verdict": "useful_breadth", "limit": "1"})
        self.assertEqual(report["result_count"], 1)
        self.assertEqual(report["queries"][0]["query_id"], "q1")

    def test_diversity_query_detail_returns_full_query_payload(self):
        payload = {
            "mode": "m",
            "method_set_id": "methods",
            "diverse_method_id": "diverse",
            "baseline_method_id": "lexical",
            "query_count": 1,
            "aggregate": {},
            "queries": [
                {"query_id": "q1", "added_titles": [{"title": "A"}]},
            ],
        }
        with patch("canon.product.service.load_diversity_report_from_params", return_value=payload):
            report = service.diversity_query_detail("q1", {})
        self.assertIn("added_titles", report["query"])

    def test_diversity_query_detail_unknown_query_raises_404(self):
        with patch("canon.product.service.load_diversity_report_from_params", return_value={"queries": []}):
            with self.assertRaises(service.ProductError) as context:
                service.diversity_query_detail("missing", {})
        self.assertEqual(context.exception.status_code, 404)


if __name__ == "__main__":
    unittest.main()
