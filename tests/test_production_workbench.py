import json
import unittest
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from canon.product import service


class ProductionWorkbenchTests(unittest.TestCase):
    def test_production_status_reports_shippable_boundary_and_quality_state(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_dir = root / "data"
            reports_dir = root / "reports"
            (data_dir / "processed").mkdir(parents=True)
            reports_dir.mkdir()
            (data_dir / "processed" / "works_ai_infra_geo_risk_demo.json").write_text("[]", encoding="utf-8")
            (data_dir / "processed" / "works_beir_scifact_full.json").write_text("[]", encoding="utf-8")
            (data_dir / "processed" / "works_social_science_ir_v1_harvest10.json").write_text("[]", encoding="utf-8")
            (data_dir / "processed" / "works_unit_tmp.json").write_text("[]", encoding="utf-8")
            (reports_dir / service.DEFAULT_PUBLISHABLE_PACKAGE).write_text(
                json.dumps(
                    {
                        "status": "publication_blocked_evidence_incomplete",
                        "checks": [
                            {"id": "automated_checks_complete", "passed": True},
                            {"id": "human_review_complete", "passed": False},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (reports_dir / "publishable_benchmark_card_v1.json").write_text(
                json.dumps(
                    {
                        "status": "benchmark_card_ready_with_automated_failures",
                        "aggregate_diagnostics": {"best_candidate_recall": 0.87},
                    }
                ),
                encoding="utf-8",
            )
            (reports_dir / "stage1_gate_v1.json").write_text(
                json.dumps({"status": "blocked", "failures": ["candidate_recall_below_target"]}),
                encoding="utf-8",
            )
            settings = SimpleNamespace(data_dir=data_dir, reports_dir=reports_dir)

            with patch("canon.product.service.load_settings", return_value=settings):
                report = service.production_status()

        self.assertEqual(report["status"], "ship_user_experience_ready")
        self.assertEqual(report["quality_state"]["package_checks_passed"], 1)
        self.assertEqual(report["quality_state"]["package_checks_total"], 2)
        self.assertEqual(report["quality_state"]["blocked_checks"], ["human_review_complete"])
        self.assertEqual(report["quality_state"]["stage1_gate_status"], "blocked")
        self.assertIn("ai_infra_geo_risk_demo", report["available_modes"])
        self.assertNotIn("unit_tmp", report["available_modes"])
        self.assertIn("corpus-backed evidence packets", report["shippable_without_human_review"])
        self.assertIn("autonomous factual conclusions", report["blocked_without_human_review"])
        self.assertEqual(report["endpoints"]["run"], "POST /v1/production/evidence-workbench")
        self.assertEqual(report["endpoints"]["corpus_setup"], "POST /v1/production/corpus-setup")
        self.assertIn("full_embedding_store", {row["scope"] for row in report["candidate_scopes"]})
        self.assertIn("moonshotai/kimi-k3", {row["model"] for row in report["generation_models"]})
        self.assertEqual(report["recommended_defaults"]["external_expansion"], "off_by_default_opt_in_openalex")
        self.assertIn("external_source/ONLINE", report["operational_boundary"]["external_web"])

    def test_production_evidence_workbench_builds_session_and_writes_telemetry(self):
        packet_response = {
            "status": "complete",
            "request_id": "prod_fixed",
            "project_id": "production_workbench",
            "query": "grid risk",
            "mode": "m",
            "policy": "rag",
            "evidence_packets": [
                {
                    "packet_id": "packet_1",
                    "support_level": "mixed",
                    "confidence": "medium",
                    "supporting_evidence": [
                        {
                            "evidence_id": "C1",
                            "rank": 1,
                            "title": "Grid Stress",
                            "source_name": "Regional Energy Monitor",
                            "published_at": "2026",
                            "source_type": "news",
                            "language": "en",
                            "evidence_scope": "private_corpus",
                            "document_id": "doc_1",
                            "chunk_id": "chunk_1",
                            "text": "AI data centers can increase grid stress.",
                            "citation": {"citation_id": "C1"},
                        }
                    ],
                    "source_diversity": {"warnings": []},
                    "frame_coverage": {"status": "partial"},
                    "evidence_scope_summary": {"private_corpus": 1, "external_source": 0},
                }
            ],
            "query_diagnostics": {
                "matched_terms": ["grid", "risk"],
                "weak_terms": [],
                "query_variants": [{"query": "AI data center grid permitting"}],
            },
            "coverage_gaps": [{"gap": "Missing permitting evidence.", "suggested_next_query": "grid permitting"}],
            "external_expansion": {"status": "planned", "executed": False},
            "answer_report": {
                "answer": "Draft answer [C1]",
                "citations": [{"citation_id": "C1"}],
                "support_assessment": {
                    "support_level": "moderate",
                    "query_term_coverage": 1.0,
                    "covered_focus_terms": ["grid", "risk"],
                    "missing_focus_terms": [],
                },
            },
            "contract_validation": {"status": "pass"},
        }
        with TemporaryDirectory() as temp_dir:
            reports_dir = Path(temp_dir)
            settings = SimpleNamespace(reports_dir=reports_dir)
            with patch("canon.product.service.load_settings", return_value=settings):
                with patch("canon.product.service.evidence_packets", return_value=packet_response) as packets:
                    report = service.production_evidence_workbench(
                        {
                            "session_id": "prod_fixed",
                            "query": "grid risk",
                            "mode": "m",
                            "top_k": 8,
                            "suggest_external_expansion": True,
                            "allowed_source_types": ["official"],
                            "max_external_queries": 2,
                        }
                    )

            telemetry_path = reports_dir / "production_workbench_telemetry_v1.jsonl"
            telemetry = [json.loads(line) for line in telemetry_path.read_text(encoding="utf-8").splitlines()]

        packets.assert_called_once()
        packet_payload = packets.call_args.args[0]
        self.assertEqual(packet_payload["request_id"], "prod_fixed")
        self.assertEqual(packet_payload["evidence_requirements"]["top_k"], 8)
        self.assertEqual(packet_payload["evidence_requirements"]["external_expansion"]["allowed_source_types"], ["official"])
        self.assertEqual(report["status"], "ready_with_gaps")
        self.assertEqual(report["session_id"], "prod_fixed")
        self.assertEqual(report["relevance_gate"]["status"], "pass")
        self.assertEqual(report["draft_brief"]["status"], "evidence_note_ready")
        self.assertEqual(report["draft_brief"]["method"], "deterministic_evidence_note")
        self.assertEqual(report["evidence_packet"]["evidence_count"], 1)
        self.assertEqual(report["evidence_packet"]["usable_evidence_count"], 1)
        self.assertEqual(report["evidence_cards"][0]["evidence_id"], "C1")
        self.assertEqual(report["evidence_cards"][0]["relevance"]["usage_status"], "usable_candidate_evidence")
        self.assertIn("Try a follow-up query: grid permitting", report["recommended_actions"])
        self.assertEqual(report["feedback"]["endpoint"], "POST /v1/production/feedback")
        self.assertEqual(telemetry[0]["session_id"], "prod_fixed")
        self.assertEqual(telemetry[0]["relevance_gate_status"], "pass")
        self.assertEqual(telemetry[0]["boundary"], "Local product telemetry only; this is not a human-review label.")

    def test_production_evidence_workbench_keeps_external_results_marked(self):
        packet_response = {
            "status": "complete",
            "query": "grid risk",
            "mode": "m",
            "policy": "rag",
            "evidence_packets": [
                {
                    "packet_id": "packet_1",
                    "support_level": "mixed",
                    "confidence": "medium",
                    "supporting_evidence": [
                        {
                            "evidence_id": "C1",
                            "rank": 1,
                            "title": "Corpus Grid Stress",
                            "source_name": "Corpus",
                            "text": "Grid risk evidence.",
                            "evidence_scope": "private_corpus",
                        }
                    ],
                    "evidence_scope_summary": {"private_corpus": 1, "external_source": 0},
                }
            ],
            "query_diagnostics": {"matched_terms": ["grid", "risk"], "weak_terms": []},
            "coverage_gaps": [],
            "external_expansion": {"status": "planned", "executed": False},
            "answer_report": {
                "citations": [{"citation_id": "C1"}],
                "support_assessment": {
                    "support_level": "moderate",
                    "query_term_coverage": 1.0,
                    "covered_focus_terms": ["grid", "risk"],
                    "missing_focus_terms": [],
                },
            },
            "contract_validation": {"status": "pass"},
        }
        external_card = {
            "evidence_id": "X1",
            "title": "Online Grid Result",
            "source_name": "OpenAlex",
            "text": "Online evidence.",
            "evidence_scope": "external_source",
            "retrieval_stage": "external_openalex_search",
            "relevance": {"usage_status": "external_candidate_only"},
        }

        with TemporaryDirectory() as temp_dir:
            settings = SimpleNamespace(reports_dir=Path(temp_dir))
            with patch("canon.product.service.load_settings", return_value=settings):
                with patch("canon.product.service.evidence_packets", return_value=packet_response):
                    with patch(
                        "canon.product.service.production_external_search",
                        return_value={
                            "status": "executed",
                            "executed": True,
                            "provider": "openalex",
                            "result_count": 1,
                            "evidence_cards": [external_card],
                        },
                    ):
                        report = service.production_evidence_workbench(
                            {
                                "session_id": "prod_external",
                                "query": "grid risk",
                                "mode": "m",
                                "execute_external_search": True,
                                "write_telemetry": False,
                            }
                        )

        self.assertEqual(report["evidence_packet"]["evidence_scope_summary"], {"private_corpus": 1, "external_source": 1})
        self.assertEqual(report["evidence_packet"]["external_evidence_count"], 1)
        self.assertEqual(report["external_expansion"]["status"], "executed")
        self.assertTrue(report["external_expansion"]["enabled"])
        self.assertEqual(report["evidence_cards"][-1]["evidence_id"], "X1")
        self.assertEqual(report["evidence_cards"][-1]["evidence_scope"], "external_source")
        self.assertEqual(report["evidence_cards"][-1]["retrieval_stage"], "external_openalex_search")

    def test_production_external_search_normalizes_openalex_cards(self):
        settings = SimpleNamespace(
            raw={
                "openalex": {
                    "base_url": "https://api.openalex.org",
                    "language": "en",
                    "from_publication_year": 2020,
                    "to_publication_year": 2026,
                }
            },
            contact_email="test@example.com",
        )
        payload = {
            "results": [
                {
                    "id": "https://openalex.org/W1",
                    "doi": "https://doi.org/10.1/example",
                    "title": "Online Grid Study",
                    "publication_year": 2026,
                    "language": "en",
                    "abstract_inverted_index": {"Grid": [0], "risk": [1]},
                    "primary_location": {
                        "landing_page_url": "https://example.test/work",
                        "source": {"display_name": "Journal"},
                    },
                    "open_access": {"is_oa": True},
                    "cited_by_count": 7,
                }
            ]
        }

        with patch("canon.product.service.load_settings", return_value=settings):
            with patch("canon.product.service.fetch_json", return_value=payload) as fetch:
                report = service.production_external_search(
                    "grid risk",
                    {"execute_external_search": True, "max_external_results": 1},
                )

        self.assertEqual(report["status"], "executed")
        self.assertEqual(report["result_count"], 1)
        self.assertIn("search=grid+risk", fetch.call_args.args[0])
        card = report["evidence_cards"][0]
        self.assertEqual(card["evidence_id"], "X1")
        self.assertEqual(card["evidence_scope"], "external_source")
        self.assertEqual(card["retrieval_stage"], "external_openalex_search")
        self.assertEqual(card["relevance"]["source_marker"], "ONLINE")

    def test_production_external_search_sanitizes_natural_language_question(self):
        settings = SimpleNamespace(raw={"openalex": {"base_url": "https://api.openalex.org"}}, contact_email="test@example.com")

        with patch("canon.product.service.load_settings", return_value=settings):
            with patch("canon.product.service.fetch_json", return_value={"results": []}) as fetch:
                report = service.production_external_search(
                    "What are the grid risks around AI data center expansion?",
                    {"execute_external_search": True, "max_external_results": 1},
                )

        self.assertEqual(report["search_query"], "What are the grid risks around AI data center expansion")
        self.assertIn("search=What+are+the+grid+risks+around+AI+data+center+expansion", fetch.call_args.args[0])
        self.assertNotIn("%3F", fetch.call_args.args[0])

    def test_merge_external_expansion_reports_failed_requested_search(self):
        report = service.merge_external_expansion(
            {"status": "disabled", "enabled": False, "executed": False},
            {
                "status": "failed",
                "executed": False,
                "provider": "openalex",
                "error": "HTTP Error 400: Bad Request",
                "evidence_cards": [],
                "boundary": "External search failed; no external evidence was mixed into corpus results.",
            },
        )

        self.assertEqual(report["status"], "failed")
        self.assertTrue(report["enabled"])
        self.assertFalse(report["executed"])
        self.assertIn("400", report["error"])

    def test_production_evidence_workbench_blocks_draft_on_corpus_mismatch(self):
        packet_response = {
            "status": "complete",
            "query": "AI risks",
            "mode": "social_science_ir_v1_harvest10",
            "policy": "rag",
            "evidence_packets": [
                {
                    "packet_id": "packet_1",
                    "support_level": "mixed",
                    "confidence": "medium",
                    "supporting_evidence": [
                        {
                            "evidence_id": "C1",
                            "rank": 1,
                            "title": "Horizontal Inequalities and Ethnonationalist Civil War",
                            "source_name": "APSR",
                            "text": "Civil war grievances and mobilization.",
                        }
                    ],
                }
            ],
            "query_diagnostics": {"matched_terms": [], "weak_terms": ["ai", "risks"]},
            "coverage_gaps": [],
            "external_expansion": {"status": "disabled", "executed": False},
            "answer_report": {
                "answer": "Template answer that should not be surfaced.",
                "citations": [{"citation_id": "C1"}],
                "support_assessment": {
                    "support_level": "weak",
                    "query_term_coverage": 0.0,
                    "covered_focus_terms": [],
                    "missing_focus_terms": ["ai", "risks"],
                },
            },
            "contract_validation": {"status": "pass"},
        }
        with TemporaryDirectory() as temp_dir:
            settings = SimpleNamespace(reports_dir=Path(temp_dir))
            with patch("canon.product.service.load_settings", return_value=settings):
                with patch("canon.product.service.evidence_packets", return_value=packet_response):
                    report = service.production_evidence_workbench(
                        {
                            "session_id": "prod_mismatch",
                            "query": "AI risks",
                            "mode": "social_science_ir_v1_harvest10",
                            "write_telemetry": False,
                        }
                    )

        self.assertEqual(report["status"], "corpus_mismatch_no_grounded_answer")
        self.assertEqual(report["relevance_gate"]["status"], "corpus_mismatch")
        self.assertFalse(report["relevance_gate"]["draft_allowed"])
        self.assertEqual(report["evidence_packet"]["usable_evidence_count"], 0)
        self.assertEqual(report["draft_brief"]["status"], "blocked_by_relevance_gate")
        self.assertEqual(report["draft_brief"]["text"], "")
        self.assertEqual(report["evidence_cards"][0]["relevance"]["relevance_label"], "off_query")
        self.assertEqual(report["evidence_cards"][0]["relevance"]["usage_status"], "diagnostic_candidate_only")

    def test_production_evidence_workbench_rejects_broad_term_overlap_only(self):
        packet_response = {
            "status": "complete",
            "query": "What are the grid risks around AI data center expansion?",
            "mode": "mixed_benchmark",
            "policy": "rag",
            "evidence_packets": [
                {
                    "packet_id": "packet_1",
                    "support_level": "mixed",
                    "confidence": "medium",
                    "supporting_evidence": [
                        {
                            "evidence_id": "C1",
                            "rank": 1,
                            "title": "Introducing the Civil Wars Mediation Dataset",
                            "source_name": "Journal",
                            "text": "This article introduces a data set on mediation in civil war.",
                        },
                        {
                            "evidence_id": "C2",
                            "rank": 2,
                            "title": "International Guidelines for Sepsis",
                            "source_name": "Medical Journal",
                            "text": "The guideline describes treatment data and study center variation.",
                        },
                    ],
                }
            ],
            "query_diagnostics": {
                "matched_terms": ["data", "center"],
                "weak_terms": ["ai", "grid", "risks", "expansion"],
            },
            "coverage_gaps": [],
            "external_expansion": {"status": "disabled", "executed": False},
            "answer_report": {
                "answer": "Template answer that should not be surfaced.",
                "citations": [{"citation_id": "C1"}, {"citation_id": "C2"}],
                "support_assessment": {
                    "support_level": "moderate",
                    "query_term_coverage": 0.285714,
                    "covered_focus_terms": ["data", "center"],
                    "missing_focus_terms": ["ai", "grid", "risks", "expansion"],
                },
            },
            "contract_validation": {"status": "pass"},
        }
        with TemporaryDirectory() as temp_dir:
            settings = SimpleNamespace(reports_dir=Path(temp_dir))
            with patch("canon.product.service.load_settings", return_value=settings):
                with patch("canon.product.service.evidence_packets", return_value=packet_response):
                    report = service.production_evidence_workbench(
                        {
                            "session_id": "prod_broad_only",
                            "query": "What are the grid risks around AI data center expansion?",
                            "mode": "mixed_benchmark",
                            "write_telemetry": False,
                        }
                    )

        self.assertEqual(report["status"], "corpus_mismatch_no_grounded_answer")
        self.assertEqual(report["relevance_gate"]["usable_evidence_count"], 0)
        self.assertEqual(report["draft_brief"]["status"], "blocked_by_relevance_gate")
        self.assertEqual(
            [row["relevance"]["usage_status"] for row in report["evidence_cards"]],
            ["diagnostic_candidate_only", "diagnostic_candidate_only"],
        )

    def test_production_evidence_workbench_can_generate_model_draft_from_gated_evidence(self):
        packet_response = {
            "status": "complete",
            "query": "grid risks around AI data center expansion",
            "mode": "m",
            "policy": "rag",
            "evidence_packets": [
                {
                    "packet_id": "packet_1",
                    "supporting_evidence": [
                        {
                            "evidence_id": "C1",
                            "rank": 1,
                            "title": "Chile Grid Interconnection Note",
                            "source_name": "Regulator",
                            "text": "AI data center loads increase grid interconnection pressure.",
                        },
                        {
                            "evidence_id": "C2",
                            "rank": 2,
                            "title": "Sepsis Guidelines",
                            "source_name": "Medical Journal",
                            "text": "Clinical data center variation in sepsis outcomes.",
                        },
                        {
                            "evidence_id": "C3",
                            "rank": 3,
                            "title": "AI Region Grid Expansion",
                            "source_name": "Newsroom",
                            "text": "AI region expansion requires grid upgrades and staged approvals.",
                        },
                    ],
                }
            ],
            "query_diagnostics": {
                "matched_terms": ["ai", "data", "center", "grid"],
                "weak_terms": ["risks"],
            },
            "coverage_gaps": [],
            "external_expansion": {"status": "disabled", "executed": False},
            "answer_report": {
                "answer": "Template answer ignored.",
                "citations": [{"citation_id": "C1"}, {"citation_id": "C2"}, {"citation_id": "C3"}],
                "support_assessment": {
                    "support_level": "moderate",
                    "query_term_coverage": 0.6,
                    "covered_focus_terms": ["ai", "data", "center", "grid"],
                    "missing_focus_terms": ["risks"],
                },
            },
            "contract_validation": {"status": "pass"},
        }

        with TemporaryDirectory() as temp_dir:
            settings = SimpleNamespace(reports_dir=Path(temp_dir))
            with patch("canon.product.service.load_settings", return_value=settings):
                with patch("canon.product.service.evidence_packets", return_value=packet_response):
                    with patch(
                        "canon.product.service.get_generation_provider",
                        return_value=FakeGenerationProvider(),
                    ):
                        report = service.production_evidence_workbench(
                            {
                                "session_id": "prod_model",
                                "query": "grid risks around AI data center expansion",
                                "mode": "m",
                                "generator_provider": "openrouter",
                                "generator_model": "openai/gpt-4.1-mini",
                                "write_telemetry": False,
                            }
                        )

        self.assertEqual(report["draft_brief"]["status"], "model_draft_ready")
        self.assertEqual(report["draft_brief"]["method"], "hosted_model_generation")
        self.assertEqual(report["draft_brief"]["generator"]["provider"], "openrouter")
        self.assertEqual(report["draft_brief"]["citations"], [{"citation_id": "C1"}, {"citation_id": "C3"}])
        self.assertIn("[C1]", report["draft_brief"]["text"])

    def test_production_evidence_workbench_can_use_model_candidate_pool_and_reranker(self):
        candidate_report = {
            "candidates": [
                {
                    "chunk_id": "grid_1",
                    "work_id": "official-grid",
                    "title": "Grid Interconnection Note",
                    "source_name": "Regulator",
                    "year": 2026,
                    "section": "grid planning",
                    "retrieval_sources": ["lexical", "vector:openrouter"],
                    "scores": {"lexical": 1.0, "vector:openrouter": 0.9},
                    "ranks": {"lexical": 1, "vector:openrouter": 1},
                    "preview": "AI data center expansion increases grid interconnection pressure and congestion.",
                },
                {
                    "chunk_id": "sepsis_1",
                    "work_id": "medical",
                    "title": "Sepsis Guidelines",
                    "source_name": "Medical Journal",
                    "year": 2021,
                    "section": "body",
                    "retrieval_sources": ["lexical"],
                    "scores": {"lexical": 0.2},
                    "ranks": {"lexical": 2},
                    "preview": "Clinical data center variation in sepsis outcomes.",
                },
                {
                    "chunk_id": "ai_region_1",
                    "work_id": "cloud",
                    "title": "AI Region Expansion",
                    "source_name": "Newsroom",
                    "year": 2026,
                    "section": "body",
                    "retrieval_sources": ["vector:openrouter"],
                    "scores": {"vector:openrouter": 0.8},
                    "ranks": {"vector:openrouter": 2},
                    "preview": "AI cloud regions require grid upgrades and staged electricity approvals.",
                },
            ],
            "source_counts": {"lexical": 2, "vector:openrouter": 2},
        }
        fake_generator = FakeGenerationProvider()

        with TemporaryDirectory() as temp_dir:
            settings = SimpleNamespace(reports_dir=Path(temp_dir))
            with patch("canon.product.service.load_settings", return_value=settings):
                with patch("canon.product.service.build_windowed_candidate_pool", return_value=candidate_report) as pool:
                    with patch("canon.product.service.get_rerank_provider", return_value=FakeRerankProvider()):
                        with patch("canon.product.service.get_generation_provider", return_value=fake_generator):
                            report = service.production_evidence_workbench(
                                {
                                    "session_id": "prod_model_candidates",
                                    "query": "grid risks around AI data center expansion",
                                    "mode": "m",
                                    "top_k": 3,
                                    "retrieval_engine": "model_candidate_pool",
                                    "retrieval_provider": "openrouter",
                                    "retrieval_model": "qwen/qwen3-embedding-8b",
                                    "reranker_provider": "cohere",
                                    "reranker_model": "rerank-v4.0-pro",
                                    "generator_provider": "openrouter",
                                    "generator_model": "openai/gpt-4.1-mini",
                                    "write_telemetry": False,
                                }
                            )

        pool.assert_called_once()
        self.assertEqual(pool.call_args.kwargs["lexical_window_k"], service.DEFAULT_PRODUCTION_LEXICAL_WINDOW_K)
        self.assertEqual(report["retrieval"]["engine"], "model_candidate_pool")
        self.assertEqual(report["retrieval"]["provider"], "openrouter")
        self.assertEqual(report["retrieval"]["candidate_scope"], "lexical_window")
        self.assertEqual(report["retrieval"]["reranker_provider"], "cohere")
        self.assertEqual(report["draft_brief"]["status"], "model_draft_ready")
        self.assertEqual(report["draft_brief"]["citations"], [{"citation_id": "C1"}, {"citation_id": "C2"}])
        self.assertEqual(report["evidence_cards"][0]["chunk_id"], "grid_1")
        self.assertEqual(report["evidence_cards"][1]["chunk_id"], "ai_region_1")
        self.assertEqual(report["evidence_cards"][2]["relevance"]["usage_status"], "diagnostic_candidate_only")
        self.assertIn("Grid Interconnection Note", fake_generator.prompt)
        self.assertNotIn("Sepsis Guidelines", fake_generator.prompt)

    def test_production_corpus_setup_profiles_only_or_builds_user_corpus(self):
        with patch("canon.product.service.source_profile", return_value={"source_shape": "folder"}) as profile:
            with patch("canon.product.service.source_ingest", return_value={"status": "ingested"}) as ingest:
                with patch(
                    "canon.product.service.corpus_build",
                    return_value={"corpus": {"corpus_id": "my_topic_corpus"}, "validation": {"status": "pass"}},
                ) as build:
                    report = service.production_corpus_setup(
                        {
                            "input_path": "data/my_docs",
                            "mode": "my_topic_v1",
                            "corpus_id": "my_topic_corpus",
                            "build_corpus": True,
                        }
                    )

        self.assertEqual(report["status"], "corpus_ready")
        self.assertEqual(report["recommended_mode"], "my_topic_corpus")
        profile.assert_called_once()
        ingest.assert_called_once()
        build.assert_called_once()

        with patch("canon.product.service.source_profile", return_value={"source_shape": "folder"}) as profile:
            with patch("canon.product.service.source_ingest") as ingest:
                report = service.production_corpus_setup(
                    {"input_path": "data/my_docs", "mode": "my_topic_v1", "profile_only": True}
                )

        self.assertEqual(report["status"], "profile_ready")
        profile.assert_called_once()
        ingest.assert_not_called()

    def test_production_feedback_writes_user_experience_jsonl(self):
        with TemporaryDirectory() as temp_dir:
            reports_dir = Path(temp_dir)
            settings = SimpleNamespace(reports_dir=reports_dir)
            with patch("canon.product.service.load_settings", return_value=settings):
                response = service.production_feedback(
                    {
                        "session_id": "prod_123",
                        "query": "grid risk",
                        "feedback_type": "useful",
                        "rating": "5",
                        "comment": "Useful for triage.",
                    }
                )

            feedback_path = reports_dir / "production_feedback_v1.jsonl"
            rows = [json.loads(line) for line in feedback_path.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(response["status"], "feedback_recorded")
        self.assertEqual(rows[0]["session_id"], "prod_123")
        self.assertEqual(rows[0]["rating"], 5)
        self.assertTrue(rows[0]["not_human_review_label"])

    def test_production_feedback_rejects_empty_and_invalid_feedback(self):
        with TemporaryDirectory() as temp_dir:
            settings = SimpleNamespace(reports_dir=Path(temp_dir))
            with patch("canon.product.service.load_settings", return_value=settings):
                with self.assertRaises(service.ProductError):
                    service.production_feedback({"session_id": "prod_123"})
                with self.assertRaises(service.ProductError):
                    service.production_feedback({"session_id": "prod_123", "rating": 6})

    def test_model_generated_draft_retries_then_marks_empty_response(self):
        provider = EmptyGenerationProvider()

        with patch("canon.product.service.get_generation_provider", return_value=provider):
            brief = service.model_generated_draft_brief(
                query="claim",
                evidence=[{"evidence_id": "C1", "title": "Evidence", "text": "Evidence text."}],
                diagnostic_count=0,
                support={"support_level": "moderate"},
                citations=[{"citation_id": "C1"}],
                generator_provider="openrouter",
                generator_model="moonshotai/kimi-k3",
            )

        self.assertEqual(provider.call_count, 2)
        self.assertEqual(brief["status"], "model_generation_empty")
        self.assertEqual(brief["citation_count"], 0)


@dataclass
class FakeGenerationResult:
    provider: str = "openrouter"
    model: str = "openai/gpt-4.1-mini"
    text: str = "Grid interconnection pressure is the clearest cited risk [C1]."


class FakeGenerationProvider:
    def generate(self, prompt):
        self.prompt = prompt
        return FakeGenerationResult()


class EmptyGenerationProvider:
    def __init__(self):
        self.call_count = 0

    def generate(self, prompt):
        self.call_count += 1
        return FakeGenerationResult(text="")


class FakeRerankProvider:
    provider = "cohere"
    model = "rerank-v4.0-pro"

    def rerank(self, query, documents, top_n):
        from canon.rerank.providers import RerankResult

        return [
            RerankResult(index=0, score=0.95),
            RerankResult(index=2, score=0.9),
            RerankResult(index=1, score=0.1),
        ][:top_n]


if __name__ == "__main__":
    unittest.main()
