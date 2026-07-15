import unittest
from unittest.mock import Mock, patch

from canon.product.server import CanonHandler, api_routes, not_found_payload


class ProductServerRouteTests(unittest.TestCase):
    def test_root_route_returns_api_index(self):
        handler = object.__new__(CanonHandler)
        handler.path = "/"
        handler.send_json = Mock()

        handler.do_GET()

        payload = handler.send_json.call_args.args[0]
        self.assertEqual(payload["service"], "canon")
        self.assertIn("/v1/routes", payload["get"])
        self.assertGreater(payload["route_count"], 0)
        self.assertIn("/v1/projects/start", payload["post"])
        self.assertIn("/v1/evidence-packets", payload["post"])
        self.assertIn("/v1/frame-coverage", payload["post"])
        self.assertIn("/v1/intelligence-brief", payload["post"])
        self.assertIn("/v1/report-quality", payload["post"])
        self.assertIn("/v1/intelligence-brief/evaluate", payload["post"])
        self.assertIn("/v1/alert-digest", payload["post"])
        self.assertIn("/v1/alert-digest/evaluate", payload["post"])
        self.assertIn("/v1/flagship-handoff", payload["post"])
        self.assertIn("/v1/acceptance-scenario", payload["post"])
        self.assertIn("/v1/intelligence-review/prepare", payload["post"])
        self.assertIn("/v1/intelligence-review/status", payload["post"])
        self.assertIn("/v1/intelligence-review/feedback", payload["post"])
        self.assertIn("/v1/intelligence-review/handoff", payload["post"])
        self.assertIn("/v1/intelligence-review/export-csv", payload["post"])
        self.assertIn("/v1/intelligence-review/import-csv", payload["post"])
        self.assertIn("/v1/prehuman-check", payload["post"])

    def test_favicon_route_returns_empty_success(self):
        handler = object.__new__(CanonHandler)
        handler.path = "/favicon.ico"
        handler.send_empty = Mock()

        handler.do_GET()

        handler.send_empty.assert_called_once()

    def test_unknown_route_returns_route_discovery_hint(self):
        handler = object.__new__(CanonHandler)
        handler.path = "/v1/missing"
        handler.send_json = Mock()

        handler.do_GET()

        payload, status = handler.send_json.call_args.args
        self.assertEqual(status.value, 404)
        self.assertEqual(payload["error"], "not_found")
        self.assertEqual(payload["routes_url"], "/v1/routes")
        self.assertIn("GET /v1/routes", payload["available_routes"])

    def test_options_route_allows_preflight(self):
        handler = object.__new__(CanonHandler)
        handler.send_response = Mock()
        handler.send_header = Mock()
        handler.end_headers = Mock()

        handler.do_OPTIONS()

        handler.send_response.assert_called_once()
        header_calls = [call.args for call in handler.send_header.call_args_list]
        self.assertIn(("Access-Control-Allow-Methods", "GET, POST, OPTIONS"), header_calls)

    def test_routes_endpoint_returns_route_metadata(self):
        handler = object.__new__(CanonHandler)
        handler.path = "/v1/routes"
        handler.send_json = Mock()

        handler.do_GET()

        payload = handler.send_json.call_args.args[0]
        self.assertIn("human_review_boundary", payload)
        paths = {route["path"]: route for route in payload["routes"]}
        self.assertEqual(paths["/v1/projects/start"]["method"], "POST")
        self.assertIn("project_name", paths["/v1/projects/start"]["required"])
        self.assertEqual(paths["/v1/flagship-handoff"]["method"], "POST")
        self.assertEqual(paths["/v1/acceptance-scenario"]["method"], "POST")
        self.assertEqual(paths["/v1/frame-coverage"]["method"], "POST")
        self.assertEqual(paths["/v1/report-quality"]["method"], "POST")
        self.assertIn("research_frame", paths["/v1/frame-coverage"]["required"])
        self.assertIn("mode", paths["/v1/flagship-handoff"]["optional"])
        self.assertEqual(paths["/v1/intelligence-review/import-csv"]["example"]["csv_path"], "reports/intelligence_brief_review_tasks_ai_infra_geo_risk_demo.review.csv")
        self.assertEqual(paths["/v1/intelligence-review/feedback"]["method"], "POST")
        self.assertEqual(paths["/v1/intelligence-review/handoff"]["method"], "POST")
        self.assertEqual(paths["/v1/prehuman-check"]["method"], "POST")

    def test_api_routes_include_examples_for_public_post_routes(self):
        metadata = api_routes()
        missing_examples = [
            route["path"]
            for route in metadata["routes"]
            if route["method"] == "POST" and route["example"] is None
        ]
        self.assertEqual(missing_examples, [])

    def test_not_found_payload_lists_only_matching_method_routes(self):
        payload = not_found_payload("POST", "/v1/nope")
        self.assertIn("POST /v1/evidence-packets", payload["available_routes"])
        self.assertNotIn("GET /v1/routes", payload["available_routes"])

    def test_post_routes_include_integration_endpoints(self):
        routes = {
            "/v1/projects/start": ("start_project", {"status": "ready"}),
            "/v1/sources/profile": ("source_profile", {"source_shape": "document_file"}),
            "/v1/sources/ingest": ("source_ingest", {"mode": "m"}),
            "/v1/corpora/build": ("corpus_build", {"corpus": {"corpus_id": "c"}}),
            "/v1/model-evaluation": ("model_evaluation", {"report_id": "semantic_model_evaluation_v1"}),
            "/v1/evidence-packets": ("evidence_packets", {"status": "complete"}),
            "/v1/frame-coverage": ("frame_coverage_report", {"status": "coverage_gap_human_review_required"}),
            "/v1/intelligence-brief": ("intelligence_brief", {"status": "ready_for_human_review"}),
            "/v1/report-quality": ("report_quality_gate", {"status": "pass_human_review_required"}),
            "/v1/intelligence-brief/evaluate": ("intelligence_brief_evaluation", {"status": "pass"}),
            "/v1/alert-digest": ("alert_digest", {"status": "ready_for_human_review"}),
            "/v1/alert-digest/evaluate": ("alert_digest_evaluation", {"status": "pass"}),
            "/v1/flagship-handoff": ("flagship_handoff", {"status": "automated_pass_human_review_required"}),
            "/v1/acceptance-scenario": ("acceptance_scenario", {"status": "automated_pass_human_review_required"}),
            "/v1/intelligence-review/prepare": ("intelligence_review_prepare", {"report_id": "intelligence_brief_review_tasks_v1"}),
            "/v1/intelligence-review/status": ("intelligence_review_status", {"status": "incomplete"}),
            "/v1/intelligence-review/feedback": ("intelligence_review_feedback", {"status": "ready"}),
            "/v1/intelligence-review/handoff": ("intelligence_review_handoff", {"status": "ready_for_human_review"}),
            "/v1/intelligence-review/export-csv": ("intelligence_review_export_csv", {"status": "review_csv_written"}),
            "/v1/intelligence-review/import-csv": ("intelligence_review_import_csv", {"status": "imported"}),
            "/v1/prehuman-check": ("prehuman_check", {"status": "automated_pass_human_review_required"}),
        }
        for path, (service_name, response) in routes.items():
            with self.subTest(path=path):
                handler = object.__new__(CanonHandler)
                handler.path = path
                handler.read_json = Mock(return_value={"ok": True})
                handler.send_json = Mock()
                with patch(f"canon.product.server.service.{service_name}", return_value=response) as service_fn:
                    handler.do_POST()

                service_fn.assert_called_once_with({"ok": True})
                handler.send_json.assert_called_once_with(response)


if __name__ == "__main__":
    unittest.main()
