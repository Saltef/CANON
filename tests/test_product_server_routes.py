import unittest
from unittest.mock import Mock, patch

from canon.product.server import CanonHandler, api_routes


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
        self.assertIn("/v1/evidence-packets", payload["post"])
        self.assertIn("/v1/frame-coverage", payload["post"])
        self.assertIn("/v1/intelligence-brief", payload["post"])
        self.assertIn("/v1/intelligence-brief/evaluate", payload["post"])
        self.assertIn("/v1/alert-digest", payload["post"])
        self.assertIn("/v1/alert-digest/evaluate", payload["post"])
        self.assertIn("/v1/flagship-handoff", payload["post"])
        self.assertIn("/v1/intelligence-review/prepare", payload["post"])
        self.assertIn("/v1/intelligence-review/status", payload["post"])
        self.assertIn("/v1/intelligence-review/export-csv", payload["post"])
        self.assertIn("/v1/intelligence-review/import-csv", payload["post"])

    def test_favicon_route_returns_empty_success(self):
        handler = object.__new__(CanonHandler)
        handler.path = "/favicon.ico"
        handler.send_empty = Mock()

        handler.do_GET()

        handler.send_empty.assert_called_once()

    def test_routes_endpoint_returns_route_metadata(self):
        handler = object.__new__(CanonHandler)
        handler.path = "/v1/routes"
        handler.send_json = Mock()

        handler.do_GET()

        payload = handler.send_json.call_args.args[0]
        self.assertIn("human_review_boundary", payload)
        paths = {route["path"]: route for route in payload["routes"]}
        self.assertEqual(paths["/v1/flagship-handoff"]["method"], "POST")
        self.assertEqual(paths["/v1/frame-coverage"]["method"], "POST")
        self.assertIn("research_frame", paths["/v1/frame-coverage"]["required"])
        self.assertIn("mode", paths["/v1/flagship-handoff"]["optional"])
        self.assertEqual(paths["/v1/intelligence-review/import-csv"]["example"]["csv_path"], "reports/intelligence_brief_review_tasks_ai_infra_geo_risk_demo.review.csv")

    def test_api_routes_include_examples_for_public_post_routes(self):
        metadata = api_routes()
        missing_examples = [
            route["path"]
            for route in metadata["routes"]
            if route["method"] == "POST" and route["example"] is None
        ]
        self.assertEqual(missing_examples, [])

    def test_post_routes_include_integration_endpoints(self):
        routes = {
            "/v1/sources/profile": ("source_profile", {"source_shape": "document_file"}),
            "/v1/sources/ingest": ("source_ingest", {"mode": "m"}),
            "/v1/corpora/build": ("corpus_build", {"corpus": {"corpus_id": "c"}}),
            "/v1/model-evaluation": ("model_evaluation", {"report_id": "semantic_model_evaluation_v1"}),
            "/v1/evidence-packets": ("evidence_packets", {"status": "complete"}),
            "/v1/frame-coverage": ("frame_coverage_report", {"status": "coverage_gap_human_review_required"}),
            "/v1/intelligence-brief": ("intelligence_brief", {"status": "ready_for_human_review"}),
            "/v1/intelligence-brief/evaluate": ("intelligence_brief_evaluation", {"status": "pass"}),
            "/v1/alert-digest": ("alert_digest", {"status": "ready_for_human_review"}),
            "/v1/alert-digest/evaluate": ("alert_digest_evaluation", {"status": "pass"}),
            "/v1/flagship-handoff": ("flagship_handoff", {"status": "automated_pass_human_review_required"}),
            "/v1/intelligence-review/prepare": ("intelligence_review_prepare", {"report_id": "intelligence_brief_review_tasks_v1"}),
            "/v1/intelligence-review/status": ("intelligence_review_status", {"status": "incomplete"}),
            "/v1/intelligence-review/export-csv": ("intelligence_review_export_csv", {"status": "review_csv_written"}),
            "/v1/intelligence-review/import-csv": ("intelligence_review_import_csv", {"status": "imported"}),
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
