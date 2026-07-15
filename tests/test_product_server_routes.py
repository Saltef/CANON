import unittest
from unittest.mock import Mock, patch

from canon.product.server import CanonHandler


class ProductServerRouteTests(unittest.TestCase):
    def test_root_route_returns_api_index(self):
        handler = object.__new__(CanonHandler)
        handler.path = "/"
        handler.send_json = Mock()

        handler.do_GET()

        payload = handler.send_json.call_args.args[0]
        self.assertEqual(payload["service"], "canon")
        self.assertIn("/v1/evidence-packets", payload["post"])
        self.assertIn("/v1/intelligence-brief", payload["post"])
        self.assertIn("/v1/alert-digest", payload["post"])
        self.assertIn("/v1/flagship-handoff", payload["post"])

    def test_favicon_route_returns_empty_success(self):
        handler = object.__new__(CanonHandler)
        handler.path = "/favicon.ico"
        handler.send_empty = Mock()

        handler.do_GET()

        handler.send_empty.assert_called_once()

    def test_post_routes_include_integration_endpoints(self):
        routes = {
            "/v1/sources/profile": ("source_profile", {"source_shape": "document_file"}),
            "/v1/sources/ingest": ("source_ingest", {"mode": "m"}),
            "/v1/corpora/build": ("corpus_build", {"corpus": {"corpus_id": "c"}}),
            "/v1/model-evaluation": ("model_evaluation", {"report_id": "semantic_model_evaluation_v1"}),
            "/v1/evidence-packets": ("evidence_packets", {"status": "complete"}),
            "/v1/intelligence-brief": ("intelligence_brief", {"status": "ready_for_human_review"}),
            "/v1/alert-digest": ("alert_digest", {"status": "ready_for_human_review"}),
            "/v1/flagship-handoff": ("flagship_handoff", {"status": "automated_pass_human_review_required"}),
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
