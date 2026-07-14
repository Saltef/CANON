import unittest
from unittest.mock import Mock, patch

from canon.product.server import CanonHandler


class ProductServerRouteTests(unittest.TestCase):
    def test_post_routes_include_integration_endpoints(self):
        routes = {
            "/v1/sources/profile": ("source_profile", {"source_shape": "document_file"}),
            "/v1/sources/ingest": ("source_ingest", {"mode": "m"}),
            "/v1/corpora/build": ("corpus_build", {"corpus": {"corpus_id": "c"}}),
            "/v1/model-evaluation": ("model_evaluation", {"report_id": "semantic_model_evaluation_v1"}),
            "/v1/evidence-packets": ("evidence_packets", {"status": "complete"}),
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
