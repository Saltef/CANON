import asyncio
import base64
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from canon.observability import close_json_logging
from canon.product.asgi import ConcurrencyGate, QueueRejected, create_app


class ProductAsgiTests(unittest.TestCase):
    def test_concurrency_gate_rejects_when_queue_is_full(self):
        async def scenario():
            gate = ConcurrencyGate(max_concurrency=1, max_queue_depth=0)
            async with gate.acquire():
                with self.assertRaises(QueueRejected):
                    async with gate.acquire():
                        pass

        asyncio.run(scenario())

    def test_workbench_route_attaches_request_id_and_logs_jsonl(self):
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            self.skipTest("fastapi serve extra is not installed")

        with TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / "ops.jsonl"
            try:
                app = create_app(max_concurrency=1, max_queue_depth=0, log_path=log_path)
                with patch(
                    "canon.product.asgi.service.production_evidence_workbench",
                    return_value={
                        "report_id": "production_evidence_workbench_session_v1",
                        "status": "ready_for_user_inspection",
                        "retrieval": {"degradation_flags": []},
                        "evidence_cards": [{"evidence_id": "C1"}],
                        "draft_brief": {"status": "evidence_note_ready"},
                    },
                ):
                    with TestClient(app) as client:
                        response = client.post(
                            "/v1/production/evidence-workbench",
                            json={"query": "grid risk"},
                            headers={"X-Request-ID": "req-asgi-test"},
                        )

                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()["request_id"], "req-asgi-test")
            finally:
                close_json_logging(log_path)
            rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]

        self.assertTrue(rows)
        self.assertTrue(all(row["request_id"] == "req-asgi-test" for row in rows))
        self.assertIn("request_start", {row["event"] for row in rows})
        self.assertIn("request_complete", {row["event"] for row in rows})

    def test_asgi_exposes_corpus_setup_refresh_and_feedback_routes(self):
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            self.skipTest("fastapi serve extra is not installed")

        with TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / "ops.jsonl"
            try:
                app = create_app(max_concurrency=1, max_queue_depth=1, log_path=log_path)
                with TestClient(app) as client:
                    with patch(
                        "canon.product.asgi.service.production_corpus_setup",
                        return_value={"status": "corpus_ready_vector_indexed"},
                    ) as corpus_setup:
                        corpus = client.post(
                            "/v1/production/corpus-setup",
                            json={"input_path": "data/my_docs", "mode": "m"},
                        )
                    with patch(
                        "canon.product.asgi.service.production_corpus_refresh",
                        return_value={"status": "no_source_changes"},
                    ) as corpus_refresh:
                        refresh = client.post(
                            "/v1/production/corpus-refresh",
                            json={"input_path": "data/my_docs", "mode": "m"},
                        )
                    with patch(
                        "canon.product.asgi.service.production_feedback",
                        return_value={"status": "feedback_recorded"},
                    ) as feedback:
                        feedback_response = client.post(
                            "/v1/production/feedback",
                            json={"session_id": "s", "feedback_type": "useful"},
                        )
            finally:
                close_json_logging(log_path)

        self.assertEqual(corpus.status_code, 200)
        self.assertEqual(corpus.json()["status"], "corpus_ready_vector_indexed")
        self.assertEqual(refresh.status_code, 200)
        self.assertEqual(refresh.json()["status"], "no_source_changes")
        self.assertEqual(feedback_response.status_code, 200)
        self.assertEqual(feedback_response.json()["status"], "feedback_recorded")
        corpus_setup.assert_called_once()
        corpus_refresh.assert_called_once()
        feedback.assert_called_once()

    def test_asgi_exposes_route_metadata(self):
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            self.skipTest("fastapi serve extra is not installed")

        with TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / "ops.jsonl"
            try:
                app = create_app(max_concurrency=1, max_queue_depth=1, log_path=log_path)
                with TestClient(app) as client:
                    response = client.get("/v1/routes")
            finally:
                close_json_logging(log_path)

        self.assertEqual(response.status_code, 200)
        paths = {route["path"] for route in response.json()["routes"]}
        self.assertIn("/v1/production/corpus-refresh", paths)

    def test_auth_keeps_health_public_and_protects_app(self):
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            self.skipTest("fastapi serve extra is not installed")

        with TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / "ops.jsonl"
            try:
                app = create_app(
                    max_concurrency=1,
                    max_queue_depth=1,
                    log_path=log_path,
                    basic_auth_user="user",
                    basic_auth_password="pass",
                )
                with TestClient(app) as client:
                    health = client.get("/health")
                    blocked = client.get("/app")
                    token = base64.b64encode(b"user:pass").decode("ascii")
                    allowed = client.get("/app", headers={"Authorization": f"Basic {token}"})
            finally:
                close_json_logging(log_path)

        self.assertEqual(health.status_code, 200)
        self.assertEqual(blocked.status_code, 401)
        self.assertEqual(blocked.headers["www-authenticate"], "Basic")
        self.assertEqual(allowed.status_code, 200)

    def test_bearer_api_key_auth_allows_api_requests(self):
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            self.skipTest("fastapi serve extra is not installed")

        with TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / "ops.jsonl"
            try:
                app = create_app(max_concurrency=1, max_queue_depth=1, log_path=log_path, api_key="secret")
                with patch(
                    "canon.product.asgi.service.production_feedback",
                    return_value={"status": "feedback_recorded"},
                ):
                    with TestClient(app) as client:
                        blocked = client.post(
                            "/v1/production/feedback",
                            json={"session_id": "s", "feedback_type": "useful"},
                        )
                        allowed = client.post(
                            "/v1/production/feedback",
                            json={"session_id": "s", "feedback_type": "useful"},
                            headers={"Authorization": "Bearer secret"},
                        )
            finally:
                close_json_logging(log_path)

        self.assertEqual(blocked.status_code, 401)
        self.assertEqual(allowed.status_code, 200)

    def test_partial_basic_auth_configuration_fails_fast(self):
        with TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / "ops.jsonl"
            try:
                with self.assertRaises(ValueError):
                    create_app(
                        max_concurrency=1,
                        max_queue_depth=1,
                        log_path=log_path,
                        basic_auth_user="user",
                    )
            finally:
                close_json_logging(log_path)

    def test_required_auth_fails_fast_without_credentials(self):
        with TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / "ops.jsonl"
            try:
                with self.assertRaises(ValueError):
                    create_app(
                        max_concurrency=1,
                        max_queue_depth=1,
                        log_path=log_path,
                        require_auth=True,
                    )
            finally:
                close_json_logging(log_path)

    def test_required_auth_accepts_api_key_credentials(self):
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            self.skipTest("fastapi serve extra is not installed")

        with TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / "ops.jsonl"
            try:
                app = create_app(
                    max_concurrency=1,
                    max_queue_depth=1,
                    log_path=log_path,
                    api_key="secret",
                    require_auth=True,
                )
                with TestClient(app) as client:
                    blocked = client.get("/app")
                    allowed = client.get("/app", headers={"X-CANON-API-Key": "secret"})
            finally:
                close_json_logging(log_path)

        self.assertEqual(blocked.status_code, 401)
        self.assertEqual(allowed.status_code, 200)

    def test_cors_allowlist_adds_origin_header(self):
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            self.skipTest("fastapi serve extra is not installed")

        with TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / "ops.jsonl"
            try:
                app = create_app(
                    max_concurrency=1,
                    max_queue_depth=1,
                    log_path=log_path,
                    allowed_origins=["https://example.github.io"],
                )
                with TestClient(app) as client:
                    response = client.get("/health", headers={"Origin": "https://example.github.io"})
            finally:
                close_json_logging(log_path)

        self.assertEqual(response.headers["access-control-allow-origin"], "https://example.github.io")


if __name__ == "__main__":
    unittest.main()
