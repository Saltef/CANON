import asyncio
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


if __name__ == "__main__":
    unittest.main()
