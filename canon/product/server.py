from __future__ import annotations

import argparse
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from canon.product import service


class CanonHandler(BaseHTTPRequestHandler):
    server_version = "CanonProductHTTP/0.1"

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        mode = first(query.get("mode")) or service.DEFAULT_MODE
        params = first_values(query)
        params["mode"] = mode
        try:
            if parsed.path == "/health":
                self.send_json(service.health())
            elif parsed.path == "/v1/summary":
                self.send_json(service.product_summary(mode))
            elif parsed.path == "/v1/diversity/queries":
                params["mode"] = first(query.get("mode")) or service.DEFAULT_DIVERSITY_MODE
                self.send_json(service.diversity_queries(params))
            elif parsed.path.startswith("/v1/diversity/queries/"):
                params["mode"] = first(query.get("mode")) or service.DEFAULT_DIVERSITY_MODE
                query_id = parsed.path.removeprefix("/v1/diversity/queries/")
                self.send_json(service.diversity_query_detail(query_id, params))
            elif parsed.path.startswith("/v1/reports/"):
                name = parsed.path.removeprefix("/v1/reports/")
                self.send_json(service.report(name, mode, params))
            else:
                self.send_json({"error": "not_found"}, HTTPStatus.NOT_FOUND)
        except service.ProductError as exc:
            self.send_json({"error": str(exc)}, exc.status_code)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            payload = self.read_json()
            if parsed.path == "/v1/answer":
                self.send_json(service.answer(payload))
            elif parsed.path == "/v1/evidence-packets":
                self.send_json(service.evidence_packets(payload))
            elif parsed.path == "/v1/compare":
                self.send_json(service.compare_retrieval(payload))
            elif parsed.path == "/v1/query-diagnostics":
                self.send_json(service.query_diagnostics(payload))
            elif parsed.path == "/v1/sources/profile":
                self.send_json(service.source_profile(payload))
            elif parsed.path == "/v1/sources/ingest":
                self.send_json(service.source_ingest(payload))
            elif parsed.path == "/v1/corpora/build":
                self.send_json(service.corpus_build(payload))
            elif parsed.path == "/v1/model-evaluation":
                self.send_json(service.model_evaluation(payload))
            elif parsed.path == "/v1/diversity-audit":
                self.send_json(service.diversity_audit(payload))
            else:
                self.send_json({"error": "not_found"}, HTTPStatus.NOT_FOUND)
        except service.ProductError as exc:
            self.send_json({"error": str(exc)}, exc.status_code)
        except json.JSONDecodeError:
            self.send_json({"error": "invalid_json"}, HTTPStatus.BAD_REQUEST)

    def read_json(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        raw = self.rfile.read(length).decode("utf-8")
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise service.ProductError("JSON body must be an object.")
        return payload

    def send_json(self, payload: object, status: int | HTTPStatus = HTTPStatus.OK) -> None:
        data = json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8")
        self.send_response(int(status))
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args) -> None:
        return


def first(values: list[str] | None) -> str | None:
    if not values:
        return None
    return values[0]


def first_values(query: dict[str, list[str]]) -> dict[str, str]:
    return {
        key: value[0]
        for key, value in query.items()
        if value
    }


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    server = ThreadingHTTPServer((host, port), CanonHandler)
    print(f"CANON product API listening on http://{host}:{port}", flush=True)
    server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the CANON product API.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    run(args.host, args.port)


if __name__ == "__main__":
    main()
