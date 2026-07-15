from __future__ import annotations

import argparse
import json
import sys
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
            if parsed.path == "/":
                self.send_json(api_index())
            elif parsed.path == "/favicon.ico":
                self.send_empty(HTTPStatus.NO_CONTENT)
            elif parsed.path == "/health":
                self.send_json(service.health())
            elif parsed.path == "/v1/summary":
                self.send_json(service.product_summary(mode))
            elif parsed.path == "/v1/routes":
                self.send_json(api_routes())
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
            elif parsed.path == "/v1/projects/start":
                self.send_json(service.start_project(payload))
            elif parsed.path == "/v1/evidence-packets":
                self.send_json(service.evidence_packets(payload))
            elif parsed.path == "/v1/frame-coverage":
                self.send_json(service.frame_coverage_report(payload))
            elif parsed.path == "/v1/intelligence-brief":
                self.send_json(service.intelligence_brief(payload))
            elif parsed.path == "/v1/report-quality":
                self.send_json(service.report_quality_gate(payload))
            elif parsed.path == "/v1/intelligence-brief/evaluate":
                self.send_json(service.intelligence_brief_evaluation(payload))
            elif parsed.path == "/v1/alert-digest":
                self.send_json(service.alert_digest(payload))
            elif parsed.path == "/v1/alert-digest/evaluate":
                self.send_json(service.alert_digest_evaluation(payload))
            elif parsed.path == "/v1/flagship-handoff":
                self.send_json(service.flagship_handoff(payload))
            elif parsed.path == "/v1/intelligence-review/prepare":
                self.send_json(service.intelligence_review_prepare(payload))
            elif parsed.path == "/v1/intelligence-review/status":
                self.send_json(service.intelligence_review_status(payload))
            elif parsed.path == "/v1/intelligence-review/export-csv":
                self.send_json(service.intelligence_review_export_csv(payload))
            elif parsed.path == "/v1/intelligence-review/import-csv":
                self.send_json(service.intelligence_review_import_csv(payload))
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

    def send_empty(self, status: int | HTTPStatus = HTTPStatus.NO_CONTENT) -> None:
        self.send_response(int(status))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, format: str, *args) -> None:
        return


class CanonHTTPServer(ThreadingHTTPServer):
    def handle_error(self, request, client_address) -> None:
        exception = sys.exc_info()[1]
        if isinstance(exception, (ConnectionResetError, ConnectionAbortedError)):
            return
        return super().handle_error(request, client_address)


def api_index() -> dict:
    routes = api_routes()
    return {
        "service": "canon",
        "status": "ok",
        "message": "CANON product API is running. Use the listed routes; POST routes require JSON bodies.",
        "get": [
            "/health",
            "/v1/routes",
            "/v1/summary",
            "/v1/reports/audit",
            "/v1/reports/claim-decision",
            "/v1/reports/data-card",
            "/v1/reports/diversity",
            "/v1/reports/diversity-gate",
            "/v1/diversity/queries",
            "/v1/diversity/queries/{query_id}",
            "/v1/reports/regression-gate",
        ],
        "post": [
            "/v1/answer",
            "/v1/projects/start",
            "/v1/evidence-packets",
            "/v1/frame-coverage",
            "/v1/intelligence-brief",
            "/v1/report-quality",
            "/v1/intelligence-brief/evaluate",
            "/v1/alert-digest",
            "/v1/alert-digest/evaluate",
            "/v1/flagship-handoff",
            "/v1/intelligence-review/prepare",
            "/v1/intelligence-review/status",
            "/v1/intelligence-review/export-csv",
            "/v1/intelligence-review/import-csv",
            "/v1/compare",
            "/v1/query-diagnostics",
            "/v1/sources/profile",
            "/v1/sources/ingest",
            "/v1/corpora/build",
            "/v1/model-evaluation",
            "/v1/diversity-audit",
        ],
        "examples": {
            "health": "http://127.0.0.1:8000/health",
            "routes": "http://127.0.0.1:8000/v1/routes",
            "summary": "http://127.0.0.1:8000/v1/summary",
        },
        "route_count": len(routes["routes"]),
    }


def api_routes() -> dict:
    return {
        "service": "canon",
        "version": "v1",
        "human_review_boundary": (
            "Automated routes prepare evidence, reports, and gates for review. "
            "They do not establish final factual correctness or release-level model quality."
        ),
        "routes": [
            route("GET", "/health", "Check API health.", [], [], None),
            route("GET", "/v1/routes", "Return machine-readable route metadata and example payloads.", [], [], None),
            route("GET", "/v1/summary", "Return compact product and corpus summary.", [], ["mode"], None),
            route("GET", "/v1/reports/audit", "Return the scientific audit report for a mode.", [], ["mode"], None),
            route("GET", "/v1/reports/claim-decision", "Return claim-boundary decision metadata for a mode.", [], ["mode"], None),
            route("GET", "/v1/reports/data-card", "Return the corpus data card for a mode.", [], ["mode"], None),
            route("GET", "/v1/reports/diversity", "Return compact source-diversity audit results.", [], ["mode", "diverse_method_id", "baseline_method_id"], None),
            route("GET", "/v1/reports/diversity-gate", "Return source-diversity gate status.", [], ["mode", "diverse_method_id", "baseline_method_id"], None),
            route("GET", "/v1/diversity/queries", "List source-diversity query diagnostics.", [], ["mode", "verdict", "query_type", "limit"], None),
            route("GET", "/v1/diversity/queries/{query_id}", "Return one source-diversity query diagnostic.", ["query_id"], ["mode"], None),
            route("GET", "/v1/reports/regression-gate", "Return regression-gate report for a mode.", [], ["mode"], None),
            route(
                "POST",
                "/v1/projects/start",
                "Create a scoped intelligence project config with ontology, source plan, and monitor boundary.",
                ["project_name", "domain", "regions", "languages", "issue_categories", "desired_report_types"],
                ["project_id", "review_cadence", "source_boundaries", "corpus_id", "write_report"],
                {
                    "project_name": "AI Infrastructure Geopolitical Risk",
                    "domain": "AI infrastructure and geopolitical risk",
                    "regions": ["Latin America", "Brazil", "Chile", "Mexico"],
                    "languages": ["English", "Spanish", "Portuguese"],
                    "issue_categories": ["energy demand", "water and cooling", "cloud dependency"],
                    "desired_report_types": ["weekly_intelligence_brief", "alert_digest"],
                    "source_boundaries": ["G:/My Drive/CANON Corpus"],
                },
            ),
            route(
                "POST",
                "/v1/sources/profile",
                "Profile a local folder, mounted Drive folder, or git checkout before ingest.",
                ["input_path"],
                ["format", "sample_size"],
                {"input_path": "data/my_docs", "sample_size": 25},
            ),
            route(
                "POST",
                "/v1/sources/ingest",
                "Ingest a profiled local source into a named mode.",
                ["input_path", "mode"],
                ["format", "domain", "provenance", "source_name", "chunk_tokens", "overlap_tokens"],
                {"input_path": "data/my_docs", "mode": "my_topic_v1", "chunk_tokens": 160},
            ),
            route(
                "POST",
                "/v1/evidence-packets",
                "Generate cited evidence packets for downstream review or intelligence runs.",
                ["query or question"],
                ["mode", "policy", "project_id", "research_frame", "evidence_requirements", "top_k"],
                {
                    "question": "What does the corpus say about grid risk?",
                    "mode": "my_topic_v1_corpus",
                    "evidence_requirements": {"top_k": 10},
                },
            ),
            route(
                "POST",
                "/v1/frame-coverage",
                "Run a focused research-frame coverage diagnostic over retrieved evidence.",
                ["query or question", "research_frame"],
                ["mode", "policy", "project_id", "evidence_requirements", "top_k", "write_report"],
                {
                    "question": "What does the corpus say about grid risk?",
                    "mode": "my_topic_v1_corpus",
                    "research_frame": {
                        "subdomains": ["energy", "water"],
                        "regions": ["Latin America"],
                        "languages": ["English", "Spanish"],
                    },
                    "evidence_requirements": {"top_k": 10, "minimum_source_types": ["official", "local_media"]},
                },
            ),
            route(
                "POST",
                "/v1/intelligence-brief",
                "Generate a grounded intelligence brief from evidence packets.",
                ["query or question"],
                ["mode", "policy", "project_id", "write_report"],
                {"query": "What are the emerging geopolitical risks around AI data center expansion in Latin America?", "mode": "ai_infra_geo_risk_demo", "write_report": True},
            ),
            route(
                "POST",
                "/v1/report-quality",
                "Run a focused grounding, citation, red-team, and structure gate for one brief.",
                ["query or question"],
                ["mode", "policy", "project_id", "write_report"],
                {
                    "query": "What are the emerging geopolitical risks around AI data center expansion in Latin America?",
                    "mode": "ai_infra_geo_risk_demo",
                    "write_report": True,
                },
            ),
            route(
                "POST",
                "/v1/intelligence-brief/evaluate",
                "Run automated grounding and report-structure gates over seed questions.",
                [],
                ["mode", "queries_path", "policy", "project_id", "write_report"],
                {"mode": "ai_infra_geo_risk_demo", "queries_path": "gold/ai_infra_geo_risk_seed_queries.json"},
            ),
            route(
                "POST",
                "/v1/alert-digest",
                "Generate evidence-triggered alert review prompts.",
                ["query or question"],
                ["mode", "policy", "project_id", "write_report"],
                {"query": "What are the emerging geopolitical risks around AI data center expansion in Latin America?", "mode": "ai_infra_geo_risk_demo"},
            ),
            route(
                "POST",
                "/v1/alert-digest/evaluate",
                "Run automated alert structure and duplicate-rate gates over seed questions.",
                [],
                ["mode", "queries_path", "policy", "project_id", "write_report"],
                {"mode": "ai_infra_geo_risk_demo", "queries_path": "gold/ai_infra_geo_risk_seed_queries.json"},
            ),
            route(
                "POST",
                "/v1/flagship-handoff",
                "Run the built-in fixture workflow and return automated gate plus human-review status.",
                [],
                ["mode", "question", "fixture_path", "queries_path", "policy", "project_id", "write_report"],
                {"mode": "ai_infra_geo_risk_demo", "write_report": True},
            ),
            route(
                "POST",
                "/v1/intelligence-review/prepare",
                "Prepare human review tasks for intelligence brief labels.",
                [],
                ["mode", "queries_path", "policy", "project_id", "records_path", "reset_review_labels", "write_report"],
                {"mode": "ai_infra_geo_risk_demo", "queries_path": "gold/ai_infra_geo_risk_seed_queries.json"},
            ),
            route(
                "POST",
                "/v1/intelligence-review/export-csv",
                "Export review tasks to CSV for human labels.",
                ["records_path or mode"],
                ["output_path"],
                {"records_path": "reports/intelligence_brief_review_tasks_ai_infra_geo_risk_demo.json"},
            ),
            route(
                "POST",
                "/v1/intelligence-review/import-csv",
                "Import and validate human-entered review labels.",
                ["records_path or mode", "csv_path"],
                ["output_path"],
                {
                    "records_path": "reports/intelligence_brief_review_tasks_ai_infra_geo_risk_demo.json",
                    "csv_path": "reports/intelligence_brief_review_tasks_ai_infra_geo_risk_demo.review.csv",
                },
            ),
            route(
                "POST",
                "/v1/intelligence-review/status",
                "Check whether required human review fields are complete and valid.",
                ["records_path or mode"],
                ["write_report"],
                {"records_path": "reports/intelligence_brief_review_tasks_ai_infra_geo_risk_demo.completed.json"},
            ),
            route(
                "POST",
                "/v1/answer",
                "Generate a cited answer from a named corpus.",
                ["query"],
                ["mode", "policy", "top_k", "freedom_level"],
                {"query": "What does the corpus say about grid risk?", "mode": "my_topic_v1_corpus", "policy": "rag"},
            ),
            route(
                "POST",
                "/v1/compare",
                "Compare retrieval policies for one query.",
                ["query"],
                ["mode", "policies", "top_k"],
                {"query": "What does the corpus say about grid risk?", "mode": "my_topic_v1_corpus", "policies": ["lexical", "rag"]},
            ),
            route(
                "POST",
                "/v1/query-diagnostics",
                "Diagnose query-corpus term match, drift risk, and candidate variants.",
                ["query"],
                ["mode", "top_k", "candidate_k", "freedom_level"],
                {"query": "grid risk", "mode": "my_topic_v1_corpus", "freedom_level": "balanced"},
            ),
            route(
                "POST",
                "/v1/corpora/build",
                "Build a named corpus from one or more ingested modes.",
                ["corpus_id", "from_modes"],
                ["top_k", "policies", "corpus_only"],
                {"corpus_id": "my_topic_v1_corpus", "from_modes": ["my_topic_v1"], "corpus_only": True},
            ),
            route(
                "POST",
                "/v1/model-evaluation",
                "Evaluate retrieval models against reviewed qrels or query fixtures.",
                ["mode", "qrels_path or queries_path"],
                ["providers", "k", "batch_size"],
                {"mode": "my_topic_v1_corpus", "qrels_path": "gold/my_topic_qrels.json", "providers": ["local"]},
            ),
            route(
                "POST",
                "/v1/diversity-audit",
                "Run source-diversity audit for a mode.",
                [],
                ["mode", "diverse_method_id", "baseline_method_id"],
                {"mode": "social_science_ir_10k"},
            ),
        ],
    }


def route(
    method: str,
    path: str,
    purpose: str,
    required: list[str],
    optional: list[str],
    example: dict | None,
) -> dict:
    return {
        "method": method,
        "path": path,
        "purpose": purpose,
        "required": required,
        "optional": optional,
        "example": example,
    }


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
    server = CanonHTTPServer((host, port), CanonHandler)
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
