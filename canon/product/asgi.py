import argparse
import asyncio
import base64
import hmac
import os
import time
import uuid
from contextlib import asynccontextmanager
from http import HTTPStatus
from pathlib import Path
from typing import Any

from canon.config import load_settings
from canon.observability import configure_json_logging, log_event, reset_request_id, set_request_id
from canon.product import service, workbench_app
from canon.secrets import load_local_env


class QueueRejected(RuntimeError):
    pass


class ConcurrencyGate:
    def __init__(self, max_concurrency: int = 8, max_queue_depth: int = 16) -> None:
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be at least 1.")
        if max_queue_depth < 0:
            raise ValueError("max_queue_depth must be non-negative.")
        self.max_concurrency = max_concurrency
        self.max_queue_depth = max_queue_depth
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._lock = asyncio.Lock()
        self.active = 0
        self.waiting = 0

    @asynccontextmanager
    async def acquire(self):
        async with self._lock:
            if self.waiting >= self.max_queue_depth and self._semaphore.locked():
                raise QueueRejected("queue_depth_exceeded")
            self.waiting += 1
        await self._semaphore.acquire()
        async with self._lock:
            self.waiting -= 1
            self.active += 1
        try:
            yield self.snapshot()
        finally:
            async with self._lock:
                self.active -= 1
            self._semaphore.release()

    def snapshot(self) -> dict[str, int]:
        return {
            "active": self.active,
            "waiting": self.waiting,
            "max_concurrency": self.max_concurrency,
            "max_queue_depth": self.max_queue_depth,
        }


def create_app(
    *,
    max_concurrency: int = 8,
    max_queue_depth: int = 16,
    retry_after_s: int = 2,
    log_path: Path | None = None,
    allowed_origins: list[str] | None = None,
    api_key: str | None = None,
    basic_auth_user: str | None = None,
    basic_auth_password: str | None = None,
    require_auth: bool | None = None,
):
    try:
        from fastapi import Body, FastAPI, Request
        from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, Response
        from starlette.concurrency import run_in_threadpool
    except ImportError as exc:
        raise RuntimeError('Install the serve extra first: pip install -e ".[serve]"') from exc

    load_local_env()
    settings = load_settings()
    configure_json_logging(log_path or settings.reports_dir / "asgi_operational_v1.jsonl")
    configure_otel()
    metrics = Metrics()
    gate = ConcurrencyGate(max_concurrency=max_concurrency, max_queue_depth=max_queue_depth)
    app = FastAPI(title="CANON Evidence Discovery API", version="0.1.0")
    configure_cors(app, allowed_origins if allowed_origins is not None else env_list("CANON_ALLOWED_ORIGINS"))
    configure_auth(
        app,
        api_key=api_key if api_key is not None else os.getenv("CANON_API_KEY"),
        basic_auth_user=basic_auth_user if basic_auth_user is not None else os.getenv("CANON_BASIC_AUTH_USER"),
        basic_auth_password=basic_auth_password
        if basic_auth_password is not None
        else os.getenv("CANON_BASIC_AUTH_PASSWORD"),
        require_auth=require_auth if require_auth is not None else env_bool("CANON_REQUIRE_AUTH", default=False),
    )
    instrument_fastapi(app)

    @app.get("/health")
    async def health():
        return service.health() | {"server": "asgi", "queue": gate.snapshot()}

    @app.get("/app", response_class=HTMLResponse)
    async def app_html():
        return HTMLResponse(workbench_app.render_app())

    @app.get("/v1/production/status")
    async def production_status(mode: str | None = None):
        return service.production_status({"mode": mode} if mode else {})

    @app.get("/v1/routes")
    async def routes():
        return asgi_routes()

    @app.post("/v1/evidence-packets")
    async def evidence_packets(request: Request, payload: dict[str, Any] = Body(...)):
        return await run_bounded(
            payload=payload,
            request=request,
            gate=gate,
            retry_after_s=retry_after_s,
            metrics=metrics,
            run_in_threadpool=run_in_threadpool,
            handler=service.evidence_packets,
            route="/v1/evidence-packets",
        )

    @app.post("/v1/production/evidence-workbench")
    async def production_workbench(request: Request, payload: dict[str, Any] = Body(...)):
        return await run_bounded(
            payload=payload,
            request=request,
            gate=gate,
            retry_after_s=retry_after_s,
            metrics=metrics,
            run_in_threadpool=run_in_threadpool,
            handler=service.production_evidence_workbench,
            route="/v1/production/evidence-workbench",
        )

    @app.post("/v1/production/corpus-setup")
    async def production_corpus_setup(request: Request, payload: dict[str, Any] = Body(...)):
        return await run_bounded(
            payload=payload,
            request=request,
            gate=gate,
            retry_after_s=retry_after_s,
            metrics=metrics,
            run_in_threadpool=run_in_threadpool,
            handler=service.production_corpus_setup,
            route="/v1/production/corpus-setup",
        )

    @app.post("/v1/production/corpus-refresh")
    async def production_corpus_refresh(request: Request, payload: dict[str, Any] = Body(...)):
        return await run_bounded(
            payload=payload,
            request=request,
            gate=gate,
            retry_after_s=retry_after_s,
            metrics=metrics,
            run_in_threadpool=run_in_threadpool,
            handler=service.production_corpus_refresh,
            route="/v1/production/corpus-refresh",
        )

    @app.post("/v1/production/feedback")
    async def production_feedback(request: Request, payload: dict[str, Any] = Body(...)):
        return await run_bounded(
            payload=payload,
            request=request,
            gate=gate,
            retry_after_s=retry_after_s,
            metrics=metrics,
            run_in_threadpool=run_in_threadpool,
            handler=service.production_feedback,
            route="/v1/production/feedback",
        )

    @app.post("/v1/stage2-synthesis")
    async def stage2_synthesis(request: Request, payload: dict[str, Any] = Body(...)):
        return await run_bounded(
            payload=payload,
            request=request,
            gate=gate,
            retry_after_s=retry_after_s,
            metrics=metrics,
            run_in_threadpool=run_in_threadpool,
            handler=service.stage2_synthesis,
            route="/v1/stage2-synthesis",
        )

    @app.get("/metrics")
    async def prometheus_metrics():
        body, content_type = metrics.render()
        if body is None:
            return PlainTextResponse("prometheus-client not installed\n", status_code=HTTPStatus.NOT_IMPLEMENTED)
        return Response(body, media_type=content_type)

    @app.get("/")
    async def root():
        return {
            "service": "canon",
            "server": "asgi",
            "routes": [
                "GET /health",
                "GET /app",
                "GET /metrics",
                "GET /v1/production/status",
                "GET /v1/routes",
                "POST /v1/evidence-packets",
                "POST /v1/production/evidence-workbench",
                "POST /v1/production/corpus-setup",
                "POST /v1/production/corpus-refresh",
                "POST /v1/production/feedback",
                "POST /v1/stage2-synthesis",
            ],
        }

    app.state.canon_gate = gate
    app.state.canon_metrics = metrics
    return app


def asgi_routes() -> dict[str, Any]:
    return {
        "routes": [
            {"method": "GET", "path": "/health", "description": "Health check for hosting platforms."},
            {"method": "GET", "path": "/app", "description": "Evidence Discovery Workbench UI."},
            {"method": "GET", "path": "/metrics", "description": "Prometheus metrics when installed."},
            {
                "method": "GET",
                "path": "/v1/production/status",
                "description": "Production defaults, corpus status, and recommended retrieval settings.",
            },
            {"method": "GET", "path": "/v1/routes", "description": "ASGI route metadata."},
            {"method": "POST", "path": "/v1/evidence-packets", "description": "Evidence packet retrieval."},
            {
                "method": "POST",
                "path": "/v1/production/evidence-workbench",
                "description": "End-to-end retrieval, diagnostics, draft evidence note, and feedback metadata.",
            },
            {
                "method": "POST",
                "path": "/v1/production/corpus-setup",
                "description": "Profile, ingest, build, and optionally index a user corpus.",
            },
            {
                "method": "POST",
                "path": "/v1/production/corpus-refresh",
                "description": "Refresh a corpus and hosted vector index only when sources changed.",
            },
            {
                "method": "POST",
                "path": "/v1/production/feedback",
                "description": "Capture local product feedback; not a formal human-review label.",
            },
            {"method": "POST", "path": "/v1/stage2-synthesis", "description": "Grounded Stage 2 synthesis."},
        ]
    }


def configure_cors(app, allowed_origins: list[str]) -> None:
    if not allowed_origins:
        return
    try:
        from fastapi.middleware.cors import CORSMiddleware
    except ImportError:
        return
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials="*" not in allowed_origins,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-CANON-API-Key", "X-Request-ID"],
    )


def configure_auth(
    app,
    *,
    api_key: str | None,
    basic_auth_user: str | None,
    basic_auth_password: str | None,
    require_auth: bool = False,
) -> None:
    api_key = (api_key or "").strip()
    basic_auth_user = (basic_auth_user or "").strip()
    basic_auth_password = (basic_auth_password or "").strip()
    if bool(basic_auth_user) != bool(basic_auth_password):
        raise ValueError("Both CANON_BASIC_AUTH_USER and CANON_BASIC_AUTH_PASSWORD must be set for Basic Auth.")
    basic_enabled = bool(basic_auth_user and basic_auth_password)
    api_key_enabled = bool(api_key)
    if not basic_enabled and not api_key_enabled:
        if require_auth:
            raise ValueError(
                "CANON_REQUIRE_AUTH is true, but no CANON_API_KEY or complete Basic Auth credentials are set."
            )
        return

    @app.middleware("http")
    async def require_auth(request, call_next):
        if request.method.upper() == "OPTIONS" or request.url.path in {"/health"}:
            return await call_next(request)
        if request_is_authorized(
            request,
            api_key=api_key if api_key_enabled else None,
            basic_auth_user=basic_auth_user if basic_enabled else None,
            basic_auth_password=basic_auth_password if basic_enabled else None,
        ):
            return await call_next(request)
        from fastapi.responses import JSONResponse

        headers = {"WWW-Authenticate": "Basic"} if basic_enabled else {}
        return JSONResponse(
            {"error": "authentication_required", "request_id": str(uuid.uuid4())},
            status_code=HTTPStatus.UNAUTHORIZED,
            headers=headers,
        )


def request_is_authorized(
    request,
    *,
    api_key: str | None,
    basic_auth_user: str | None,
    basic_auth_password: str | None,
) -> bool:
    if api_key:
        supplied = request.headers.get("x-canon-api-key") or bearer_token(request.headers.get("authorization"))
        if supplied and hmac.compare_digest(supplied, api_key):
            return True
    if basic_auth_user and basic_auth_password:
        username, password = basic_credentials(request.headers.get("authorization"))
        if username and password:
            return hmac.compare_digest(username, basic_auth_user) and hmac.compare_digest(
                password,
                basic_auth_password,
            )
    return False


def bearer_token(header: str | None) -> str | None:
    if not header:
        return None
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer" or not token:
        return None
    return token.strip()


def basic_credentials(header: str | None) -> tuple[str | None, str | None]:
    if not header:
        return None, None
    scheme, _, payload = header.partition(" ")
    if scheme.lower() != "basic" or not payload:
        return None, None
    try:
        decoded = base64.b64decode(payload, validate=True).decode("utf-8")
    except (ValueError, UnicodeDecodeError):
        return None, None
    username, separator, password = decoded.partition(":")
    if not separator:
        return None, None
    return username, password


def env_list(key: str) -> list[str]:
    return [piece.strip() for piece in (os.getenv(key) or "").split(",") if piece.strip()]


def env_bool(key: str, default: bool = False) -> bool:
    value = os.getenv(key)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


async def run_bounded(
    *,
    payload: dict[str, Any],
    request,
    gate: ConcurrencyGate,
    retry_after_s: int,
    metrics,
    run_in_threadpool,
    handler,
    route: str,
):
    from fastapi.responses import JSONResponse

    request_id = request.headers.get("x-request-id") or str(payload.get("request_id") or uuid.uuid4())
    payload = dict(payload)
    payload.setdefault("request_id", request_id)
    token = set_request_id(request_id)
    started = time.perf_counter()
    try:
        try:
            async with gate.acquire() as queue:
                log_event("request_start", route=route, method=request.method, queue=queue)
                report = await run_in_threadpool(handler, payload)
        except QueueRejected:
            metrics.queue_rejections.inc()
            log_event("request_rejected", route=route, reason="queue_depth_exceeded")
            return JSONResponse(
                {"error": "queue_depth_exceeded", "request_id": request_id},
                status_code=HTTPStatus.SERVICE_UNAVAILABLE,
                headers={"Retry-After": str(retry_after_s)},
            )
        except service.ProductError as exc:
            metrics.requests.labels(route=route, status=str(exc.status_code)).inc()
            log_event("request_product_error", route=route, status_code=exc.status_code, error=str(exc))
            return JSONResponse({"error": str(exc), "request_id": request_id}, status_code=exc.status_code)

        elapsed = time.perf_counter() - started
        report.setdefault("request_id", request_id)
        metrics.observe_report(route, report, elapsed)
        log_event(
            "request_complete",
            route=route,
            status_code=200,
            elapsed_ms=round(elapsed * 1000, 3),
            report_status=report.get("status"),
            degradation_flags=(report.get("retrieval") or report).get("degradation_flags") or [],
        )
        return report
    finally:
        reset_request_id(token)


class Metrics:
    def __init__(self) -> None:
        try:
            from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest
            from prometheus_client.openmetrics.exposition import CONTENT_TYPE_LATEST
        except ImportError:
            self.enabled = False
            return
        self.enabled = True
        self.content_type = CONTENT_TYPE_LATEST
        self.generate_latest = generate_latest
        self.registry = CollectorRegistry()
        self.requests = Counter(
            "canon_requests_total",
            "CANON ASGI requests by route and status.",
            ["route", "status"],
            registry=self.registry,
        )
        self.latency = Histogram(
            "canon_request_latency_seconds",
            "CANON ASGI request latency.",
            ["route"],
            registry=self.registry,
        )
        self.queue_rejections = Counter(
            "canon_queue_rejections_total",
            "Requests rejected because queue depth was exceeded.",
            registry=self.registry,
        )
        self.zero_hit_queries = Counter(
            "canon_zero_hit_queries_total",
            "Requests that returned no evidence cards.",
            registry=self.registry,
        )
        self.rerank_degraded = Counter(
            "canon_rerank_degraded_total",
            "Requests degraded from rerank to RRF fallback.",
            registry=self.registry,
        )
        self.embed_degraded = Counter(
            "canon_embed_degraded_total",
            "Requests degraded from embedding retrieval to BM25-only.",
            registry=self.registry,
        )
        self.abstentions = Counter(
            "canon_abstentions_total",
            "Requests where drafting abstained or was blocked.",
            registry=self.registry,
        )

    def observe_report(self, route: str, report: dict[str, Any], elapsed: float) -> None:
        if not self.enabled:
            return
        self.requests.labels(route=route, status="200").inc()
        self.latency.labels(route=route).observe(elapsed)
        evidence_cards = report.get("evidence_cards") or []
        if not evidence_cards:
            self.zero_hit_queries.inc()
        retrieval = report.get("retrieval") or report
        flags = retrieval.get("degradation_flags") or []
        if any("rerank" in str(flag) for flag in flags):
            self.rerank_degraded.inc()
        if any("embed" in str(flag) or "bm25_only" in str(flag) for flag in flags):
            self.embed_degraded.inc()
        draft_status = (report.get("draft_brief") or {}).get("status")
        if draft_status in {"blocked_by_relevance_gate", "insufficient_cited_evidence"}:
            self.abstentions.inc()

    def render(self) -> tuple[bytes | None, str]:
        if not self.enabled:
            return None, "text/plain"
        return self.generate_latest(self.registry), self.content_type


def configure_otel() -> None:
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    except ImportError:
        return
    if not isinstance(trace.get_tracer_provider(), TracerProvider):
        provider = TracerProvider(resource=Resource.create({"service.name": "canon"}))
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
        trace.set_tracer_provider(provider)


def instrument_fastapi(app) -> None:
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    except ImportError:
        return
    FastAPIInstrumentor.instrument_app(app)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the CANON FastAPI ASGI server.")
    parser.add_argument("--host", default=os.getenv("HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8000")))
    parser.add_argument("--max-concurrency", type=int, default=int(os.getenv("CANON_MAX_CONCURRENCY", "8")))
    parser.add_argument("--max-queue-depth", type=int, default=int(os.getenv("CANON_MAX_QUEUE_DEPTH", "16")))
    parser.add_argument("--retry-after", type=int, default=2)
    args = parser.parse_args()
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError('Install the serve extra first: pip install -e ".[serve]"') from exc
    uvicorn.run(
        create_app(
            max_concurrency=args.max_concurrency,
            max_queue_depth=args.max_queue_depth,
            retry_after_s=args.retry_after,
        ),
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
