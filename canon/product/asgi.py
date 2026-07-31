import argparse
import asyncio
import time
import uuid
from contextlib import asynccontextmanager
from http import HTTPStatus
from pathlib import Path
from typing import Any

from canon.config import load_settings
from canon.observability import configure_json_logging, log_event, reset_request_id, set_request_id
from canon.product import service, workbench_app


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
):
    try:
        from fastapi import Body, FastAPI, Request
        from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, Response
        from starlette.concurrency import run_in_threadpool
    except ImportError as exc:
        raise RuntimeError('Install the serve extra first: pip install -e ".[serve]"') from exc

    settings = load_settings()
    configure_json_logging(log_path or settings.reports_dir / "asgi_operational_v1.jsonl")
    configure_otel()
    metrics = Metrics()
    gate = ConcurrencyGate(max_concurrency=max_concurrency, max_queue_depth=max_queue_depth)
    app = FastAPI(title="CANON Evidence Discovery API", version="0.1.0")
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
                "POST /v1/evidence-packets",
                "POST /v1/production/evidence-workbench",
            ],
        }

    app.state.canon_gate = gate
    app.state.canon_metrics = metrics
    return app


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
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-concurrency", type=int, default=8)
    parser.add_argument("--max-queue-depth", type=int, default=16)
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
