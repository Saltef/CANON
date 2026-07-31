from __future__ import annotations

import contextlib
import contextvars
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


_REQUEST_ID = contextvars.ContextVar("canon_request_id", default="")
_JSON_LOGGER_NAME = "canon.operations"


class JsonLineFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        fields = getattr(record, "fields", {})
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "epoch_ms": round(time.time() * 1000, 3),
            "level": record.levelname.lower(),
            "event": getattr(record, "event", record.getMessage()),
            "request_id": getattr(record, "request_id", None) or current_request_id() or "unknown",
        }
        if isinstance(fields, dict):
            payload.update(json_safe(fields))
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def configure_json_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(_JSON_LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    resolved = str(log_path.resolve())
    for handler in logger.handlers:
        if getattr(handler, "canon_json_log_path", None) == resolved:
            return logger
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(JsonLineFormatter())
    handler.canon_json_log_path = resolved  # type: ignore[attr-defined]
    logger.addHandler(handler)
    return logger


def close_json_logging(log_path: Path | None = None) -> None:
    logger = logging.getLogger(_JSON_LOGGER_NAME)
    resolved = str(log_path.resolve()) if log_path is not None else None
    for handler in list(logger.handlers):
        if resolved is not None and getattr(handler, "canon_json_log_path", None) != resolved:
            continue
        logger.removeHandler(handler)
        handler.close()


def set_request_id(request_id: str) -> contextvars.Token[str]:
    return _REQUEST_ID.set(request_id)


def reset_request_id(token: contextvars.Token[str]) -> None:
    _REQUEST_ID.reset(token)


def current_request_id() -> str:
    return _REQUEST_ID.get()


def log_event(event: str, **fields: Any) -> None:
    logger = logging.getLogger(_JSON_LOGGER_NAME)
    logger.info(
        event,
        extra={
            "event": event,
            "request_id": fields.pop("request_id", None) or current_request_id() or "unknown",
            "fields": fields,
        },
    )


@dataclass
class ActiveStage:
    stage: str
    attributes: dict[str, Any] = field(default_factory=dict)
    otel_span: Any | None = None

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value
        if self.otel_span is not None:
            with contextlib.suppress(Exception):
                self.otel_span.set_attribute(key, json_safe_scalar(value))


@contextlib.contextmanager
def stage_span(stage: str, **attributes: Any) -> Iterator[ActiveStage]:
    start = time.perf_counter()
    active = ActiveStage(stage=stage, attributes={"stage": stage, **attributes})
    with start_otel_span(stage, active.attributes) as otel_span:
        active.otel_span = otel_span
        log_event("stage_start", **active.attributes)
        try:
            yield active
        except Exception as exc:
            active.set_attribute("latency_ms", elapsed_ms(start))
            active.set_attribute("error_type", exc.__class__.__name__)
            active.set_attribute("error", str(exc))
            log_event("stage_error", **active.attributes)
            raise
        else:
            active.set_attribute("latency_ms", elapsed_ms(start))
            log_event("stage_end", **active.attributes)


@contextlib.contextmanager
def start_otel_span(stage: str, attributes: dict[str, Any]) -> Iterator[Any | None]:
    try:
        from opentelemetry import trace
    except ImportError:
        yield None
        return
    tracer = trace.get_tracer("canon")
    with tracer.start_as_current_span(f"canon.{stage}") as span:
        for key, value in attributes.items():
            with contextlib.suppress(Exception):
                span.set_attribute(key, json_safe_scalar(value))
        yield span


def elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 3)


def estimate_tokens(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, list | tuple | set):
        return sum(estimate_tokens(item) for item in value)
    if isinstance(value, dict):
        return sum(estimate_tokens(item) for item in value.values())
    text = str(value)
    if not text:
        return 0
    return max(1, round(len(text.split()) * 1.3))


def json_safe(value: dict[str, Any]) -> dict[str, Any]:
    return {key: json_safe_value(item) for key, item in value.items()}


def json_safe_value(value: Any) -> Any:
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    if isinstance(value, list | tuple | set):
        return [json_safe_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): json_safe_value(item) for key, item in value.items()}
    return str(value)


def json_safe_scalar(value: Any) -> str | int | float | bool:
    if isinstance(value, str | int | float | bool):
        return value
    return json.dumps(json_safe_value(value), ensure_ascii=False, sort_keys=True)
