from __future__ import annotations

import argparse
import asyncio
from collections import Counter
import json
import math
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_LEVELS = [1, 5, 10, 20, 35, 50]


async def run_load_test(
    *,
    base_url: str,
    queries: list[dict[str, Any]],
    levels: list[int],
    duration_s: float,
    timeout_s: float,
    output_dir: Path,
    log_path: Path,
) -> dict[str, Any]:
    import httpx

    run_id = f"load_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / f"{run_id}_samples.jsonl"
    started_epoch_ms = time.time() * 1000
    summaries = []
    raw_records: list[dict[str, Any]] = []
    async with httpx.AsyncClient(base_url=base_url, timeout=timeout_s) as client:
        for level in levels:
            records = await run_level(
                client=client,
                queries=queries,
                concurrency=level,
                duration_s=duration_s,
                run_id=run_id,
            )
            raw_records.extend(records)
            append_jsonl(raw_path, records)
            summaries.append(summarize_level(level, duration_s, records))
    stage_breakdown = summarize_stages(log_path, run_id, started_epoch_ms)
    report = {
        "report_id": "canon_asgi_load_test_v1",
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "base_url": base_url,
        "query_count": len(queries),
        "levels": levels,
        "duration_per_level_s": duration_s,
        "total_duration_s": round(duration_s * len(levels), 3),
        "raw_samples_path": str(raw_path).replace("\\", "/"),
        "summary": summaries,
        "sample_categories": summarize_samples(raw_records),
        "stage_breakdown": stage_breakdown,
        "knee_point": knee_point(summaries),
        "named_failure": named_failure(summaries, raw_records),
        "methodology": {
            "client": "httpx.AsyncClient closed-loop concurrency ramp",
            "boundary": (
                "Use vegeta or another open-loop tool for publication-grade p99 claims when available; "
                "this script is the repo-local fallback for repeatable local ASGI testing."
            ),
        },
    }
    json_path = output_dir / f"{run_id}_report.json"
    md_path = output_dir / f"{run_id}_report.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    report["json_report_path"] = str(json_path).replace("\\", "/")
    report["markdown_report_path"] = str(md_path).replace("\\", "/")
    return report


async def run_level(
    *,
    client,
    queries: list[dict[str, Any]],
    concurrency: int,
    duration_s: float,
    run_id: str,
) -> list[dict[str, Any]]:
    deadline = time.perf_counter() + duration_s
    counter = RequestCounter()
    records: list[dict[str, Any]] = []

    async def worker(worker_id: int) -> None:
        while time.perf_counter() < deadline:
            index = counter.next()
            payload = dict(queries[index % len(queries)])
            payload["request_id"] = f"{run_id}_c{concurrency}_w{worker_id}_r{index}"
            started = time.perf_counter()
            row = {
                "request_id": payload["request_id"],
                "query_id": payload.get("id"),
                "concurrency": concurrency,
                "status_code": None,
                "latency_ms": None,
                "error": None,
                "degradation_flags": [],
            }
            try:
                response = await client.post(
                    "/v1/production/evidence-workbench",
                    json=payload,
                    headers={"X-Request-ID": payload["request_id"]},
                )
                row["status_code"] = response.status_code
                row["latency_ms"] = round((time.perf_counter() - started) * 1000, 3)
                with suppress_json_error():
                    body = response.json()
                    retrieval = body.get("retrieval") or body
                    row["degradation_flags"] = retrieval.get("degradation_flags") or []
                    row["response_status"] = body.get("status")
                    row["retry_after"] = response.headers.get("Retry-After")
            except Exception as exc:  # noqa: BLE001 - load reports should preserve client-side failures.
                row["latency_ms"] = round((time.perf_counter() - started) * 1000, 3)
                row["error"] = f"{exc.__class__.__name__}: {exc}"
            records.append(row)

    await asyncio.gather(*(worker(index) for index in range(concurrency)))
    return records


class suppress_json_error:
    def __enter__(self):
        return self

    def __exit__(self, *_exc_info):
        return True


class RequestCounter:
    def __init__(self) -> None:
        self.value = 0

    def next(self) -> int:
        current = self.value
        self.value += 1
        return current


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def summarize_level(concurrency: int, duration_s: float, records: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [float(row["latency_ms"]) for row in records if isinstance(row.get("latency_ms"), int | float)]
    total = len(records)
    ok = sum(1 for row in records if row.get("status_code") == 200)
    rejected = sum(1 for row in records if row.get("status_code") == 503)
    errors = sum(1 for row in records if row.get("error"))
    degraded = sum(1 for row in records if row.get("degradation_flags"))
    status_counts = Counter(str(row.get("status_code")) for row in records)
    error_types = Counter(error_type(row.get("error")) for row in records if row.get("error"))
    retry_after = Counter(str(row.get("retry_after")) for row in records if row.get("retry_after") is not None)
    degradation_flags: Counter[str] = Counter()
    for row in records:
        degradation_flags.update(row.get("degradation_flags") or [])
    return {
        "concurrency": concurrency,
        "requests": total,
        "ok": ok,
        "http_503": rejected,
        "client_errors": errors,
        "degraded": degraded,
        "rps": round(total / duration_s, 3) if duration_s else 0.0,
        "error_rate": round((total - ok) / total, 6) if total else 0.0,
        "p50_ms": percentile(latencies, 50),
        "p95_ms": percentile(latencies, 95),
        "p99_ms": percentile(latencies, 99),
        "max_ms": round(max(latencies), 3) if latencies else None,
        "status_counts": dict(sorted(status_counts.items())),
        "error_types": dict(sorted(error_types.items())),
        "retry_after": dict(sorted(retry_after.items())),
        "degradation_flags": dict(sorted(degradation_flags.items())),
    }


def summarize_samples(records: list[dict[str, Any]]) -> dict[str, Any]:
    status_counts = Counter(str(row.get("status_code")) for row in records)
    error_types = Counter(error_type(row.get("error")) for row in records if row.get("error"))
    retry_after = Counter(str(row.get("retry_after")) for row in records if row.get("retry_after") is not None)
    degradation_flags: Counter[str] = Counter()
    for row in records:
        degradation_flags.update(row.get("degradation_flags") or [])
    return {
        "status_counts": dict(sorted(status_counts.items())),
        "error_types": dict(sorted(error_types.items())),
        "retry_after": dict(sorted(retry_after.items())),
        "degradation_flags": dict(sorted(degradation_flags.items())),
    }


def error_type(message: object) -> str:
    if not message:
        return ""
    return str(message).split(":", 1)[0]


def summarize_stages(log_path: Path, run_id: str, started_epoch_ms: float) -> list[dict[str, Any]]:
    if not log_path.exists():
        return []
    rows: dict[str, list[float]] = {}
    for line in log_path.read_text(encoding="utf-8").splitlines():
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if payload.get("event") != "stage_end":
            continue
        if float(payload.get("epoch_ms") or 0.0) < started_epoch_ms:
            continue
        if not str(payload.get("request_id") or "").startswith(run_id):
            continue
        stage = str(payload.get("stage") or "unknown")
        latency = payload.get("latency_ms")
        if isinstance(latency, int | float):
            rows.setdefault(stage, []).append(float(latency))
    return [
        {
            "stage": stage,
            "count": len(values),
            "p50_ms": percentile(values, 50),
            "p95_ms": percentile(values, 95),
            "p99_ms": percentile(values, 99),
            "max_ms": round(max(values), 3) if values else None,
        }
        for stage, values in sorted(rows.items())
    ]


def knee_point(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    if not summaries:
        return {"status": "not_measured"}
    first_p99 = summaries[0].get("p99_ms") or 0.0
    for row in summaries:
        if row["http_503"] or row["client_errors"]:
            return {
                "status": "capacity_limit_observed",
                "concurrency": row["concurrency"],
                "reason": "errors_or_queue_rejections",
            }
        p99 = row.get("p99_ms") or 0.0
        p50 = row.get("p50_ms") or 0.0
        if p99 and p50 and (p99 > max(first_p99 * 3, p50 * 4)):
            return {
                "status": "latency_detached",
                "concurrency": row["concurrency"],
                "reason": "p99_detached_from_p50",
            }
    return {"status": "not_reached", "highest_concurrency": summaries[-1]["concurrency"]}


def named_failure(summaries: list[dict[str, Any]], records: list[dict[str, Any]]) -> dict[str, Any]:
    for row in summaries:
        if row["http_503"]:
            return {
                "name": "queue_depth_exceeded",
                "before": "The stdlib server path had no explicit bounded queue or Retry-After response.",
                "after": f"ASGI returned {row['http_503']} HTTP 503 response(s) at concurrency {row['concurrency']}.",
            }
    for record in records:
        if record.get("degradation_flags"):
            return {
                "name": "degraded_retrieval",
                "before": "Model retrieval or rerank failures previously collapsed into a failed packet.",
                "after": "The ASGI/service path returned a completed packet with explicit degradation_flags.",
                "flags": record.get("degradation_flags"),
            }
    for row in summaries:
        if row["client_errors"]:
            return {
                "name": "client_error",
                "before": "No named client-side failure was isolated before the ASGI load run.",
                "after": f"{row['client_errors']} client-side error(s) occurred at concurrency {row['concurrency']}.",
            }
    return {
        "name": "no_failure_observed",
        "before": "Previous stdlib serving lacked queue-depth enforcement and operational stage logs.",
        "after": "No queue rejection or degradation was observed during this run; stage logs and metrics were emitted.",
    }


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(ordered[0], 3)
    rank = (len(ordered) - 1) * p / 100
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return round(ordered[int(rank)], 3)
    weight = rank - low
    return round(ordered[low] * (1 - weight) + ordered[high] * weight, 3)


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# CANON ASGI Load Test Report",
        "",
        f"- Run ID: `{report['run_id']}`",
        f"- Base URL: `{report['base_url']}`",
        f"- Total duration: `{report['total_duration_s']}s`",
        f"- Frozen queries: `{report['query_count']}`",
        "",
        "## Ramp Table",
        "",
        "| concurrency | requests | ok | 503 | errors | degraded | rps | p50 ms | p95 ms | p99 ms | max ms |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["summary"]:
        lines.append(
            "| {concurrency} | {requests} | {ok} | {http_503} | {client_errors} | {degraded} | "
            "{rps} | {p50_ms} | {p95_ms} | {p99_ms} | {max_ms} |".format(**row)
        )
    lines.extend(["", "## Per-Stage Breakdown", ""])
    if report["stage_breakdown"]:
        lines.extend(
            [
                "| stage | count | p50 ms | p95 ms | p99 ms | max ms |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in report["stage_breakdown"]:
            lines.append(
                "| {stage} | {count} | {p50_ms} | {p95_ms} | {p99_ms} | {max_ms} |".format(**row)
            )
    else:
        lines.append("No stage log lines were found for this run.")
    categories = report.get("sample_categories") or {}
    lines.extend(
        [
            "",
            "## Sample Categories",
            "",
            f"- Status codes: `{json.dumps(categories.get('status_counts', {}), sort_keys=True)}`",
            f"- Client error types: `{json.dumps(categories.get('error_types', {}), sort_keys=True)}`",
            f"- Retry-After headers: `{json.dumps(categories.get('retry_after', {}), sort_keys=True)}`",
            f"- Degradation flags: `{json.dumps(categories.get('degradation_flags', {}), sort_keys=True)}`",
        ]
    )
    knee = report["knee_point"]
    failure = report["named_failure"]
    lines.extend(
        [
            "",
            "## Knee Point",
            "",
            f"`{knee.get('status')}` at concurrency `{knee.get('concurrency', knee.get('highest_concurrency', '-'))}`: "
            f"{knee.get('reason', '')}",
            "",
            "## Named Failure",
            "",
            f"- Name: `{failure['name']}`",
            f"- Before: {failure['before']}",
            f"- After: {failure['after']}",
            "",
            "## Methodology Boundary",
            "",
            report["methodology"]["boundary"],
            "",
        ]
    )
    return "\n".join(lines)


def load_queries(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("Frozen query set must be a non-empty JSON list.")
    return [dict(row) for row in payload]


def parse_levels(value: str) -> list[int]:
    return [int(piece.strip()) for piece in value.split(",") if piece.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a local CANON ASGI concurrency ramp.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8002")
    parser.add_argument("--queries", default="load/frozen_queries.json")
    parser.add_argument("--levels", default=",".join(str(level) for level in DEFAULT_LEVELS))
    parser.add_argument("--duration-per-level", type=float, default=200.0)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--output-dir", default="reports/load")
    parser.add_argument("--log-path", default="reports/asgi_operational_v1.jsonl")
    args = parser.parse_args()
    report = asyncio.run(
        run_load_test(
            base_url=args.base_url.rstrip("/"),
            queries=load_queries(Path(args.queries)),
            levels=parse_levels(args.levels),
            duration_s=args.duration_per_level,
            timeout_s=args.timeout,
            output_dir=Path(args.output_dir),
            log_path=Path(args.log_path),
        )
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
