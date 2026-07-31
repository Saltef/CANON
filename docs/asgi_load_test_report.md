# CANON ASGI Operational Load Test

This report publishes the first full local ASGI serving run for CANON's
production evidence workbench path.

## Configuration

- Date: 2026-07-31
- Server: `python -m canon.product.asgi --host 127.0.0.1 --port 8002 --max-concurrency 8 --max-queue-depth 16`
- Endpoint: `POST /v1/production/evidence-workbench`
- Frozen query set: `load/frozen_queries.json`
- Runner: `python -m scripts.load_test_asgi --base-url http://127.0.0.1:8002 --levels 1,5,10,20,35,50 --duration-per-level 200`
- Duration: 1,200 seconds total
- Providers: local deterministic embedding/rerank/generation controls
- Hosted calls: none
- Run ID: `load_20260731T141017Z_054cd321`

External load-test binaries (`oha`, `k6`, `vegeta`, `hey`, `wrk`) were not
available in this local environment. The committed runner uses `httpx` as a
repo-local closed-loop fallback. This is suitable for operational smoke and knee
detection, but open-loop tooling should be used before making publication-grade
p99 or capacity claims.

## Ramp Table

| concurrency | requests | 200 | 503 | client errors | degraded | rps | p50 ms | p95 ms | p99 ms | max ms |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8,640 | 8,640 | 0 | 0 | 0 | 43.200 | 21.980 | 31.191 | 39.288 | 82.689 |
| 5 | 9,880 | 9,880 | 0 | 0 | 1 | 49.400 | 92.369 | 178.491 | 230.510 | 354.374 |
| 10 | 10,389 | 10,389 | 0 | 0 | 7 | 51.945 | 170.852 | 376.830 | 509.301 | 751.462 |
| 20 | 10,593 | 10,593 | 0 | 0 | 5 | 52.965 | 358.847 | 567.915 | 704.984 | 1,127.857 |
| 35 | 42,021 | 5,008 | 20,781 | 16,232 | 6 | 210.105 | 73.884 | 782.627 | 1,307.175 | 5,491.302 |
| 50 | 51,133 | 6,707 | 18,809 | 25,617 | 4 | 255.665 | 115.135 | 771.337 | 1,036.284 | 2,253.557 |

## Sample Categories

| category | count |
|---|---:|
| HTTP 200 | 51,217 |
| HTTP 503 | 39,590 |
| client `ConnectError` | 41,849 |
| `Retry-After: 2` | 39,590 |
| `embed_failed_bm25_only` degradation flag | 23 |

The high-concurrency RPS values include quick queue rejections and connection
failures, so they should not be read as successful evidence-workbench
throughput. Clean successful throughput topped out around 53 completed
requests/sec at concurrency 20 in this run.

## Per-Stage Breakdown

| stage | count | p50 ms | p95 ms | p99 ms | max ms |
|---|---:|---:|---:|---:|---:|
| bm25 | 51,240 | 16.121 | 126.685 | 225.309 | 926.593 |
| embed | 51,217 | 5.886 | 68.636 | 156.520 | 688.188 |
| fuse | 51,240 | 5.524 | 56.601 | 138.551 | 556.002 |
| rerank | 51,217 | 11.072 | 113.525 | 205.815 | 678.382 |
| synthesise | 51,217 | 5.337 | 87.585 | 174.192 | 650.146 |

Stage counts are derived from `stage_end` JSONL events in
`reports/asgi_operational_v1.jsonl` for the run's request-id prefix. The BM25
count is higher than embed/rerank/synthesise because 23 requests exercised the
BM25-only degradation path after embedding failure.

## Knee Point

The knee point is concurrency 35. Concurrency 20 completed without 503s or
client errors; concurrency 35 triggered both bounded queue rejection and local
client connection failures.

This gives a conservative operating envelope for the current local deterministic
path: use concurrency below 35 on this machine, and prefer a production process
manager/load balancer setup before treating this as an externally serviceable
capacity number.

## Named Failure

Name: `queue_depth_exceeded`

Before: the legacy stdlib serving path did not expose an explicit bounded queue
contract or `Retry-After` response. Requests could accumulate until the client
or server failed. This is a structural before/after, not a direct latency
baseline against the legacy server.

After: the ASGI path returned `503` with `Retry-After: 2` for 20,781 requests at
concurrency 35 and 39,590 requests across the two overloaded levels. This is the
intended degradation behavior: reject excess work explicitly rather than hiding
unbounded queueing behind slow or failed requests.

## Evidence Trail

- Raw sample path: `reports/load/load_20260731T141017Z_054cd321_samples.jsonl`
- JSON report path: `reports/load/load_20260731T141017Z_054cd321_report.json`
- Markdown report path: `reports/load/load_20260731T141017Z_054cd321_report.md`
- Operational log path: `reports/asgi_operational_v1.jsonl`

The raw load and operational log files are intentionally kept under ignored
runtime report paths. This committed document is the reviewable, compact
publication surface for the run.
