# Deploy CANON Backend On Render

CANON's working deployment shape is a protected backend service plus optional
frontend. GitHub Pages can host a static frontend later, but it cannot run the
Python API or hold provider secrets.

## Backend Shape

Use Render as a Docker web service. The repo includes `render.yaml` for a free
preview service:

- Runtime: Docker
- Dockerfile path: `Dockerfile`
- Health check path: `/health`
- Public HTTP binding: handled by the Docker `CMD`, which binds `0.0.0.0` and
  uses Render's `PORT` environment variable.
- Blueprint service: `canon-api`
- Plan in `render.yaml`: `free`

Render environment variables:

```text
CANON_VECTORSTORE=qdrant
CANON_SETTINGS=/app/conf/settings.toml
CANON_DATA_DIR=/app/data
CANON_REPORTS_DIR=/app/reports
CANON_ALLOWED_ORIGINS=
CANON_REQUIRE_AUTH=true
CANON_BASIC_AUTH_USER=<set in Render>
CANON_BASIC_AUTH_PASSWORD=<set in Render>
CANON_API_KEY=<set in Render if API clients need bearer auth>
CANON_MAX_CONCURRENCY=8
CANON_MAX_QUEUE_DEPTH=16
OPENROUTER_API_KEY=<set in Render>
COHERE_API_KEY=<set in Render>
QDRANT_URL=<set in Render>
QDRANT_API_KEY=<set in Render>
```

Direct OpenAI provider keys are not used. OpenAI-named model IDs are routed
through OpenRouter.

## Free Preview Limits

The free Render service is a working preview target, not durable production
storage. Render free web services use an ephemeral filesystem, spin down when
idle, and cannot attach persistent disks.

The Docker image includes only the small `ai_infra_geo_risk_demo` processed
corpus so a fresh deploy can answer a demo query without a bind mount. User
corpora ingested through `/app` or `/v1/production/corpus-setup` on the free
service can disappear on restart, redeploy, or spin-down. Qdrant stores the ANN
index, but CANON still treats `data/processed` as the source corpus for BM25 and
evidence joins.

Use the free service to test login, routing, model-provider wiring, Qdrant
connectivity, and the interactive workflow. Do not treat free Render storage as
a place to keep private corpora.

## Durable Alpha Storage

For durable hosted user corpora, upgrade the web service to a paid instance and
attach a persistent disk. Render only preserves filesystem changes under the
disk mount path.

Recommended disk:

- Mount path: `/app/storage`
- App data env: `CANON_DATA_DIR=/app/storage/data`
- App reports env: `CANON_REPORTS_DIR=/app/storage/reports`

This preserves processed corpora, source manifests, vector-index manifests, and
run reports. Qdrant remains the hosted vector index, not the source of truth.

## Access Control

The Render Blueprint sets `CANON_REQUIRE_AUTH=true`. The service should fail at
startup unless at least one auth mechanism is configured:

- Browser/workbench access: `CANON_BASIC_AUTH_USER` and
  `CANON_BASIC_AUTH_PASSWORD`
- API clients: `CANON_API_KEY` with `Authorization: Bearer ...` or
  `X-CANON-API-Key`

`/health` remains public for Render health checks. Other routes are protected
when auth is configured. For private localhost-only development you can leave
`CANON_REQUIRE_AUTH=false`, but do not use that setting on a public backend.

Set `CANON_ALLOWED_ORIGINS` to the backend URL and any static frontend URL that
should be allowed to call the API. Do not use a broad origin allowlist for a
public alpha.

## Local Docker Smoke

```powershell
docker compose build canon
docker compose up canon
```

Then check:

```powershell
Invoke-WebRequest http://localhost:8000/health
```

If `CANON_BASIC_AUTH_USER` and `CANON_BASIC_AUTH_PASSWORD` are set, browser
access to `http://localhost:8000/app` should prompt for credentials.

For a Render-like image smoke that does not mount local `data/` or `reports/`,
run:

```powershell
docker compose --profile render-smoke up -d --build canon-render-smoke
Invoke-WebRequest http://localhost:8001/health
docker compose --profile render-smoke down
```

## Frontend Split

The current workbench is served by the backend at `/app`. That is the simplest
deployable alpha.

A GitHub Pages frontend should be a later static app that calls the Render API
through a configurable backend URL. It must not contain Qdrant, OpenRouter, or
Cohere keys.
