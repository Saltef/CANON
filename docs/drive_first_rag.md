# Drive-First RAG Workflow

CANON is designed to start from a user's own corpus, such as a mounted Google
Drive folder, and then expand outward only when the private corpus does not
fully answer the question.

The product pattern is:

```text
user query
  -> search private Drive/local corpus
  -> cite relevant internal evidence
  -> identify gaps, uncertainty, and entities
  -> expand to approved external sources
  -> merge private and external evidence with provenance labels
  -> produce a cited answer or evidence packet for human review
```

## Why Drive First

Many useful research questions begin inside a team's own documents:

- memos
- PDFs
- spreadsheets
- meeting notes
- presentations
- exported Google Docs
- local analysis notebooks
- source-code repositories

External search should not replace those materials. It should help corroborate,
update, contextualize, or challenge them.

## Access Model

The current safe path is filesystem-based:

1. The user mounts or syncs Google Drive locally.
2. CANON reads the selected folder path.
3. CANON writes raw/processed artifacts under local `data/` and `reports/`.
4. CANON does not read Google credentials or call Google APIs in this mode.

Example:

```powershell
python -m canon.product.mounted_corpus --input "G:\My Drive\CANON Corpus" --mode project_drive_v1 --profile-only
python -m canon.product.mounted_corpus --input "G:\My Drive\CANON Corpus" --mode project_drive_v1 --corpus-id project_drive_v1_corpus --domain project_domain
```

A future direct Google Drive connector should use least-privilege OAuth scopes,
explicit folder selection, no committed credentials, and clear user consent
before sending private document text to any external model provider.

## Query Flow

For each user query, CANON should do three passes.

### 1. Private Corpus Retrieval

Search the Drive/local corpus first.

Output:

- top cited internal chunks
- source titles and paths
- file-type and provenance metadata
- confidence and weak-support assessment
- gaps in the private corpus

### 2. Expansion Planning

Use the private results to decide what external evidence is needed.

Examples:

- If the Drive corpus mentions a company, query external sources for recent
  filings, announcements, or reputable coverage.
- If the Drive corpus contains an old policy memo, look for newer official
  sources.
- If the Drive corpus has a claim without support, search for corroborating or
  conflicting evidence.
- If the private corpus is sufficient, avoid unnecessary external expansion.

### 3. External Evidence Merge

External results must be marked separately from private corpus results.

Each evidence item should carry:

- `evidence_scope`: `private_corpus` or `external_source`
- `source_name`
- `source_type`
- `retrieval_stage`
- `citation`
- `limitations`

The answer should make it obvious which claims are supported by the user's
documents, which are supported externally, and which remain uncertain.

## Evidence Packet Shape

The `/v1/evidence-packets` API is the natural integration point. A Drive-first
request should include the private corpus mode and expansion preferences:

```json
{
  "request_id": "req_001",
  "project_id": "project_drive",
  "question": "What does our corpus say about grid risk, and what newer external evidence should we consider?",
  "mode": "project_drive_v1_corpus",
  "evidence_requirements": {
    "top_k": 10,
    "include_conflicts": true,
    "include_source_diversity": true,
    "include_query_diagnostics": true,
    "external_expansion": {
      "enabled": true,
      "allowed_source_types": ["official", "academic", "news", "filing"],
      "max_external_queries": 5
    }
  }
}
```

## Safety Rules

- Never ingest folders containing `.env` files, private keys, tokens, or
  credentials.
- Keep private Drive evidence and external evidence visibly separate.
- Do not send private document text to hosted models unless the user explicitly
  enables that provider.
- Use local/heuristic providers for the safest first test.
- Treat generated external search queries as reviewable artifacts.
- Do not claim completeness; report coverage gaps and uncertainty.

## Testing Path

1. Create a small Drive folder with 10-30 representative documents.
2. Run `--profile-only` and inspect supported/unsupported files.
3. Build a named corpus.
4. Ask 5-10 realistic questions.
5. Prepare qrels review candidates.
6. Run the pre-human automated gate.
7. Review qrels and re-run model evaluation against human labels.

See [quickstart.md](quickstart.md), [supported_sources.md](supported_sources.md),
and [evaluation.md](evaluation.md).
