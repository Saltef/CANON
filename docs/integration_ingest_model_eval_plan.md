# CANON Integration, Flexible Ingest, and Model Evaluation Plan

This plan turns CANON from a local evidence workbench into a portable evidence
service that can plug into CRMs, research platforms, and user-owned document
stores.

## Product Direction

CANON should be packaged as an evidence layer, not a monolithic app.

The core jobs are:

- accept documents or records from many systems
- infer the data shape and normalize it into CANON's evidence schema
- retrieve and explain evidence for a user's query
- teach the user which terminology changed retrieval
- compare semantic models on the user's actual tasks
- return auditable citations, limitations, and human-review status

The product boundary stays human-in-the-loop. CANON can prepare evidence and
diagnostics, but the connected CRM, research workspace, or reviewer remains the
place where final decisions are made.

## 1. MCP and Platform Integrations

Expose CANON as an MCP server plus a normal HTTP API.

MCP should make CANON easy to use inside tools that already speak MCP. The HTTP
API should remain the stable integration layer for CRMs, research platforms,
internal dashboards, and batch jobs.

Recommended MCP tools:

- `canon_ingest_source`: ingest files, JSON records, CRM notes, transcripts, or
  platform exports.
- `canon_profile_source`: inspect a sample and return detected fields, document
  types, risks, and a proposed ingestion plan.
- `canon_build_corpus`: build or update a named corpus from ingested sources.
- `canon_answer`: return a cited answer with evidence, limitations, and query
  diagnostics.
- `canon_compare_retrieval`: compare retrieval policies or query variants.
- `canon_query_diagnostics`: return matched terms, weak terms, field phrases,
  variants, semantic drift risk, and result-set stability.
- `canon_evaluate_models`: compare semantic models on a named evaluation set.
- `canon_export_audit`: export answer, citations, model choice, review labels,
  and corpus version.

Recommended HTTP endpoints:

- `POST /v1/sources/profile`
- `POST /v1/sources/ingest`
- `POST /v1/corpora/build`
- `POST /v1/answer`
- `POST /v1/compare`
- `POST /v1/query-diagnostics`
- `POST /v1/model-evaluation`
- `GET /v1/reports/{report_name}`

These integration endpoints are now exposed by the product API. They wrap the
same local pipeline used by the CLI, so an external system can profile a source,
ingest it, build a corpus, and run semantic model evaluation without shelling
out to commands.

CRM integration pattern:

1. CRM sends account notes, emails, call transcripts, tasks, deal metadata, or
   uploaded documents to `sources/profile`.
2. CANON returns a proposed mapping and risk warnings.
3. The user accepts or edits the mapping.
4. CANON ingests the records into a named corpus such as
   `crm_account_acme_q3_v1`.
5. The CRM calls `answer`, `compare`, or `query-diagnostics` from account,
   opportunity, support, or research views.
6. CANON returns citations back to the CRM using source ids that point to the
   original records.

Research-platform integration pattern:

1. Platform sends papers, reports, notes, highlights, tags, and collections.
2. CANON preserves collection names, document types, author/source metadata, and
   user annotations as evidence metadata.
3. A researcher asks a question against a named project corpus.
4. CANON returns evidence, query terminology coaching, answer stability, and
   review tasks.
5. Accepted or rejected evidence becomes learned local vocabulary and retrieval
   feedback for that project.

## 2. Shape-Flexible Ingestion

The ingest pipeline should move from "bring JSONL in the expected shape" to
"bring a source, then CANON profiles, maps, chunks, and validates it."

Target ingestion flow:

1. **Profile:** sample the source and infer its shape.
2. **Classify:** identify whether rows are documents, messages, CRM records,
   papers, notes, tickets, transcripts, or mixed records.
3. **Map:** propose field mappings into CANON's normalized schema.
4. **Extract:** collect text, sections, timestamps, authors, titles, links,
   source ids, and provenance.
5. **Chunk:** choose a chunking strategy based on document type and structure.
6. **Validate:** fail closed on empty text, duplicate ids, bad encodings,
   missing provenance, and unsafe prompt-injection-like content.
7. **Report:** write a data-card-style ingest report before retrieval.

Minimum normalized schema:

```json
{
  "source_record_id": "crm-note-123",
  "canonical_document_id": "account-acme-note-123",
  "title": "Discovery call with ACME",
  "source_name": "Salesforce",
  "document_type": "call_note",
  "provenance": "crm",
  "domain": "enterprise_sales",
  "created_at": "2026-07-14",
  "author": "Jane Smith",
  "url": "https://crm.example.test/acme/note/123",
  "access_scope": "internal",
  "sections": [
    {
      "section": "pain points",
      "text": "The customer described renewal risk caused by slow reporting."
    }
  ],
  "metadata": {
    "account_id": "acme",
    "opportunity_stage": "evaluation"
  }
}
```

Source profiler heuristics:

- wide table with many short text fields: CRM or structured records
- one large text field: document export or note archive
- repeated `sender`, `timestamp`, `message`: chat, email, or transcript
- `title`, `abstract`, `authors`, `doi`: scholarly/research record
- `url`, `html`, `body`: web or article export
- file extension plus extracted text: uploaded document
- nested arrays of comments, sections, messages, or pages: structured document
  with subrecords

Chunking strategies:

- `sectioned_document`: preserve headings and sections.
- `message_thread`: chunk by conversational turn groups, preserving speaker and
  time.
- `crm_record`: chunk by meaningful fields and keep record metadata attached.
- `research_paper`: prioritize abstract, methods, results, limitations, and
  conclusion signals.
- `table_record`: create field-aware text with explicit labels.
- `mixed_source`: profile each record independently and report uncertainty.

User-facing ingest result:

- what CANON thinks the source is
- which fields became evidence text
- which fields became metadata
- which records were skipped and why
- which records may contain prompt-injection text
- what limitations the corpus has before the user queries it

## 3. Semantic Model Evaluation

The goal is not to declare a global winner. The goal is to show which semantic
model works better for a user's corpus, topic, and query style, and why.

CANON already has an embedding provider interface and provider comparison path.
The first task-level model-evaluation harness is implemented as:

```powershell
python -m canon.eval.model_evaluation --mode my_topic_v1_corpus --qrels gold/my_topic_qrels.json --providers local,openai,cohere --k 10
```

It runs the same question set through each available embedding provider and
reports retrieval metrics, slice summaries, provider availability, elapsed time,
and a conservative model recommendation. Remote providers degrade cleanly to
`unavailable` when API keys are missing.

Model registry fields:

- `provider`
- `model`
- `dimensions`
- `embedding_task_type`
- `max_input_tokens`
- `normalization`
- `cost_per_1k_chunks`
- `latency_ms_per_1k_chunks`
- `supports_query_document_modes`
- `notes`

Evaluation inputs:

- named corpus id
- 20 to 50 user questions
- optional qrels or reviewer labels
- query variants from query diagnostics
- accepted/rejected evidence labels
- target use case, such as CRM account research, literature review, policy
  briefing, legal memo triage, customer discovery synthesis, or market research

Core retrieval metrics:

- `Recall@k`: whether known relevant evidence appears in the top k.
- `Precision@k`: how much of the top k is useful.
- `nDCG@k`: whether more useful evidence appears higher in the ranking.
- `MRR`: how quickly the first relevant item appears.
- `Rank overlap`: how much the model changes the evidence set compared with a
  baseline.
- `Coverage`: how many questions retrieve at least one, three, or five relevant
  items.

Grounded-answer metrics:

- citation support rate
- unsupported claim rate
- answer usefulness rating
- abstention quality when evidence is weak
- conflict visibility when retrieved evidence disagrees
- review time per brief

Query-lingo metrics:

- useful phrase rate
- very-useful phrase rate
- query variant retrieval gain
- drift rate
- result-set stability across wording changes
- learned-pattern lift from accepted/rejected reviewer feedback

Model-behavior diagnostics:

- domain lift: improvement on domain-specific terminology
- context lift: improvement on long or section-heavy documents
- relational semantics: improvement on queries that involve relationships,
  such as buyer-to-pain, policy-to-outcome, method-to-finding, or cause-to-risk
- lexical robustness: performance when users ask with non-expert wording
- terminology sensitivity: how much rankings change when phrasing changes
- hard-negative resistance: ability to avoid plausible but wrong nearby results
- source-type bias: whether one document type dominates results unfairly
- stability: variance under query rewrites

Operational metrics:

- embedding build time
- query latency
- storage size
- provider availability
- cost estimate
- privacy posture
- reproducibility

Recommended report sections:

- best model for this corpus and task
- where each model wins
- where each model fails
- whether performance differences are meaningful or too close to call
- model choice recommendation with confidence level
- examples of changed results and why they matter

## 4. Finish Line

The next industry-quality milestone should be:

**CANON can connect to an external system, ingest arbitrary user-owned records
through a profiled mapping, build a named corpus, answer with citations and query
diagnostics, and compare at least three semantic models on a reviewer-labeled
question set.**

Pass criteria:

- MCP server exposes ingest, profile, answer, diagnostics, and model evaluation
  tools.
- HTTP API exposes equivalent endpoints for non-MCP integrations.
- Source profiler correctly maps at least five source shapes:
  document JSONL, table/CSV-like records, CRM notes, message transcripts, and
  research-paper records.
- Ingest reports skipped records, inferred document types, source provenance,
  chunking strategy, and corpus limitations.
- User can run a full test from their own files without editing code.
- Model evaluation compares at least `local`, `openai`, and `cohere` when keys
  are available, while degrading cleanly when remote providers are unavailable.
- Evaluation reports retrieval quality, answer support, query-lingo usefulness,
  drift, latency, and cost.
- Human review labels remain required before production claims.

## Suggested Build Order

1. Add a source profiler and mapping report.
2. Add a flexible ingest CLI that accepts JSONL, CSV, TXT, Markdown, and folder
   inputs.
3. Add MCP tools that wrap the existing product service functions.
4. Add HTTP endpoints for source profiling, ingest, corpus build, and model
   evaluation.
5. Upgrade provider comparison into task-level model evaluation.
6. Add a user-facing local test command that creates a corpus, runs diagnostics,
   runs model comparison, and writes a single pilot report.

The result is a product that is easy to try, easy to integrate, and honest about
where the evidence and model choice are strong or weak.
