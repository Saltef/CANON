# Phase 18: Technical Probe Suite

Phase 18 adds adversarial, annotation-free probes for the named social science
RAG corpus. The probe suite is meant to expose behavior that broad averages can
hide:

- disagreement handling
- relevance traps
- cross-topic diversity pressure
- recency pressure
- out-of-domain weak-support behavior
- method-focused retrieval

Run it with:

```powershell
python -m canon.eval.probes --mode social_science_ir_v1_harvest10 --method-ids diverse_k5_template,rag_k5_template
```

The Docker target is:

```powershell
make eval-probes
```

Inputs live in `gold/probe_queries.json`. Results are written to
`reports/probe_eval_<mode>_<method_set_id>.json`.

The checks are technical expectations over existing metrics such as context
relevance, semantic alignment, claim coverage, source diversity, cluster
diversity, citation support, and conflict awareness. No human annotation is
required.

Follow-on technical runners:

```powershell
python -m canon.eval.batches --mode social_science_ir_v1_harvest10 --batch-sizes 5,10,25,50
python -m canon.eval.providers --mode social_science_ir_v1_harvest10 --providers local,openai,cohere
python -m canon.embeddings.pgvector --mode social_science_ir_v1_harvest10 --provider local
python -m canon.fulltext.grobid --mode social_science_ir_v1_harvest10
python -m canon.eval.tuning --mode social_science_ir_v1_harvest10
python -m canon.reports.dashboard --mode social_science_ir_v1_harvest10
python -m canon.experiments.manifest --mode social_science_ir_v1_harvest10 --experiment-id social_science_ir_v1_harvest10_full
```

## Mixed Unstructured Document Routing

`canon/eval/unstructured.py` adds a deterministic fixture benchmark for
heterogeneous unstructured text. It is separate from the social-science corpus
because its job is to test routing and safety behavior across document types,
not to measure field-level retrieval quality.

The fixture includes:

- academic article
- policy report
- regulatory filing
- market report
- legal authority
- transcript
- archival primary source
- news article
- prompt-injected web page

Run it with:

```powershell
python -m canon.eval.unstructured --write-report
```

The Docker target is:

```powershell
make mixed-unstructured
```

The report is written to
`reports/mixed_unstructured_document_routing_v1.json`.

The checks cover:

- expected document-type hits for market, legal, policy/method, and
  primary-source queries
- document-type coverage in final retrieval results
- safety rejection of prompt-injected web text before final context
- per-query final and rejected document-type counts

This benchmark is now Phase H in the deterministic phase gate. Passing it means
the current pipeline can exercise mixed document types and current safety
fixtures. It does not mean document-type detection is validated, the corpus is
representative, or answers are substantively correct.

## Document-Type Slice Diagnostics

`canon/eval/document_type_slices.py` adds a reusable technical diagnostic for
heterogeneous corpora. The mixed-unstructured benchmark asks whether expected
document types appear for a fixed fixture. The slice diagnostic asks how each
document type behaves across retrieval policies before generation.

Run it with:

```powershell
python -m canon.eval.document_type_slices --mode unstructured_demo_corpus --policies lexical,balanced,rag
```

The Docker target is:

```powershell
make document-type-slices
```

The target ingests the JSONL demo fixture, assembles `unstructured_demo_corpus`,
and writes `reports/document_type_slices_unstructured_demo_corpus.json`.

The report includes:

- corpus document-type counts
- per-query and per-policy candidate document-type counts
- per-query and per-policy final document-type counts
- per-query and per-policy safety-rejected document-type counts
- candidate safety-decision counts
- slice-level exposure share, average candidate rank, average final rank, and
  rejection share
- corpus document types missing from final results or from safety rejections

This is intentionally a deterministic diagnostic. It is useful for finding
policy sensitivity, overexposure, underexposure, and safety-gate behavior by
document type. It is not a relevance benchmark, and it does not prove that any
document type is intrinsically high or low quality. Production use needs
domain-specific query suites, qrels or human labels, and hand-audited
document-type metadata.

## Domain Slice Diagnostics

`canon/eval/domain_slices.py` adds a second slice diagnostic for mixed-domain
unstructured corpora. Document-type slices ask whether filings, legal
authorities, web pages, transcripts, and other source forms behave differently.
Domain slices ask whether declared domains behave differently: economics,
psychology, anthropology, history, cultural studies, legal/market, and general
academic text.

Run it with:

```powershell
python -m canon.ingest.unstructured --input data/fixtures/mixed_domain_sample.jsonl --mode mixed_domain_demo
python -m canon.corpus.build --corpus-id mixed_domain_demo_corpus --from-modes mixed_domain_demo --corpus-only
python -m canon.eval.domain_slices --mode mixed_domain_demo_corpus --policies lexical,balanced,rag
```

The Docker target is:

```powershell
make domain-slices
```

The report is written to `reports/domain_slices_<mode>.json`.

The report includes:

- corpus domain counts
- candidate, final, and rejected domain counts by query and policy
- domain-fit tier counts
- candidate safety-decision counts
- slice-level exposure share, average rank, average domain-fit score, and
  safety rejection counts
- domains missing from final results or from safety rejections

The current domain-fit model is deterministic and descriptive. It should be
used to find profile mismatches, metadata gaps, and policy sensitivity. It
does not yet prove that a domain profile is correct, and it does not replace
domain-specific human labels, qrels, expert review, or adversarial benchmarks.

## Heterogeneous Unstructured Readiness Gate

`canon/eval/unstructured_readiness.py` combines the heterogeneous fixtures and
slice reports into one readiness artifact. It answers a narrow but important
question: is the current unstructured-data experiment bundle coherent enough to
inspect without silently missing source forms, domains, contracts, or safety
checks?

Run it with:

```powershell
python -m canon.eval.unstructured_readiness --document-type-mode unstructured_demo_corpus --domain-mode mixed_domain_demo_corpus
```

The Docker target is:

```powershell
make unstructured-readiness
```

The target generates:

- mixed-unstructured routing benchmark
- document-type slices
- domain slices
- data cards for the document-type and mixed-domain demo corpora
- `reports/heterogeneous_unstructured_readiness_unstructured_demo_corpus_mixed_domain_demo_corpus.json`
- `reports/heterogeneous_unstructured_readiness_unstructured_demo_corpus_mixed_domain_demo_corpus.md`

The gate checks:

- artifact contracts pass
- mixed-unstructured final document-type coverage is high enough
- prompt-injected web text does not reach final context
- document-type slices expose multiple final source forms
- web-page attack fixtures are safety-rejected
- domain slices cover at least five corpus and final domains
- legal/market adversarial web evidence is safety-rejected
- data cards expose sufficient document-type and domain coverage
- unknown document-type and low/review-needed domain-fit shares remain below
  thresholds

A passing readiness gate means the experiment bundle is internally coherent and
ready for inspection. It does not mean the deterministic quality profiles are
validated, the fixtures are representative, or the system is production-ready.

## Unstructured Coverage Matrix

`canon/eval/unstructured_matrix.py` adds an experiment-design audit for the
heterogeneous unstructured bundle. It maps the current processed corpora against
target domains, preferred document types, task families, and required label
families.

Run it with:

```powershell
python -m canon.eval.unstructured_matrix
```

The Docker target is:

```powershell
make unstructured-coverage-matrix
```

It writes:

- `reports/unstructured_experiment_coverage_matrix_v1.json`
- `reports/unstructured_experiment_coverage_matrix_v1.md`

The matrix currently reports `needs_human_labels`. That is the correct
scientific answer for the expanded fixture bundle. The experiment now includes
fixture coverage for all target source forms used by the task matrix, including
preprints, policy reports, market reports, legal authority, filings, web pages,
transcripts, archival primary sources, social posts, forum comments, and an
unknown-text negative control. The matrix still flags:

- fixture-only evidence for all target domains
- preferred document-type gaps inside specific domain profiles, such as books
  or legal authorities for some interpretive domains
- missing or limited label families for document type, evidence role, source
  trust, and answer usefulness
- task families that need corpus expansion or human/external labels before
  broad validity claims

This matrix is deliberately not a leaderboard. It is a guardrail against
mistaking coherent plumbing for validated cross-domain performance.

## Unstructured Experiment Portfolio

`canon/reports/unstructured_portfolio.py` rolls the heterogeneous experiment
bundle into one inspectable artifact. This is separate from the 10k OpenAlex
portfolio because it answers a different question: can CANON run a disciplined
unstructured-data experiment across source forms, domains, public-opinion text,
chunking, prompt-injection safety, and claim boundaries?

Run it with:

```powershell
python -m canon.reports.unstructured_portfolio
```

The Docker target is:

```powershell
make unstructured-portfolio
```

The target depends on:

- heterogeneous unstructured readiness
- public-opinion corpus diagnostics
- public-opinion synthesis smoke test
- unstructured coverage matrix
- chunking benchmark
- adversarial RAG security
- adversarial corroboration
- evaluation anchors
- deterministic importance phase gate

It writes:

- `reports/unstructured_experiment_portfolio_v1.json`
- `reports/unstructured_experiment_portfolio_v1.md`

The portfolio checks:

- readiness and phase-gate contracts pass
- prompt-injection attacks have zero context exposure
- boundary-aware chunking preserves evidence spans at least as well as fixed
  windows and isolates the unsafe social span
- chunk data cards expose typed chunk decisions such as resolution, parent
  context mode, generation role, expansion policy, evidence-containment score,
  and safety-contamination risk
- synthesis can attach bounded parent context to atomic evidence only when the
  chunk policy allows expansion and the assembled parent context passes safety
  screening
- the coverage matrix exposes corpus-expansion and label gaps instead of
  allowing broad validity claims
- public-opinion analysis declares aggregation readiness rather than treating
  posts as factual authority
- the evidence committee marks public-opinion evidence as aggregate-only
- the public-opinion synthesis smoke test has zero ordinary generator-context
  items while excluding aggregate-only evidence
- evaluation anchors include explicit overclaiming limits

The report's most important design feature is not a green status. It is the
claim boundary: CANON can currently demonstrate experiment plumbing and
deterministic safety behavior on fixtures, but it still cannot claim production
robustness, representative public opinion, legal truth, market sentiment, or
validated disciplinary source quality.

## Generic Unstructured JSONL Ingestion

`canon/ingest/unstructured.py` ingests arbitrary JSONL records into the same
processed `works_<mode>.json` and `chunks_<mode>.json` files used by the rest of
CANON.

Each JSONL row must include one of:

- `text`
- `content`
- `body`
- `sections`

Recommended metadata:

- `id`
- `title`
- `year`
- `source_name`
- `document_type`
- `provenance`
- `domain`
- `url`
- `doi`
- `authors`
- `metadata`

`sections` may be a list of objects like:

```json
{"section": "risk factors", "text": "The filing discloses..."}
```

Run the demo fixture with:

```powershell
python -m canon.ingest.unstructured --input data/fixtures/unstructured_sample.jsonl --mode unstructured_demo
```

The Docker target is:

```powershell
make ingest-unstructured-demo
```

The ingest report is written to `reports/unstructured_ingest_<mode>.json`.
The report includes document-type profile coverage and metadata limitations,
including missing `document_type` and missing `provenance` rates.

Unstructured modes can also be assembled into named corpora:

```powershell
python -m canon.corpus.build --corpus-id unstructured_demo_corpus --from-modes unstructured_demo --corpus-only
python -m canon.reports.data_card --mode unstructured_demo_corpus
```

Named corpus assembly now accepts `raw/openalex_<mode>.json`,
`raw/unstructured_<mode>.json`, or `raw/raw_<mode>.json` source files. Mixed
raw outputs are written with the neutral `raw_<corpus_id>.json` name so the
artifact does not imply that every source came from OpenAlex.

## Public Social-Media And Forum Ingestion

`canon/ingest/social_media.py` adds a public-opinion ingestion lane for
allowed public social-media exports, forum comments, CSV/JSON captures, and
simple HTML fixtures. This is not a live scraper yet. It is the safety- and
metadata-aware ingestion boundary that a live collector should feed.

Supported input formats:

- JSONL
- JSON objects with `records`, `posts`, or `comments`
- CSV
- HTML elements annotated with `data-canon-social`

Required behavior:

- private, direct, deleted, friends-only, and followers-only records are skipped
- records without text are skipped
- public records are normalized into CANON works/chunks
- author identifiers are hashed before storage
- platform, visibility, collection method, thread, parent, engagement, URL, and
  timestamp metadata are preserved when available
- social posts are typed as `social_media_post`
- public forum comments are typed as `forum_comment`
- the domain defaults to `public_opinion`

Run the fixture ingest with:

```powershell
python -m canon.ingest.social_media --input data/fixtures/social_media_public_opinion_sample.jsonl --mode social_public_opinion_demo --format jsonl
```

The Docker target is:

```powershell
make ingest-social-public-opinion-demo
```

Build the full public-opinion demo corpus and diagnostics with:

```powershell
python -m canon.corpus.build --corpus-id social_public_opinion_demo_corpus --from-modes social_public_opinion_demo --corpus-only
python -m canon.reports.data_card --mode social_public_opinion_demo_corpus
python -m canon.eval.document_type_slices --mode social_public_opinion_demo_corpus --policies lexical,balanced,rag
python -m canon.eval.domain_slices --mode social_public_opinion_demo_corpus --policies lexical,balanced,rag
python -m canon.eval.public_opinion --mode social_public_opinion_demo_corpus
```

The Docker target is:

```powershell
make social-public-opinion-demo
```

The fixture deliberately includes a prompt-injected public post. The expected
behavior is that retrieval may surface it as a candidate, but Stage 3 safety
gating rejects it before final ranking and generation. This makes the public
opinion path useful for testing high-risk unstructured text without allowing
retrieved text to instruct the model.

`canon/eval/public_opinion.py` writes
`reports/public_opinion_analysis_<mode>.json` and `.md`. It summarizes:

- platform, thread, visibility, and collection-method coverage
- safety decisions and safety categories
- deterministic theme counts for all public text and safety-allowed text
- deterministic stance labels for all public text and safety-allowed text
- duplicate-text risk
- author-hash activity concentration
- temporal coverage
- aggregation-readiness checks

This report is a technical readiness diagnostic. It is not a sentiment model,
poll, bot detector, or market signal by itself.

Current deterministic limitations:

- the fixture is too small for any public-opinion inference
- no representativeness, bot, coordination, or campaign-detection model exists
  yet
- platform engagement metrics are annotation-only and should not be treated as
  reliability signals
- public visibility is accepted from metadata and needs collector-side legal,
  consent, robots, and terms-of-service checks
- HTML parsing is fixture-oriented, not a general-purpose web scraper

Future work should add compliant platform collectors, de-duplication,
cross-platform sampling metadata, stance/theme extraction, temporal burst
detection, bot/coordination signals, and adversarial tests for prompt injection
in scraped text.
