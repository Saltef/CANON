# Internal Plan: Drive-First RAG With External Expansion

This is the product architecture for turning CANON into a Drive-first research
workflow.

## Product Intent

The user grants access to a Drive folder or provides a mounted Drive path. CANON
uses the user's query to search those private documents first. It then decides
whether external sources are needed for corroboration, freshness, contradiction,
or missing coverage.

The key distinction:

- Private corpus evidence answers "what do our documents say?"
- External evidence answers "what else should we consider?"

## Current Capability

Already implemented:

- Mounted Drive/local folder ingestion through `canon.product.mounted_corpus`.
- Flexible file parsing for PDF, DOCX, XLSX, PPTX, HTML, notebooks, source code,
  CSV/JSON/JSONL, Markdown, and text.
- Google-native pointer and image detection without false extraction.
- Local git repository corpus support.
- Evidence-packet API shape through `/v1/evidence-packets`.
- Pre-human evaluation gate and human-qrels retest path.

## Missing Product Layer

Needed next:

1. Store corpus-source scope on evidence records:
   - `private_corpus`
   - `external_source`
   - `fixture`
2. Add a query-planning step after private retrieval:
   - extract entities
   - identify weak support
   - identify stale evidence
   - generate external queries
3. Add an external-source adapter boundary:
   - OpenAlex for academic/policy-like literature
   - user-approved web/news sources later
   - explicit allowlist before any external search
4. Merge private and external evidence with visible provenance.
5. Add evaluation slices:
   - private-only answer quality
   - expansion usefulness
   - external corroboration precision
   - external drift/noise rate
   - private/external citation separation

## API Direction

Extend `evidence_requirements`:

```json
{
  "external_expansion": {
    "enabled": true,
    "allowed_source_types": ["official", "academic", "news", "filing"],
    "max_external_queries": 5,
    "send_private_text_to_external_models": false
  }
}
```

Default should be fail-closed:

- external expansion disabled unless requested
- hosted model providers disabled unless configured
- private text not sent externally unless explicitly enabled

## Evaluation Gate

A Drive-first expansion run should not pass unless:

- private evidence is cited separately from external evidence
- expansion queries are logged
- external results answer a stated gap
- external evidence does not drown out private evidence
- no unsupported claims are introduced by expansion
- human review can inspect every private and external citation

## Recommended Build Order

1. Add evidence-scope metadata to corpus records/chunks.
2. Add a private-retrieval diagnostics report for a query.
3. Generate external expansion query candidates without executing them.
4. Add an OpenAlex-only external expansion proof of concept.
5. Merge private/external evidence packets.
6. Add qrels/review tasks for expansion usefulness.
7. Add API support and README examples.
