# Phase 2: Baseline Retrieval And Trace Harness

## Goal

Build the first experiment harness for comparing retrieval policies.

Phase 2 does not try to produce beautiful final answers. It makes the retrieval
decision process visible so we can test whether different importance signals
change which sources and chunks are selected.

## Retrieval Policies

- `lexical`: BM25-style chunk retrieval only.
- `importance`: quality/text-importance ranking with weak query matching.
- `balanced`: lexical relevance plus source/text importance.
- `recency`: lexical relevance plus recency.

Author prominence is available as an explicit, separately weighted signal. It is
not folded into source quality.

## Trace Contract

Every result stores:

- chunk id
- work id
- title
- source
- year
- section
- text preview
- final score
- score components
- policy weights

This trace contract is the backbone for later dense retrieval, reranking,
generation, and ablation reports.

## Comparison Output

`canon.retrieval.compare` runs one query across several policies and writes a
rank table showing how each chunk moved. This is the first visible test of the
importance-aware thesis.
