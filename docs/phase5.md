# Phase 5: Citation Graph Scaffold

## Goal

Build the first citation graph diagnostics from OpenAlex `referenced_works`.

The graph module does not claim that citation clusters are paradigms. It reports
what the graph can support:

- node count
- edge count
- matched-reference fraction
- in-corpus degree
- connected components as deterministic first-pass clusters
- cluster summaries by title/source/year

This gives later diversity-aware retrieval something concrete to use while
keeping the interpretation honest.

## Retrieval Integration

The retrieval layer can now attach connected-component cluster IDs to traces and
use a transparent diversity bonus to avoid selecting only one graph component.
