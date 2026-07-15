# Phase 14: Topic Packs

## Goal

Expand the corpus deliberately across multiple social-science topics.

## Implementation

Topic packs define a reviewable set of topic files and recommended harvest
commands. Ingestion now accepts `--topic-file`, so each topic can be harvested
with explicit provenance.

## Outputs

- `reports/topic_pack_<pack_id>.json`
- topic configs under `conf/topics/`
