# Phase 16: Named Corpus Validation Loop

## Goal

Move from a tiny live sample to a named, reproducible corpus with validation
artifacts.

## Practice Basis

The validation loop follows the direction of current RAG and scientific-QA work:

- RAGAS-style reference-free checks for retrieval context, faithfulness, and
  answer quality.
- ARES-style separation of context relevance, answer faithfulness, and answer
  relevance, with room for a small human-labeled set.
- OpenScholar/SQuAI-style emphasis on citation-backed scientific synthesis and
  supporting evidence.
- Hybrid sparse/semantic retrieval with transparent traces.

Sources reviewed while designing this phase:

- RAGAS: Automated Evaluation of Retrieval Augmented Generation
- ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation Systems
- OpenScholar: Synthesizing Scientific Literature with Retrieval-augmented LMs
- SQuAI: Scientific Question-Answering with Multi-Agent Retrieval-Augmented Generation
- TREC 2024 Retrieval Augmented Generation Track materials

## Outputs

- `data/raw/openalex_<corpus_id>.json`
- `data/processed/works_<corpus_id>.json`
- `data/processed/chunks_<corpus_id>.json`
- `reports/corpus_<corpus_id>.json`
- `reports/phase16_<corpus_id>.json`
- downstream quality, graph, claims, conflicts, embeddings, RAG eval, and
  workbench reports for the named corpus
