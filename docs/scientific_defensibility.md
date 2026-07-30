# Scientific Defensibility Plan

CANON is a research-oriented RAG system for testing how different notions of
text and source importance change retrieval, citation selection, grounded answer
quality, and disagreement handling in social-science literature.

The project should be presented as an experimental system, not as a finished
truth engine. Its defensibility comes from explicit hypotheses, auditable
signals, reproducible reports, adversarial probes, and clear limitations.

## Claim Boundary

Defensible claim:

> CANON implements and evaluates multiple importance-aware retrieval policies
> over a controlled social-science corpus, and reports how those policies trade
> off relevance, source quality, diversity, recency, claim coverage, conflict
> awareness, and abstention behavior.

Claims to avoid until more evidence exists:

- CANON is scientifically superior to state-of-the-art RAG systems.
- CANON can judge the truth of social-science claims.
- CANON's current social-science corpus is representative of the field.
- CANON's deterministic claim model is a validated claim classifier.

## Evaluation Anchors

The project should be compared against established evaluation practice:

- BEIR: heterogeneous zero-shot retrieval evaluation across domains and tasks.
  CANON should adopt BEIR-style retrieval metrics such as nDCG@k, Recall@k,
  MAP, and MRR when external qrels are available.
  https://arxiv.org/abs/2104.08663
- RAGAS: reference-free RAG evaluation dimensions such as faithfulness, answer
  relevance, and context relevance. CANON already measures technical analogs of
  context relevance, citation support, and abstention, but still needs stronger
  answer faithfulness checks.
  https://arxiv.org/abs/2309.15217
- RAG evaluation survey: separates retrieval, generation, faithfulness,
  robustness, and benchmark design. CANON should preserve that separation in
  reports.
  https://arxiv.org/abs/2405.07437
- MTEB: embedding systems should be evaluated across retrieval, clustering,
  reranking, and related tasks rather than a single narrow benchmark.
  https://arxiv.org/abs/2210.07316

## Current Evidence Bundle

The committed evidence bundle is intentionally compact. It includes:

- Public qrels fixtures: `gold/beir_scifact_full_qrels.json`,
  `gold/beir_nfcorpus_full_qrels.json`, and
  `gold/beir_nfcorpus_stage1_title_preserve_qrels.json`
- Disagreement-preservation fixture:
  `gold/disagreement_preservation_publishable.json`
- Stage 1 fixed-qrels summary:
  `reports/stage1_fixed_qrels_v2_summary.json`
- Public benchmark suite definition:
  `conf/benchmark_suites/stage1_public_full.json`
- Publishable evaluation package guide:
  `docs/publishable_evaluation_package.md`

The broader generated portfolio bundle is local by default because many reports
are large, cache-like, or project-specific. Before citing a report publicly,
regenerate it from the committed commands, check its claim boundary, and commit
a compact summary artifact rather than raw caches.

Generated local artifacts may include:

- Method definitions: `conf/methods/baseline_methods.json`
- Retrieval policy weights: `conf/settings.toml`
- Seed queries: `gold/seed_queries.json`
- Probe queries: `gold/probe_queries.json`
- Main evaluation: `reports/evaluation_suite_social_science_ir_v1_harvest10_baseline_methods_v1.json`
- Probe evaluation: `reports/probe_eval_social_science_ir_v1_harvest10_baseline_methods_v1.json`
- Batch stability: `reports/batch_eval_social_science_ir_v1_harvest10_baseline_methods_v1.json`
- Qrels validation: `reports/qrels_validation_internal_social_science_ir_qrels_v1.json`
- Public-qrels smoke validation: `reports/qrels_validation_public_beir_scifact_smoke.json`
- BEIR/TREC-style IR metrics: `reports/external_ir_internal_social_science_ir_qrels_v1_social_science_ir_v1_harvest10_baseline_methods_v1.json`
- Bootstrap uncertainty: `reports/bootstrap_ir_internal_social_science_ir_qrels_v1_social_science_ir_v1_harvest10_baseline_methods_v1.json`
- Paired significance: `reports/paired_significance_internal_social_science_ir_qrels_v1_social_science_ir_v1_harvest10_baseline_methods_v1.json`
- Citation faithfulness: `reports/faithfulness_social_science_ir_v1_harvest10_baseline_methods_v1.json`
- Query perturbation robustness: `reports/perturbation_eval_social_science_ir_v1_harvest10_baseline_methods_v1.json`
- Corpus data card: `reports/data_card_social_science_ir_v1_harvest10.md`
- Weight sensitivity: `reports/weight_tuning_social_science_ir_v1_harvest10_baseline_methods_v1.json`
- Claim decision: `reports/claim_decision_social_science_ir_v1_harvest10_baseline_methods_v1.md`
- Regression gates: `reports/regression_gate_social_science_ir_v1_harvest10_baseline_methods_v1.json`
- Dashboard: `reports/dashboard_social_science_ir_v1_harvest10_baseline_methods_v1.html`
- Reproducibility manifest: `reports/experiment_manifest_social_science_ir_v1_harvest10_full.json`
- Scientific audit: `reports/scientific_audit_social_science_ir_v1_harvest10_baseline_methods_v1.md`

The heterogeneous unstructured portfolio should include:

- Mixed unstructured routing benchmark: `reports/mixed_unstructured_document_routing_v1.json`
- Document-type slices: `reports/document_type_slices_unstructured_demo_corpus.json`
- Domain slices: `reports/domain_slices_mixed_domain_demo_corpus.json`
- Heterogeneous readiness gate:
  `reports/heterogeneous_unstructured_readiness_unstructured_demo_corpus_mixed_domain_demo_corpus.json`
- Unstructured coverage matrix:
  `reports/unstructured_experiment_coverage_matrix_v1.json`
- Public-opinion analysis:
  `reports/public_opinion_analysis_social_public_opinion_demo_corpus.json`
- Evidence committee:
  `reports/evidence_committee_social_public_opinion_demo_corpus.json`
- Public-opinion synthesis smoke test:
  `reports/synthesis_social_public_opinion_demo_corpus_rag_public-opinion-battery-storage-safety-concerns.json`
- Chunking strategy benchmark: `reports/chunking_strategy_eval_ct14_ov0.json`
- Adversarial RAG security benchmark: `reports/adversarial_rag_security_v1.json`
- Importance phase gate: `reports/importance_phase_gate_v1.json`
- Unstructured experiment portfolio:
  `reports/unstructured_experiment_portfolio_v1.md`

## Scientific Controls Already Present

- Multiple baselines: lexical, balanced, semantic, hybrid RAG, diversity-first,
  conflict-aware, source-quality-heavy, and recency-heavy.
- Versioned method set and scoring weights.
- Technical probes for disagreement, relevance traps, diversity pressure,
  recency pressure, off-topic queries, and method-focused retrieval.
- Batch-size stability checks.
- Weight-sensitivity analysis.
- BEIR/TREC-style retrieval metrics on internal weak qrels: nDCG@k, Recall@k,
  MAP@k, and MRR@k.
- Qrels import and validation utilities for canonical JSON and TREC-style qrels.
- Bootstrap confidence intervals and winner probabilities for retrieval-metric
  rankings.
- Paired bootstrap intervals and sign-flip permutation checks over per-query
  method differences.
- Corpus data-card reporting for source modes, coverage, years, sources, and
  known limitations.
- Claim-decision report that blocks global-winner and external-benchmark claims
  when instability, narrow margins, weak bootstrap identification, or internal
  qrels are present.
- Citation faithfulness checks that require answer sentences to cite valid
  evidence and share support terms with cited chunks or extracted claims.
- Query perturbation robustness tests for light paraphrases and framing changes.
- Regression gates that fail when required technical evidence falls below
  explicit thresholds.
- Query-aware abstention and weak-support diagnostics.
- Heterogeneous unstructured portfolio checks that connect document-type,
  domain, public-opinion, chunking, safety, committee, synthesis, and anchor
  artifacts into one claim-bounded report.
- Coverage-matrix checks that distinguish fixture coverage from broad
  validation readiness across domains, document types, task families, and label
  anchors.
- Reproducible report hashes through experiment manifests.
- Dockerized execution path.

## Remaining Scientific Gaps

- External benchmark validation: replace or supplement internal weak qrels with
  a full public BEIR/TREC corpus and qrels evaluation, not only smoke-test qrels
  validation.
- Statistical uncertainty: widen bootstrap/significance runs and add larger
  external benchmark qrels before strong claims.
- Faithfulness evaluation: supplement lexical citation checks with an external
  NLI or judge-based verifier when network/model use is allowed.
- Corpus validity: expand data-card coverage with explicit inclusion/exclusion
  criteria and public benchmark corpus comparisons.
- Claim model validity: compare deterministic claim extraction against at least
  one external model or public argument-mining benchmark.
- Generalization: test non-IR social-science corpora and non-social-science
  corpora to separate domain effects from method effects.
- Unstructured validation: replace the current fixture-heavy portfolio with
  larger corpora and domain-expert labels for economics, psychology,
  anthropology, history, cultural studies, legal/market, and public-opinion
  tasks.
- Coverage gaps currently flagged by the matrix are no longer basic source-form
  absence for the task matrix; the expanded fixtures include market reports,
  policy reports, preprints, public-opinion records, web pages, legal authority,
  filings, transcripts, archival primary sources, and an unknown-text negative
  control. The remaining matrix gaps are scientific: fixture-only scale,
  preferred-type gaps inside some domain profiles, and missing or limited labels
  for document type, evidence role, source trust, and answer usefulness.
- Human-free for now does not mean assumption-free. Technical-only evaluation
  should be labeled as such until human expert review is added.

## Portfolio Presentation Guidance

Lead with the scientific question:

> What happens when a RAG system treats importance as multi-dimensional rather
> than a single relevance score?

Then show:

1. System architecture and importance signals.
2. Method set and controlled comparisons.
3. Probe suite and failure-driven iteration.
4. Main leaderboard with near-tie warnings.
5. Limitations and next validation steps.

The strongest portfolio signal is not that CANON has a perfect winner. It is
that the system makes trade-offs visible, auditable, and reproducible.

For heterogeneous unstructured data, the strongest portfolio signal is similar:
CANON now makes unsafe, aggregate-only, weakly supported, and domain-mismatched
evidence visible before generation. The current artifact still proves
experiment discipline, not broad real-world validity.
