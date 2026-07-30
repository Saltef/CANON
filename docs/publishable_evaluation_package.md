# Publishable Evaluation Package

CANON's public research claim should be packaged as an auditable evidence
workflow, not as a generic RAG leaderboard.

The package card freezes the current benchmark suite, public qrels files,
disagreement-preservation fixture, report artifacts, review CSV handoffs, git
commit, artifact hashes, reproduction commands, and human-review blockers:

```powershell
python -m canon.product.publishable_workflow --suite conf/benchmark_suites/stage1_public_full.json
python -m canon.product.publishable_package --suite conf/benchmark_suites/stage1_public_full.json
python -m canon.product.publishable_verify --package reports/publishable_package_canon_publishable_evidence_workflow_v1.json
python -m canon.product.publishable_export --package reports/publishable_package_canon_publishable_evidence_workflow_v1.json
```

The output is written to:

- `reports/publishable_workflow_v1.json`
- `reports/publishable_workflow_v1.md`
- `reports/publishable_package_canon_publishable_evidence_workflow_v1.json`
- `reports/publishable_package_canon_publishable_evidence_workflow_v1.md`
- `reports/publishable_benchmark_card_v1.json`
- `reports/publishable_benchmark_card_v1.md`
- `reports/publishable_human_review_status_v1.json`
- `reports/publishable_human_review_status_v1.md`
- `reports/publishable_artifact_verification_v1.json`
- `reports/publishable_artifact_verification_v1.md`
- `reports/publishable_bundle_export_v1.json`
- `reports/publishable_bundle_export_v1.md`
- `reports/publishable_bundle_canon_publishable_evidence_workflow_v1.zip`

## What The Card Proves

The card can prove that a specific CANON checkout has a frozen automated
benchmark artifact, Stage 1 optimization report, disagreement-preservation
benchmark report, human-review status artifact, and hashable reproduction
bundle.

It does not prove final factual correctness, clinical safety, durable model
superiority, or publication-quality unsupported-claim rates. Those claims remain
blocked until reviewed qrels and reviewed synthesis labels are complete.

## Required Evidence

- Full public benchmark qrels for configured benchmark targets, not only mini or
  30-query pilot slices.
- The configured public qrels files must appear in the frozen artifact manifest
  with SHA-256 hashes and byte counts.
- Retrieval diagnostics that include candidate recall, ranking metrics, fusion
  comparisons, and score-observability signals.
- A resumable Stage 1 optimizer report.
- Completed Stage 1 optimizer trials for every fusion method declared by the
  publishable suite.
- An automated benchmark-suite report.
- A passing disagreement-preservation benchmark report.
- The disagreement-preservation input fixture used to produce the synthesis
  report.
- A publishable human-review scaffold for retrieval and synthesis labels that
  meets the configured distinct retrieval-query and synthesis-case targets.
- The retrieval and synthesis review CSV templates handed to reviewers.
- Human-reviewed qrels and synthesis labels before release-quality claims.

Import the full public qrels from locally mounted BEIR-format data with:

```powershell
python -m canon.ingest.beir --dataset-dir data/raw/external/scifact --mode beir_scifact_full --benchmark-id beir_scifact_full_qrels --split test --include-qrels-documents --chunk-tokens 220 --overlap-tokens 0
python -m canon.ingest.beir --dataset-dir data/raw/external/nfcorpus --mode beir_nfcorpus_full --benchmark-id beir_nfcorpus_full_qrels --split test --include-qrels-documents --chunk-tokens 220 --overlap-tokens 0
```

The publishable suite is:

```powershell
conf/benchmark_suites/stage1_public_full.json
```

Full public evaluation runs can be interrupted and resumed. Automated suite
benchmark caches live under `reports/automated_benchmark_cache/`; per-query
rerank caches live under `reports/automated_benchmark_query_cache/`. Rerun the
same command after a timeout to continue from cached queries.

By default, `canon.product.publishable_workflow` freezes the lightweight package
artifacts from existing Stage 1 benchmark reports. To rerun the heavy resumable
Stage 1 stages as part of the workflow, use:

```powershell
python -m canon.product.publishable_workflow --suite conf/benchmark_suites/stage1_public_full.json --run-benchmarks --model-providers local --rerankers heuristic --candidate-values 50 --fusion-methods union,rrf,weighted_bm25_dense
```

Summarize full-suite metrics and failure diagnostics with:

```powershell
python -m canon.product.publishable_benchmark_card --automated-suite-report reports/automated_benchmark_suite_stage1_public_full_v1.json --optimizer-report reports/stage1_optimizer_v1.json
```

The benchmark card is the reviewer-facing interpretation layer. It reports
candidate-recall failures, ranking failures, score-observability signals,
disagreement-preservation status, human-review status, and blocked claims. It
does not replace the hashed package manifest.

Verify the frozen artifact manifest after package generation with:

```powershell
python -m canon.product.publishable_verify --package reports/publishable_package_canon_publishable_evidence_workflow_v1.json
```

The verifier checks artifact paths, byte counts, and SHA-256 hashes. It does
not prove human-review completeness, factual correctness, or model superiority.

Export a shareable bundle after verification with:

```powershell
python -m canon.product.publishable_export --package reports/publishable_package_canon_publishable_evidence_workflow_v1.json
```

The export copies the package, verification report, and every manifest artifact
into `reports/publishable_bundle_canon_publishable_evidence_workflow_v1/` and
writes a zip archive beside it. The export is still a review handoff; it does
not convert automated diagnostics into publication-quality claims.

Run the controlled disagreement benchmark with:

```powershell
python -m canon.eval.disagreement_fixture --output gold/disagreement_preservation_publishable.json
python -m canon.eval.disagreement_preservation --benchmark gold/disagreement_preservation_publishable.json
```

The benchmark sends a fixture evidence packet through Stage 2 and scores link
recall, link precision, stance accuracy, contradiction recall, citation
integrity, unsupported-claim rate, and an aggregate disagreement-preservation
score. This is still an automated pre-human signal; it is not a substitute for
reviewed synthesis labels.

Prepare the retrieval and synthesis review scaffolds with:

```powershell
python -m canon.product.human_review_scaffold --suite conf/benchmark_suites/stage1_public_full.json --disagreement-benchmark gold/disagreement_preservation_publishable.json
```

This writes JSON plus CSV templates for qrels labels and synthesis labels. The
scaffold records the current query/case counts against publication targets and
keeps `human_review_complete` false until reviewers provide valid labels. The
package gate now checks those target flags separately: review CSV files are not
enough if the scaffold only covers a pilot-sized synthesis fixture. Completed
review rows also require a stable `reviewer_id` so label provenance is auditable.

After reviewers complete the CSV labels, validate and import the publishable
review status with:

```powershell
python -m canon.product.publishable_review --retrieval-csv reports/publishable_retrieval_review_scaffold.csv --synthesis-csv reports/publishable_synthesis_review_scaffold.csv
```

This writes `reports/publishable_human_review_status_v1.json`. The publishable
package only accepts that suite-specific status artifact; older generic
`human_review_status_v1.json` files do not satisfy the gate.

The intended status before human labels are complete is
`publication_blocked_evidence_incomplete`. That is a useful result: it keeps
CANON's public claims honest while making the remaining work explicit.
