.PHONY: build ci bootstrap-fixtures product-api product-smoke product-readiness product-release-audit product-release-audit-report product-final-check product-final-check-report industry-pilot industry-pilot-review industry-pilot-review-status industry-pilot-review-csv industry-pilot-import-review-csv dry-run test live diagnostics eval graph claims conflicts claim-model embeddings synthesize rag-eval topic-pack corpus-expansion harvest-v2 harvest-10k ingest-unstructured-demo ingest-mixed-domain-demo ingest-social-public-opinion-demo chunking-eval evidence-committee document-type-slices domain-slices social-public-opinion-analysis social-public-opinion-demo public-opinion-synthesis-smoke unstructured-readiness unstructured-coverage-matrix unstructured-portfolio workbench phase16 methods eval-pipeline eval-diversity diversity-diagnostics diversity-gate eval-batches eval-slices eval-probes eval-batches-large qrels-validate public-qrels-validate external-ir bootstrap-ir paired-significance faithfulness label-tasks label-calibration technical-calibration calibration-model preference-model hard-negative-anchors preference-model-anchors mixed-unstructured adversarial-corroboration adversarial-rag-security evaluation-anchors contract-validate importance-phase-gate perturbations data-card claim-decision regression-gate provider-compare pgvector-plan grobid-plan tune-weights dashboard manifest scientific-audit portfolio finished-demo full-eval

build:
	docker compose build canon

bootstrap-fixtures:
	python -m canon.ingest.unstructured --input data/fixtures/unstructured_sample.jsonl --mode unstructured_demo
	python -m canon.corpus.build --corpus-id unstructured_demo_corpus --from-modes unstructured_demo --corpus-only
	python -m canon.ingest.unstructured --input data/fixtures/mixed_domain_sample.jsonl --mode mixed_domain_demo
	python -m canon.corpus.build --corpus-id mixed_domain_demo_corpus --from-modes mixed_domain_demo --corpus-only
	python -m canon.ingest.social_media --input data/fixtures/social_media_public_opinion_sample.jsonl --mode social_public_opinion_demo --format jsonl
	python -m canon.corpus.build --corpus-id social_public_opinion_demo_corpus --from-modes social_public_opinion_demo --corpus-only
	python -m canon.eval.committee_gate_usefulness
	python -m canon.eval.chunking_variants --write-report
	python -c "from pathlib import Path; import json; path = Path('reports'); path.mkdir(parents=True, exist_ok=True); (path / 'human_review_tasks_v1.json').write_text(json.dumps({'records': []}) + '\\n', encoding='utf-8')"

ci: bootstrap-fixtures
	python -m pip install -e .
	python -m unittest discover -s tests
	python -m canon.ingest.pipeline --dry-run
	python -m canon.quality.diagnostics --mode dry_run
	python -m canon.eval.pipeline --mode dry_run
	python -m canon.eval.batches --mode dry_run --batch-sizes 1,3,5
	python -m canon.eval.probes --mode dry_run --method-ids diverse_k5_template,rag_k5_template
	python -m canon.eval.qrels --input gold/ir_qrels_social_science_ir_v1_harvest10.json --format canon --benchmark-id internal_social_science_ir_qrels_v1
	python -m canon.eval.qrels --input gold/public_qrels_beir_scifact_smoke.json --format canon --benchmark-id public_beir_scifact_smoke
	python -m canon.eval.external_ir --mode dry_run --k 10
	python -m canon.eval.uncertainty --mode dry_run --metric nDCG@10 --samples 500
	python -m canon.eval.significance --mode dry_run --metric nDCG@10 --samples 1000
	python -m canon.eval.faithfulness --mode dry_run --query-limit 5
	python -m canon.eval.perturbations --mode dry_run --query-limit 8
	python -m canon.reports.data_card --mode dry_run
	python -m canon.reports.claim_decision --mode dry_run
	python -m canon.product.industry_pilot --mode dry_run --prepare-review
	python -m canon.product.industry_pilot --mode dry_run --records reports/human_review_tasks_v1.json --review-status
	python -m canon.product.industry_pilot --mode dry_run --records reports/human_review_tasks_v1.json --no-fail
	python -c "from canon.eval.regression_gate import run_regression_gate; import json; print(json.dumps(run_regression_gate('dry_run', thresholds={'data_card_work_count_min': 1, 'claim_decision_required_resolutions_min': 0}), indent=2))"
	python -m canon.reports.scientific_audit --mode dry_run
	python -m canon.product.smoke --mode dry_run
	python -m canon.product.readiness --mode dry_run
	python -m canon.product.release_audit --mode dry_run --no-fail

product-api:
	docker compose up canon

product-smoke:
	docker compose run --rm --no-deps canon python -m canon.product.smoke --mode social_science_ir_v1_harvest10

product-readiness:
	docker compose run --rm --no-deps canon python -m canon.product.readiness --mode social_science_ir_v1_harvest10

product-release-audit:
	docker compose run --rm --no-deps canon python -m canon.product.release_audit --mode social_science_ir_v1_harvest10

product-release-audit-report:
	docker compose run --rm --no-deps canon python -m canon.product.release_audit --mode social_science_ir_v1_harvest10 --no-fail

product-final-check:
	docker compose run --rm --no-deps canon python -m canon.product.final_check --mode social_science_ir_v1_harvest10 --records /app/reports/human_review_tasks_v1.json

product-final-check-report:
	docker compose run --rm --no-deps canon python -m canon.product.final_check --mode social_science_ir_v1_harvest10 --records /app/reports/human_review_tasks_v1.json --no-fail

industry-pilot:
	docker compose run --rm --no-deps canon python -m canon.product.industry_pilot --mode social_science_ir_v1_harvest10

industry-pilot-review:
	docker compose run --rm --no-deps canon python -m canon.product.industry_pilot --mode social_science_ir_v1_harvest10 --prepare-review

industry-pilot-review-status:
	docker compose run --rm --no-deps canon python -m canon.product.industry_pilot --mode social_science_ir_v1_harvest10 --records /app/reports/human_review_tasks_v1.json --review-status

industry-pilot-review-csv:
	docker compose run --rm --no-deps canon python -m canon.product.industry_pilot --mode social_science_ir_v1_harvest10 --records /app/reports/human_review_tasks_v1.json --export-review-csv --output /app/reports/human_review_tasks_v1.review.csv

industry-pilot-import-review-csv:
	docker compose run --rm --no-deps canon python -m canon.product.industry_pilot --mode social_science_ir_v1_harvest10 --records /app/reports/human_review_tasks_v1.json --import-review-csv /app/reports/human_review_tasks_v1.review.csv

dry-run:
	docker compose run --rm canon python -m canon.ingest.pipeline --dry-run

test:
	docker compose run --rm canon python -m unittest discover -s tests

live:
	docker compose run --rm canon python -m canon.ingest.pipeline --live --max-results 50

diagnostics:
	docker compose run --rm --no-deps canon python -m canon.quality.diagnostics --mode live

eval:
	docker compose run --rm --no-deps canon python -m canon.eval.harness --mode live --top-k 5

graph:
	docker compose run --rm --no-deps canon python -m canon.graph.build --mode live

claims:
	docker compose run --rm --no-deps canon python -m canon.claims.extract --mode live

conflicts:
	docker compose run --rm --no-deps canon python -m canon.claims.conflict --mode live

claim-model:
	docker compose run --rm --no-deps canon python -m canon.claims.model --write-report "We find robust support for the democratic peace."

synthesize:
	docker compose run --rm --no-deps canon python -m canon.synthesis.answer "What does the literature say about democratic peace?" --mode live --policy rag

rag-eval:
	docker compose run --rm --no-deps canon python -m canon.eval.rag --mode live --top-k 5 --policies lexical,balanced,semantic,rag

embeddings:
	docker compose run --rm --no-deps canon python -m canon.embeddings.store --mode live --provider local

topic-pack:
	docker compose run --rm --no-deps canon python -m canon.corpus.packs --max-results 50

corpus-expansion:
	docker compose run --rm --no-deps canon python -m canon.corpus.expansion --mode social_science_ir_v1_harvest10 --target-work-count 10000

harvest-v2:
	docker compose run --rm --no-deps canon python -m canon.corpus.build --corpus-id social_science_ir_v2 --harvest --max-results 200 --top-k 5 --policies lexical,balanced,semantic,rag,diverse,conflict_aware

harvest-10k:
	docker compose run --rm --no-deps canon python -m canon.corpus.build --corpus-id social_science_ir_10k --harvest --max-results 2200 --corpus-only --top-k 5 --policies lexical,balanced,semantic,rag,diverse,conflict_aware

ingest-unstructured-demo:
	docker compose run --rm --no-deps canon python -m canon.ingest.unstructured --input /app/data/fixtures/unstructured_sample.jsonl --mode unstructured_demo

ingest-mixed-domain-demo:
	docker compose run --rm --no-deps canon python -m canon.ingest.unstructured --input /app/data/fixtures/mixed_domain_sample.jsonl --mode mixed_domain_demo

ingest-social-public-opinion-demo:
	docker compose run --rm --no-deps canon python -m canon.ingest.social_media --input /app/data/fixtures/social_media_public_opinion_sample.jsonl --mode social_public_opinion_demo --format jsonl

chunking-eval:
	docker compose run --rm --no-deps canon python -m canon.eval.chunking --chunk-tokens 14 --overlap-tokens 0 --write-report

evidence-committee:
	docker compose run --rm --no-deps canon python -m canon.eval.committee --mode social_public_opinion_demo_corpus --policies balanced,rag --top-k 3

document-type-slices: ingest-unstructured-demo
	docker compose run --rm --no-deps canon python -m canon.corpus.build --corpus-id unstructured_demo_corpus --from-modes unstructured_demo --corpus-only
	docker compose run --rm --no-deps canon python -m canon.eval.document_type_slices --mode unstructured_demo_corpus --policies lexical,balanced,rag

domain-slices: ingest-mixed-domain-demo
	docker compose run --rm --no-deps canon python -m canon.corpus.build --corpus-id mixed_domain_demo_corpus --from-modes mixed_domain_demo --corpus-only
	docker compose run --rm --no-deps canon python -m canon.eval.domain_slices --mode mixed_domain_demo_corpus --policies lexical,balanced,rag

social-public-opinion-analysis:
	docker compose run --rm --no-deps canon python -m canon.eval.public_opinion --mode social_public_opinion_demo_corpus

social-public-opinion-demo: ingest-social-public-opinion-demo
	docker compose run --rm --no-deps canon python -m canon.corpus.build --corpus-id social_public_opinion_demo_corpus --from-modes social_public_opinion_demo --corpus-only
	docker compose run --rm --no-deps canon python -m canon.reports.data_card --mode social_public_opinion_demo_corpus
	docker compose run --rm --no-deps canon python -m canon.eval.document_type_slices --mode social_public_opinion_demo_corpus --policies lexical,balanced,rag
	docker compose run --rm --no-deps canon python -m canon.eval.domain_slices --mode social_public_opinion_demo_corpus --policies lexical,balanced,rag
	docker compose run --rm --no-deps canon python -m canon.eval.public_opinion --mode social_public_opinion_demo_corpus
	docker compose run --rm --no-deps canon python -m canon.eval.committee --mode social_public_opinion_demo_corpus --policies balanced,rag --top-k 3

public-opinion-synthesis-smoke: social-public-opinion-demo
	docker compose run --rm --no-deps canon python -m canon.synthesis.answer "public opinion battery storage safety concerns" --mode social_public_opinion_demo_corpus --policy rag --top-k 3

unstructured-readiness: mixed-unstructured document-type-slices domain-slices
	docker compose run --rm --no-deps canon python -m canon.reports.data_card --mode unstructured_demo_corpus
	docker compose run --rm --no-deps canon python -m canon.reports.data_card --mode mixed_domain_demo_corpus
	docker compose run --rm --no-deps canon python -m canon.eval.unstructured_readiness --document-type-mode unstructured_demo_corpus --domain-mode mixed_domain_demo_corpus

unstructured-coverage-matrix: unstructured-readiness social-public-opinion-demo evaluation-anchors
	docker compose run --rm --no-deps canon python -m canon.eval.unstructured_matrix

unstructured-portfolio: unstructured-readiness unstructured-coverage-matrix social-public-opinion-demo public-opinion-synthesis-smoke chunking-eval adversarial-rag-security adversarial-corroboration evaluation-anchors importance-phase-gate
	docker compose run --rm --no-deps canon python -m canon.reports.unstructured_portfolio

workbench:
	docker compose run --rm --no-deps canon python -m canon.workbench.build --mode live

portfolio:
	docker compose run --rm --no-deps canon python -m canon.reports.portfolio --mode social_science_ir_10k

finished-demo:
	docker compose run --rm --no-deps canon python -m canon.eval.diversity --mode social_science_ir_10k
	docker compose run --rm --no-deps canon python -m canon.eval.diversity_diagnostics --mode social_science_ir_10k
	docker compose run --rm --no-deps canon python -m canon.eval.diversity --mode social_science_ir_10k --diverse-method-id focus_diverse_k5_template --baseline-method-id lexical_k5_template
	docker compose run --rm --no-deps canon python -m canon.eval.diversity_diagnostics --mode social_science_ir_10k --diverse-method-id focus_diverse_k5_template --baseline-method-id lexical_k5_template
	docker compose run --rm --no-deps canon python -m canon.eval.diversity_gate --mode social_science_ir_10k --diverse-method-id focus_diverse_k5_template --baseline-method-id lexical_k5_template
	docker compose run --rm --no-deps canon python -m canon.workbench.build --mode social_science_ir_10k
	docker compose run --rm --no-deps canon python -m canon.reports.portfolio --mode social_science_ir_10k

phase16:
	docker compose run --rm --no-deps canon python -m canon.corpus.build --corpus-id social_science_ir_v1 --from-modes live --top-k 5

methods:
	docker compose run --rm --no-deps canon python -m canon.eval.methods --mode social_science_ir_v1_harvest10

eval-pipeline:
	docker compose run --rm --no-deps canon python -m canon.eval.pipeline --mode social_science_ir_v1_harvest10

eval-diversity:
	docker compose run --rm --no-deps canon python -m canon.eval.diversity --mode social_science_ir_10k

diversity-diagnostics:
	docker compose run --rm --no-deps canon python -m canon.eval.diversity_diagnostics --mode social_science_ir_10k

diversity-gate:
	docker compose run --rm --no-deps canon python -m canon.eval.diversity_gate --mode social_science_ir_10k

eval-batches:
	docker compose run --rm --no-deps canon python -m canon.eval.batches --mode social_science_ir_v1_harvest10 --batch-sizes 1,3,5

eval-slices:
	docker compose run --rm --no-deps canon python -m canon.eval.slices --mode social_science_ir_v1_harvest10 --method-ids diverse_k5_template,rag_k5_template

eval-probes:
	docker compose run --rm --no-deps canon python -m canon.eval.probes --mode social_science_ir_v1_harvest10 --method-ids diverse_k5_template,rag_k5_template

eval-batches-large:
	docker compose run --rm --no-deps canon python -m canon.eval.batches --mode social_science_ir_v1_harvest10 --batch-sizes 5,10,25,50

qrels-validate:
	docker compose run --rm --no-deps canon python -m canon.eval.qrels --input /app/gold/ir_qrels_social_science_ir_v1_harvest10.json --format canon

public-qrels-validate:
	docker compose run --rm --no-deps canon python -m canon.eval.qrels --input /app/gold/public_qrels_beir_scifact_smoke.json --format canon

external-ir:
	docker compose run --rm --no-deps canon python -m canon.eval.external_ir --mode social_science_ir_v1_harvest10 --k 10

bootstrap-ir:
	docker compose run --rm --no-deps canon python -m canon.eval.uncertainty --mode social_science_ir_v1_harvest10 --metric nDCG@10 --samples 500

paired-significance:
	docker compose run --rm --no-deps canon python -m canon.eval.significance --mode social_science_ir_v1_harvest10 --metric nDCG@10 --samples 1000

faithfulness:
	docker compose run --rm --no-deps canon python -m canon.eval.faithfulness --mode social_science_ir_v1_harvest10 --query-limit 5

label-tasks:
	docker compose run --rm --no-deps canon python -m canon.labeling.tasks --mode social_science_ir_v1_harvest10 --policies lexical,balanced,semantic,rag

label-calibration: label-tasks
	docker compose run --rm --no-deps canon python -m canon.labeling.calibration --mode social_science_ir_v1_harvest10

technical-calibration:
	docker compose run --rm --no-deps canon python -m canon.eval.technical_calibration --mode social_science_ir_v1_harvest10 --policies lexical,balanced,semantic,rag

calibration-model:
	docker compose run --rm --no-deps canon python -m canon.modeling.calibration_model --mode social_science_ir_v1_harvest10 --source technical --policies lexical,balanced,semantic,rag

preference-model:
	docker compose run --rm --no-deps canon python -m canon.modeling.preference_model --mode social_science_ir_v1_harvest10 --source technical --policies lexical,balanced,semantic,rag

hard-negative-anchors:
	docker compose run --rm --no-deps canon python -m canon.eval.hard_negatives --write-report

preference-model-anchors: hard-negative-anchors
	docker compose run --rm --no-deps canon python -m canon.modeling.preference_model --mode hard_negative_anchor_preferences_v1 --source anchors

mixed-unstructured:
	docker compose run --rm --no-deps canon python -m canon.eval.unstructured --write-report

adversarial-corroboration:
	docker compose run --rm --no-deps canon python -m canon.eval.corroboration --write-report

adversarial-rag-security:
	docker compose run --rm --no-deps canon python -m canon.eval.security --write-report

evaluation-anchors:
	docker compose run --rm --no-deps canon python -m canon.eval.anchors --write-report

contract-validate: adversarial-corroboration adversarial-rag-security evaluation-anchors unstructured-readiness unstructured-portfolio
	docker compose run --rm --no-deps canon python -m canon.eval.contracts /app/reports/adversarial_corroboration_v1.json --type adversarial_corroboration
	docker compose run --rm --no-deps canon python -m canon.eval.contracts /app/reports/adversarial_rag_security_v1.json --type adversarial_rag_security
	docker compose run --rm --no-deps canon python -m canon.eval.contracts /app/reports/evaluation_anchor_registry_v1.json --type evaluation_anchors
	docker compose run --rm --no-deps canon python -m canon.eval.contracts /app/reports/document_type_slices_unstructured_demo_corpus.json --type document_type_slices
	docker compose run --rm --no-deps canon python -m canon.eval.contracts /app/reports/domain_slices_mixed_domain_demo_corpus.json --type domain_slices
	docker compose run --rm --no-deps canon python -m canon.eval.contracts /app/reports/heterogeneous_unstructured_readiness_unstructured_demo_corpus_mixed_domain_demo_corpus.json --type heterogeneous_unstructured_readiness
	docker compose run --rm --no-deps canon python -m canon.eval.contracts /app/reports/unstructured_experiment_coverage_matrix_v1.json --type unstructured_coverage_matrix
	docker compose run --rm --no-deps canon python -m canon.eval.contracts /app/reports/unstructured_experiment_portfolio_v1.json --type unstructured_experiment_portfolio

importance-phase-gate:
	docker compose run --rm --no-deps canon python -m canon.eval.phase_gate --write-report

perturbations:
	docker compose run --rm --no-deps canon python -m canon.eval.perturbations --mode social_science_ir_v1_harvest10 --query-limit 8

data-card:
	docker compose run --rm --no-deps canon python -m canon.reports.data_card --mode social_science_ir_v1_harvest10

claim-decision:
	docker compose run --rm --no-deps canon python -m canon.reports.claim_decision --mode social_science_ir_v1_harvest10

regression-gate:
	docker compose run --rm --no-deps canon python -m canon.eval.regression_gate --mode social_science_ir_v1_harvest10

provider-compare:
	docker compose run --rm --no-deps canon python -m canon.eval.providers --mode social_science_ir_v1_harvest10 --providers local,openai,cohere

pgvector-plan:
	docker compose run --rm --no-deps canon python -m canon.embeddings.pgvector --mode social_science_ir_v1_harvest10 --provider local

grobid-plan:
	docker compose run --rm --no-deps canon python -m canon.fulltext.grobid --mode social_science_ir_v1_harvest10

tune-weights:
	docker compose run --rm --no-deps canon python -m canon.eval.tuning --mode social_science_ir_v1_harvest10

dashboard:
	docker compose run --rm --no-deps canon python -m canon.reports.dashboard --mode social_science_ir_v1_harvest10

manifest:
	docker compose run --rm --no-deps canon python -m canon.experiments.manifest --mode social_science_ir_v1_harvest10 --experiment-id social_science_ir_v1_harvest10_full

scientific-audit:
	docker compose run --rm --no-deps canon python -m canon.reports.scientific_audit --mode social_science_ir_v1_harvest10

full-eval: eval-pipeline eval-batches-large eval-slices eval-probes qrels-validate public-qrels-validate external-ir bootstrap-ir paired-significance faithfulness label-calibration technical-calibration calibration-model preference-model preference-model-anchors mixed-unstructured document-type-slices domain-slices unstructured-readiness unstructured-coverage-matrix unstructured-portfolio adversarial-corroboration adversarial-rag-security evaluation-anchors contract-validate importance-phase-gate perturbations data-card provider-compare pgvector-plan grobid-plan tune-weights dashboard claim-decision product-readiness regression-gate manifest scientific-audit
