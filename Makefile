.PHONY: build product-api product-readiness dry-run test live diagnostics eval graph claims conflicts claim-model embeddings synthesize rag-eval topic-pack corpus-expansion harvest-v2 harvest-10k workbench phase16 methods eval-pipeline eval-diversity diversity-diagnostics diversity-gate eval-batches eval-slices eval-probes eval-batches-large qrels-validate public-qrels-validate external-ir bootstrap-ir paired-significance faithfulness perturbations data-card claim-decision regression-gate provider-compare pgvector-plan grobid-plan tune-weights dashboard manifest scientific-audit full-eval

build:
	docker compose build canon

product-api:
	docker compose up canon

product-readiness:
	docker compose run --rm --no-deps canon python -m canon.product.readiness --mode social_science_ir_v1_harvest10

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

workbench:
	docker compose run --rm --no-deps canon python -m canon.workbench.build --mode live

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

full-eval: eval-pipeline eval-batches-large eval-slices eval-probes qrels-validate public-qrels-validate external-ir bootstrap-ir paired-significance faithfulness perturbations data-card provider-compare pgvector-plan grobid-plan tune-weights dashboard claim-decision product-readiness regression-gate manifest scientific-audit
