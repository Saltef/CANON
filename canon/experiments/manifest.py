from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from canon.config import load_settings


ARTIFACT_PATTERNS = [
    "method_eval_{mode}_{method_set_id}.json",
    "evaluation_suite_{mode}_{method_set_id}.json",
    "batch_eval_{mode}_{method_set_id}.json",
    "stress_slices_{mode}_{method_set_id}.json",
    "probe_eval_{mode}_{method_set_id}.json",
    "qrels_validation_internal_social_science_ir_qrels_v1.json",
    "qrels_validation_public_beir_scifact_smoke.json",
    "external_ir_internal_social_science_ir_qrels_v1_{mode}_{method_set_id}.json",
    "bootstrap_ir_internal_social_science_ir_qrels_v1_{mode}_{method_set_id}.json",
    "paired_significance_internal_social_science_ir_qrels_v1_{mode}_{method_set_id}.json",
    "faithfulness_{mode}_{method_set_id}.json",
    "perturbation_eval_{mode}_{method_set_id}.json",
    "data_card_{mode}.json",
    "data_card_{mode}.md",
    "corpus_expansion_{mode}.json",
    "weight_tuning_{mode}_{method_set_id}.json",
    "claim_decision_{mode}_{method_set_id}.json",
    "claim_decision_{mode}_{method_set_id}.md",
    "product_readiness_{mode}.json",
    "regression_gate_{mode}_{method_set_id}.json",
    "provider_compare_{mode}.json",
    "pgvector_{mode}_local.json",
    "grobid_plan_{mode}.json",
    "dashboard_{mode}_{method_set_id}.json",
    "scientific_audit_{mode}_{method_set_id}.json",
    "scientific_audit_{mode}_{method_set_id}.md",
]


def build_experiment_manifest(
    experiment_id: str,
    mode: str,
    method_set_id: str = "baseline_methods_v1",
) -> dict:
    settings = load_settings()
    artifacts = []
    for pattern in ARTIFACT_PATTERNS:
        path = settings.reports_dir / pattern.format(mode=mode, method_set_id=method_set_id)
        if path.exists():
            artifacts.append(artifact_record(path, settings.root))
    manifest = {
        "experiment_id": experiment_id,
        "mode": mode,
        "method_set_id": method_set_id,
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
        "commands": recommended_commands(mode),
    }
    output_path = settings.reports_dir / f"experiment_manifest_{experiment_id}.json"
    write_json(output_path, manifest)
    return manifest


def artifact_record(path: Path, root: Path) -> dict:
    data = path.read_bytes()
    return {
        "path": str(path.relative_to(root)).replace("\\", "/"),
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
    }


def recommended_commands(mode: str) -> list[str]:
    return [
        f"python -m canon.eval.pipeline --mode {mode}",
        f"python -m canon.eval.batches --mode {mode} --batch-sizes 5,10,25,50",
        f"python -m canon.eval.probes --mode {mode}",
        "python -m canon.eval.qrels --input gold/ir_qrels_social_science_ir_v1_harvest10.json --format canon",
        "python -m canon.eval.qrels --input gold/public_qrels_beir_scifact_smoke.json --format canon",
        f"python -m canon.eval.external_ir --mode {mode} --k 10",
        f"python -m canon.eval.uncertainty --mode {mode} --metric nDCG@10 --samples 500",
        f"python -m canon.eval.significance --mode {mode} --metric nDCG@10 --samples 1000",
        f"python -m canon.eval.faithfulness --mode {mode} --query-limit 5",
        f"python -m canon.eval.perturbations --mode {mode} --query-limit 8",
        f"python -m canon.reports.data_card --mode {mode}",
        f"python -m canon.corpus.expansion --mode {mode} --target-work-count 1000",
        f"python -m canon.eval.tuning --mode {mode}",
        f"python -m canon.reports.dashboard --mode {mode}",
        f"python -m canon.reports.claim_decision --mode {mode}",
        f"python -m canon.product.readiness --mode {mode}",
        f"python -m canon.eval.regression_gate --mode {mode}",
        f"python -m canon.reports.scientific_audit --mode {mode}",
    ]


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a reproducible CANON experiment manifest.")
    parser.add_argument("--experiment-id", default="social_science_ir_v1_harvest10_full")
    parser.add_argument("--mode", default="social_science_ir_v1_harvest10")
    parser.add_argument("--method-set-id", default="baseline_methods_v1")
    args = parser.parse_args()
    print(json.dumps(build_experiment_manifest(args.experiment_id, args.mode, args.method_set_id), indent=2))


if __name__ == "__main__":
    main()
