from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

from canon.config import load_settings
from canon.eval.methods import composite_score


WEIGHT_VARIANTS = {
    "baseline": {},
    "support_heavy": {"support_confidence": 0.18, "context_relevance": 0.18},
    "conflict_heavy": {"conflict_awareness": 0.2, "cluster_diversity": 0.16},
    "source_diverse": {"source_diversity": 0.2, "cluster_diversity": 0.22},
    "recall_claims": {"claim_coverage": 0.3, "citation_support": 0.14},
}


def tune_weights(mode: str, method_set_id: str = "baseline_methods_v1") -> dict:
    settings = load_settings()
    method_eval_path = settings.reports_dir / f"method_eval_{mode}_{method_set_id}.json"
    method_eval = json.loads(method_eval_path.read_text(encoding="utf-8"))
    baseline_weights = method_eval["score_weights"]
    runs = []
    for variant_id, overrides in WEIGHT_VARIANTS.items():
        weights = deepcopy(baseline_weights)
        weights.update(overrides)
        rows = []
        for method_id, summary in method_eval["method_summary"].items():
            rows.append(
                {
                    "method_id": method_id,
                    "score": composite_score(summary["average_metrics"], weights),
                }
            )
        rows.sort(key=lambda row: row["score"], reverse=True)
        runs.append(
            {
                "variant_id": variant_id,
                "weights": weights,
                "leaderboard": [
                    {"rank": index, **row}
                    for index, row in enumerate(rows, start=1)
                ],
            }
        )
    report = {
        "mode": mode,
        "method_set_id": method_set_id,
        "variant_count": len(runs),
        "variants": runs,
        "top_methods_by_variant": {
            run["variant_id"]: run["leaderboard"][0]["method_id"] for run in runs if run["leaderboard"]
        },
    }
    write_json(settings.reports_dir / f"weight_tuning_{mode}_{method_set_id}.json", report)
    return report


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune CANON method-eval score weights over fixed runs.")
    parser.add_argument("--mode", default="social_science_ir_v1_harvest10")
    parser.add_argument("--method-set-id", default="baseline_methods_v1")
    args = parser.parse_args()
    print(json.dumps(tune_weights(args.mode, args.method_set_id), indent=2))


if __name__ == "__main__":
    main()
