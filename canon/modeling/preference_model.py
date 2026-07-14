from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from canon.config import load_settings
from canon.eval.hard_negatives import anchor_preference_examples
from canon.eval.technical_calibration import evaluate_technical_calibration
from canon.labeling.calibration import calibrate_label_tasks, load_tasks
from canon.modeling.calibration_model import (
    FEATURE_NAMES,
    CalibrationExample,
    LinearCalibrationModel,
    average,
    examples_from_label_calibration,
    examples_from_technical_report,
    feature_means,
    feature_scales,
    ranked_weights,
    scaled,
    solve_ridge,
    target_source_name,
    write_json,
)


@dataclass(frozen=True)
class PairwisePreferenceExample:
    pair_id: str
    query_id: str
    better_policy: str
    worse_policy: str
    delta_features: dict[str, float]
    margin: float
    target: float
    target_source: str


@dataclass(frozen=True)
class PairwisePreferenceModel:
    intercept: float
    weights: dict[str, float]
    feature_means: dict[str, float]
    feature_scales: dict[str, float]
    ridge_lambda: float
    target_source: str

    def preference_probability(self, delta_features: dict[str, float]) -> float:
        value = self.intercept
        for name, weight in self.weights.items():
            normalized = (float(delta_features.get(name, 0.0)) - self.feature_means.get(name, 0.0)) / self.feature_scales.get(name, 1.0)
            value += normalized * weight
        return max(0.0, min(1.0, round(value, 6)))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_preference_report_from_technical(
    mode: str,
    policies: list[str],
    top_k: int | None = None,
    ridge_lambda: float = 0.75,
    min_margin: float = 0.025,
) -> dict[str, Any]:
    technical_report = evaluate_technical_calibration(mode, policies, top_k)
    examples = examples_from_technical_report(technical_report)
    return preference_report(
        examples=examples,
        mode=mode,
        source_report="technical_calibration",
        ridge_lambda=ridge_lambda,
        min_margin=min_margin,
        output_name=f"preference_model_{mode}_technical.json",
    )


def build_preference_report_from_labels(
    mode: str,
    tasks_path: Path | None = None,
    ridge_lambda: float = 0.75,
    min_margin: float = 0.025,
) -> dict[str, Any]:
    settings = load_settings()
    path = tasks_path or settings.reports_dir / f"label_tasks_{mode}.json"
    label_report = calibrate_label_tasks(load_tasks(path))
    examples = examples_from_label_calibration(label_report)
    return preference_report(
        examples=examples,
        mode=mode,
        source_report="label_calibration",
        ridge_lambda=ridge_lambda,
        min_margin=min_margin,
        output_name=f"preference_model_{mode}_labels.json",
    )


def build_preference_report_from_anchors(
    mode: str = "hard_negative_anchor_preferences_v1",
    ridge_lambda: float = 0.75,
    min_margin: float = 0.025,
) -> dict[str, Any]:
    examples = anchor_preference_examples()
    return preference_report(
        examples=examples,
        mode=mode,
        source_report="hard_negative_anchor_preferences",
        ridge_lambda=ridge_lambda,
        min_margin=min_margin,
        output_name=f"preference_model_{mode}_anchors.json",
    )


def preference_report(
    examples: list[CalibrationExample],
    mode: str,
    source_report: str,
    ridge_lambda: float,
    min_margin: float,
    output_name: str,
) -> dict[str, Any]:
    pairs = pairwise_examples(examples, min_margin=min_margin)
    model = fit_pairwise_model(pairs, ridge_lambda=ridge_lambda)
    rows = pair_prediction_rows(model, pairs)
    loo = leave_one_query_out_pairs(pairs, ridge_lambda=ridge_lambda)
    report = {
        "mode": mode,
        "source_report": source_report,
        "example_count": len(examples),
        "pair_count": len(pairs),
        "unique_comparison_count": len(pairs) // 2,
        "min_margin": min_margin,
        "target_source_counts": source_counts(pairs),
        "model": model.to_dict(),
        "top_positive_features": ranked_weights(model, reverse=True)[:8],
        "top_negative_features": ranked_weights(model, reverse=False)[:8],
        "in_sample_metrics": pairwise_metrics(rows),
        "leave_one_query_out_metrics": loo["metrics"],
        "leave_one_query_out_rows": loo["rows"],
        "policy_target_summary": policy_target_summary(examples),
        "pair_predictions": rows,
        "limitations": preference_limitations(examples),
    }
    settings = load_settings()
    write_json(settings.reports_dir / output_name, report)
    return report


def pairwise_examples(examples: list[CalibrationExample], min_margin: float = 0.025) -> list[PairwisePreferenceExample]:
    grouped: dict[str, list[CalibrationExample]] = {}
    for example in examples:
        grouped.setdefault(example.query_id, []).append(example)
    pairs: list[PairwisePreferenceExample] = []
    for query_id, group in sorted(grouped.items()):
        ordered = sorted(group, key=lambda item: (item.policy, item.task_id))
        for left_index, left in enumerate(ordered):
            for right in ordered[left_index + 1 :]:
                margin = left.target - right.target
                if abs(margin) < min_margin:
                    continue
                better, worse = (left, right) if margin > 0 else (right, left)
                abs_margin = abs(margin)
                delta = feature_delta(better.features, worse.features)
                source = target_source_name([better, worse])
                pair_id = f"{query_id}::{better.policy}>{worse.policy}"
                pairs.append(
                    PairwisePreferenceExample(
                        pair_id=pair_id,
                        query_id=query_id,
                        better_policy=better.policy,
                        worse_policy=worse.policy,
                        delta_features=delta,
                        margin=round(abs_margin, 6),
                        target=1.0,
                        target_source=source,
                    )
                )
                pairs.append(
                    PairwisePreferenceExample(
                        pair_id=f"{query_id}::{worse.policy}>{better.policy}",
                        query_id=query_id,
                        better_policy=worse.policy,
                        worse_policy=better.policy,
                        delta_features=feature_delta(worse.features, better.features),
                        margin=round(abs_margin, 6),
                        target=0.0,
                        target_source=source,
                    )
                )
    return pairs


def feature_delta(left: dict[str, float], right: dict[str, float]) -> dict[str, float]:
    return {name: float(left.get(name, 0.0)) - float(right.get(name, 0.0)) for name in FEATURE_NAMES}


def fit_pairwise_model(pairs: list[PairwisePreferenceExample], ridge_lambda: float = 0.75) -> PairwisePreferenceModel:
    if not pairs:
        return PairwisePreferenceModel(0.5, {name: 0.0 for name in FEATURE_NAMES}, {}, {}, ridge_lambda, "none")
    means = pair_feature_means(pairs)
    scales = pair_feature_scales(pairs, means)
    x_rows = [[1.0, *[scaled(pair.delta_features, name, means, scales) for name in FEATURE_NAMES]] for pair in pairs]
    y = [pair.target for pair in pairs]
    coefficients = solve_ridge(x_rows, y, ridge_lambda)
    return PairwisePreferenceModel(
        intercept=round(coefficients[0], 6),
        weights={name: round(coefficients[index + 1], 6) for index, name in enumerate(FEATURE_NAMES)},
        feature_means={name: round(means[name], 6) for name in FEATURE_NAMES},
        feature_scales={name: round(scales[name], 6) for name in FEATURE_NAMES},
        ridge_lambda=ridge_lambda,
        target_source=pair_target_source_name(pairs),
    )


def pair_feature_means(pairs: list[PairwisePreferenceExample]) -> dict[str, float]:
    pseudo_examples = [
        CalibrationExample(pair.pair_id, pair.query_id, pair.better_policy, pair.delta_features, pair.target, pair.target_source, pair.target)
        for pair in pairs
    ]
    return feature_means(pseudo_examples)


def pair_feature_scales(pairs: list[PairwisePreferenceExample], means: dict[str, float]) -> dict[str, float]:
    pseudo_examples = [
        CalibrationExample(pair.pair_id, pair.query_id, pair.better_policy, pair.delta_features, pair.target, pair.target_source, pair.target)
        for pair in pairs
    ]
    return feature_scales(pseudo_examples, means)


def pair_prediction_rows(model: PairwisePreferenceModel, pairs: list[PairwisePreferenceExample]) -> list[dict[str, Any]]:
    rows = []
    for pair in pairs:
        prediction = model.preference_probability(pair.delta_features)
        rows.append(
            {
                "pair_id": pair.pair_id,
                "query_id": pair.query_id,
                "better_policy": pair.better_policy,
                "worse_policy": pair.worse_policy,
                "target": pair.target,
                "prediction": prediction,
                "predicted_preference": 1.0 if prediction >= 0.5 else 0.0,
                "correct": 1.0 if (prediction >= 0.5) == (pair.target >= 0.5) else 0.0,
                "margin": pair.margin,
                "target_source": pair.target_source,
            }
        )
    return rows


def leave_one_query_out_pairs(pairs: list[PairwisePreferenceExample], ridge_lambda: float) -> dict[str, Any]:
    query_ids = sorted({pair.query_id for pair in pairs})
    if len(query_ids) <= 1:
        model = fit_pairwise_model(pairs, ridge_lambda)
        rows = pair_prediction_rows(model, pairs)
    else:
        rows = []
        for query_id in query_ids:
            train = [pair for pair in pairs if pair.query_id != query_id]
            test = [pair for pair in pairs if pair.query_id == query_id]
            model = fit_pairwise_model(train, ridge_lambda)
            rows.extend(pair_prediction_rows(model, test))
    return {"metrics": pairwise_metrics(rows), "rows": rows}


def pairwise_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {"accuracy": 0.0, "positive_accuracy": 0.0, "negative_accuracy": 0.0, "mean_margin": 0.0, "mean_confidence": 0.0}
    positives = [row for row in rows if float(row["target"]) >= 0.5]
    negatives = [row for row in rows if float(row["target"]) < 0.5]
    return {
        "accuracy": average(float(row["correct"]) for row in rows),
        "positive_accuracy": average(float(row["correct"]) for row in positives),
        "negative_accuracy": average(float(row["correct"]) for row in negatives),
        "mean_margin": average(float(row["margin"]) for row in rows),
        "mean_confidence": average(abs(float(row["prediction"]) - 0.5) * 2.0 for row in rows),
    }


def policy_target_summary(examples: list[CalibrationExample]) -> list[dict[str, Any]]:
    grouped: dict[str, list[CalibrationExample]] = {}
    for example in examples:
        grouped.setdefault(example.policy, []).append(example)
    rows = []
    for policy, items in sorted(grouped.items()):
        rows.append(
            {
                "policy": policy,
                "count": len(items),
                "average_target": average(item.target for item in items),
                "average_proxy_baseline": average(item.proxy_baseline for item in items),
                "target_source_counts": counts(item.target_source for item in items),
            }
        )
    return rows


def source_counts(pairs: list[PairwisePreferenceExample]) -> dict[str, int]:
    return counts(pair.target_source for pair in pairs)


def preference_limitations(examples: list[CalibrationExample]) -> list[str]:
    sources = {example.target_source for example in examples}
    limitations = ["Pairwise preferences are only as valid as the target used to order policies."]
    if "technical_proxy" in sources:
        limitations.append("With technical-proxy targets, this tests internal ranking consistency rather than real-world quality.")
    if "hard_negative_anchor" in sources:
        limitations.append("With hard-negative anchors, this tests known failure modes rather than broad source-quality validity.")
    if "human_label" in sources:
        limitations.append("Human-label targets still need inter-annotator agreement and domain-slice analysis before weights become policy.")
    limitations.extend(
        [
            "Pairwise models improve ranking diagnostics but do not replace safety gates, source quarantine, or domain-specific trust rules.",
            "Small query sets can make policy win rates unstable; treat this as a regression diagnostic until larger human and external anchors are added.",
        ]
    )
    return limitations


def pair_target_source_name(pairs: list[PairwisePreferenceExample]) -> str:
    sources = sorted({pair.target_source for pair in pairs})
    return "+".join(sources)


def counts(values) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        result[str(value)] = result.get(str(value), 0) + 1
    return dict(sorted(result.items()))


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit transparent pairwise preference models over CANON calibration targets.")
    parser.add_argument("--mode", default="dry_run")
    parser.add_argument("--source", choices=["technical", "labels", "anchors"], default="technical")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--policies", default="lexical,balanced,semantic,rag")
    parser.add_argument("--tasks", default=None)
    parser.add_argument("--ridge-lambda", type=float, default=0.75)
    parser.add_argument("--min-margin", type=float, default=0.025)
    args = parser.parse_args()
    if args.source == "anchors":
        report = build_preference_report_from_anchors(
            mode=args.mode,
            ridge_lambda=args.ridge_lambda,
            min_margin=args.min_margin,
        )
    elif args.source == "labels":
        report = build_preference_report_from_labels(
            mode=args.mode,
            tasks_path=Path(args.tasks) if args.tasks else None,
            ridge_lambda=args.ridge_lambda,
            min_margin=args.min_margin,
        )
    else:
        policies = [policy.strip() for policy in args.policies.split(",") if policy.strip()]
        report = build_preference_report_from_technical(
            mode=args.mode,
            policies=policies,
            top_k=args.top_k,
            ridge_lambda=args.ridge_lambda,
            min_margin=args.min_margin,
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
