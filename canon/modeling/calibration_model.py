from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from canon.config import load_settings
from canon.eval.technical_calibration import TECHNICAL_DIMENSIONS, evaluate_technical_calibration
from canon.labeling.calibration import LABEL_ORDER, average, calibrate_label_tasks, load_tasks


FEATURE_NAMES = [
    "citation_support_score",
    "answer_faithfulness_score",
    "importance_fit_score",
    "paradigm_coverage_score",
    "evidence_role_fit_score",
    "source_trust_fit_score",
    "safety_handling_score",
    "corroboration_handling_score",
    "citation_count",
    "claim_citation_count",
    "conflict_note_count",
    "distinct_sources",
    "distinct_clusters",
    "relevance",
    "semantic_similarity",
    "syntax_relevance",
    "source_quality",
    "method_signal",
    "limitation_signal",
    "evidence_specificity",
    "uncertainty_signal",
]


@dataclass(frozen=True)
class CalibrationExample:
    task_id: str
    query_id: str
    policy: str
    features: dict[str, float]
    target: float
    target_source: str
    proxy_baseline: float


@dataclass(frozen=True)
class LinearCalibrationModel:
    intercept: float
    weights: dict[str, float]
    feature_means: dict[str, float]
    feature_scales: dict[str, float]
    ridge_lambda: float
    target_source: str

    def predict(self, features: dict[str, float]) -> float:
        value = self.intercept
        for name, weight in self.weights.items():
            scaled = (float(features.get(name, 0.0)) - self.feature_means.get(name, 0.0)) / self.feature_scales.get(name, 1.0)
            value += scaled * weight
        return clamp(value)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_modeling_report_from_technical(
    mode: str,
    policies: list[str],
    top_k: int | None = None,
    ridge_lambda: float = 0.75,
) -> dict[str, Any]:
    technical_report = evaluate_technical_calibration(mode, policies, top_k)
    examples = examples_from_technical_report(technical_report)
    return modeling_report(
        examples=examples,
        mode=mode,
        source_report="technical_calibration",
        ridge_lambda=ridge_lambda,
        output_name=f"calibration_model_{mode}_technical.json",
    )


def build_modeling_report_from_labels(
    mode: str,
    tasks_path: Path | None = None,
    ridge_lambda: float = 0.75,
) -> dict[str, Any]:
    settings = load_settings()
    path = tasks_path or settings.reports_dir / f"label_tasks_{mode}.json"
    label_report = calibrate_label_tasks(load_tasks(path))
    examples = examples_from_label_calibration(label_report)
    return modeling_report(
        examples=examples,
        mode=mode,
        source_report="label_calibration",
        ridge_lambda=ridge_lambda,
        output_name=f"calibration_model_{mode}_labels.json",
    )


def modeling_report(
    examples: list[CalibrationExample],
    mode: str,
    source_report: str,
    ridge_lambda: float,
    output_name: str,
) -> dict[str, Any]:
    model = fit_ridge_model(examples, ridge_lambda=ridge_lambda)
    predictions = prediction_rows(model, examples)
    loo = leave_one_query_out(examples, ridge_lambda=ridge_lambda)
    report = {
        "mode": mode,
        "source_report": source_report,
        "example_count": len(examples),
        "target_source_counts": counts(example.target_source for example in examples),
        "feature_names": FEATURE_NAMES,
        "model": model.to_dict(),
        "top_positive_features": ranked_weights(model, reverse=True)[:8],
        "top_negative_features": ranked_weights(model, reverse=False)[:8],
        "in_sample_metrics": regression_metrics(predictions),
        "leave_one_query_out_metrics": loo["metrics"],
        "leave_one_query_out_rows": loo["rows"],
        "predictions": predictions,
        "limitations": [
            "This is a transparent calibration model, not a validated truth model.",
            "When no human labels exist, targets are weak technical proxy scores.",
            "Feature weights are diagnostic and may change under domain-specific labels.",
            "Leave-one-query-out is small-sample stress testing, not a final generalization estimate.",
        ],
    }
    settings = load_settings()
    write_json(settings.reports_dir / output_name, report)
    return report


def examples_from_technical_report(report: dict[str, Any]) -> list[CalibrationExample]:
    return [
        CalibrationExample(
            task_id=row["task_id"],
            query_id=row["query_id"],
            policy=row["policy"],
            features=features_from_row(row),
            target=float(row.get("technical_calibration_score", 0.0)),
            target_source="technical_proxy",
            proxy_baseline=float(row.get("technical_calibration_score", 0.0)),
        )
        for row in report.get("rows", [])
    ]


def examples_from_label_calibration(report: dict[str, Any]) -> list[CalibrationExample]:
    examples = []
    for row in report.get("rows", []):
        technical_labels = row.get("technical_labels") or {}
        proxy = label_score(technical_labels)
        human_labels = row.get("human_labels") or {}
        if human_labels:
            target = label_score(human_labels)
            source = "human_label"
        else:
            target = proxy
            source = "technical_proxy"
        examples.append(
            CalibrationExample(
                task_id=row.get("task_id", ""),
                query_id=row.get("query_id", ""),
                policy=row.get("policy", ""),
                features=features_from_labels(technical_labels),
                target=target,
                target_source=source,
                proxy_baseline=proxy,
            )
        )
    return examples


def features_from_row(row: dict[str, Any]) -> dict[str, float]:
    features = {name: 0.0 for name in FEATURE_NAMES}
    dimension_scores = row.get("dimension_scores") or {}
    for dimension in TECHNICAL_DIMENSIONS:
        features[f"{dimension}_score"] = float(dimension_scores.get(dimension, 0.0))
    metrics = row.get("metrics") or {}
    features["citation_count"] = min(float(metrics.get("citation_count", 0.0) or 0.0) / 5.0, 1.0)
    features["claim_citation_count"] = min(float(metrics.get("claim_citation_count", 0.0) or 0.0) / 5.0, 1.0)
    features["conflict_note_count"] = min(float(metrics.get("conflict_note_count", 0.0) or 0.0) / 3.0, 1.0)
    features["distinct_sources"] = min(float(metrics.get("distinct_sources", 0.0) or 0.0) / 5.0, 1.0)
    features["distinct_clusters"] = min(float(metrics.get("distinct_clusters", 0.0) or 0.0) / 5.0, 1.0)
    components = row.get("average_components") or metrics.get("average_components") or {}
    for name in ["relevance", "semantic_similarity", "syntax_relevance", "source_quality", "method_signal", "limitation_signal", "evidence_specificity", "uncertainty_signal"]:
        features[name] = float(components.get(name, 0.0) or 0.0)
    return features


def features_from_labels(labels: dict[str, str]) -> dict[str, float]:
    features = {name: 0.0 for name in FEATURE_NAMES}
    for dimension in TECHNICAL_DIMENSIONS:
        features[f"{dimension}_score"] = normalized_label_value(dimension, labels.get(dimension))
    return features


def normalized_label_value(label: str, value: str | None) -> float:
    order = LABEL_ORDER.get(label) or []
    if value not in order or not order:
        return 0.0
    if len(order) == 1:
        return 1.0
    return order.index(value) / (len(order) - 1)


def label_score(labels: dict[str, str]) -> float:
    values = [
        normalized_label_value(label, value)
        for label, value in labels.items()
        if label in LABEL_ORDER and value is not None
    ]
    return average(values)


def fit_ridge_model(examples: list[CalibrationExample], ridge_lambda: float = 0.75) -> LinearCalibrationModel:
    if not examples:
        return LinearCalibrationModel(0.0, {name: 0.0 for name in FEATURE_NAMES}, {}, {}, ridge_lambda, "none")
    means = feature_means(examples)
    scales = feature_scales(examples, means)
    x_rows = [[1.0, *[scaled(example.features, name, means, scales) for name in FEATURE_NAMES]] for example in examples]
    y = [example.target for example in examples]
    coefficients = solve_ridge(x_rows, y, ridge_lambda)
    return LinearCalibrationModel(
        intercept=round(coefficients[0], 6),
        weights={name: round(coefficients[index + 1], 6) for index, name in enumerate(FEATURE_NAMES)},
        feature_means={name: round(means[name], 6) for name in FEATURE_NAMES},
        feature_scales={name: round(scales[name], 6) for name in FEATURE_NAMES},
        ridge_lambda=ridge_lambda,
        target_source=target_source_name(examples),
    )


def solve_ridge(x_rows: list[list[float]], y: list[float], ridge_lambda: float) -> list[float]:
    size = len(x_rows[0])
    xtx = [[0.0 for _ in range(size)] for _ in range(size)]
    xty = [0.0 for _ in range(size)]
    for row, target in zip(x_rows, y, strict=True):
        for i in range(size):
            xty[i] += row[i] * target
            for j in range(size):
                xtx[i][j] += row[i] * row[j]
    for i in range(1, size):
        xtx[i][i] += ridge_lambda
    return gaussian_solve(xtx, xty)


def gaussian_solve(matrix: list[list[float]], vector: list[float]) -> list[float]:
    n = len(vector)
    augmented = [row[:] + [vector[index]] for index, row in enumerate(matrix)]
    for column in range(n):
        pivot = max(range(column, n), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-12:
            augmented[column][column] += 1e-8
            pivot = column
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        for j in range(column, n + 1):
            augmented[column][j] /= divisor
        for row in range(n):
            if row == column:
                continue
            factor = augmented[row][column]
            for j in range(column, n + 1):
                augmented[row][j] -= factor * augmented[column][j]
    return [augmented[i][n] for i in range(n)]


def feature_means(examples: list[CalibrationExample]) -> dict[str, float]:
    return {
        name: sum(example.features.get(name, 0.0) for example in examples) / len(examples)
        for name in FEATURE_NAMES
    }


def feature_scales(examples: list[CalibrationExample], means: dict[str, float]) -> dict[str, float]:
    scales = {}
    for name in FEATURE_NAMES:
        variance = sum((example.features.get(name, 0.0) - means[name]) ** 2 for example in examples) / len(examples)
        scales[name] = math.sqrt(variance) or 1.0
    return scales


def scaled(features: dict[str, float], name: str, means: dict[str, float], scales: dict[str, float]) -> float:
    return (float(features.get(name, 0.0)) - means.get(name, 0.0)) / scales.get(name, 1.0)


def prediction_rows(model: LinearCalibrationModel, examples: list[CalibrationExample]) -> list[dict[str, Any]]:
    return [
        {
            "task_id": example.task_id,
            "query_id": example.query_id,
            "policy": example.policy,
            "target": round(example.target, 6),
            "prediction": round(model.predict(example.features), 6),
            "proxy_baseline": round(example.proxy_baseline, 6),
            "model_abs_error": round(abs(model.predict(example.features) - example.target), 6),
            "proxy_abs_error": round(abs(example.proxy_baseline - example.target), 6),
            "target_source": example.target_source,
        }
        for example in examples
    ]


def leave_one_query_out(examples: list[CalibrationExample], ridge_lambda: float) -> dict[str, Any]:
    groups = sorted({example.query_id for example in examples})
    rows = []
    if len(groups) <= 1:
        model = fit_ridge_model(examples, ridge_lambda)
        rows = prediction_rows(model, examples)
    else:
        for group in groups:
            train = [example for example in examples if example.query_id != group]
            test = [example for example in examples if example.query_id == group]
            model = fit_ridge_model(train, ridge_lambda)
            rows.extend(prediction_rows(model, test))
    return {"metrics": regression_metrics(rows), "rows": rows}


def regression_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {"model_mae": 0.0, "proxy_mae": 0.0, "model_rmse": 0.0, "proxy_rmse": 0.0, "model_beats_proxy": 0.0}
    model_errors = [float(row["model_abs_error"]) for row in rows]
    proxy_errors = [float(row["proxy_abs_error"]) for row in rows]
    return {
        "model_mae": average(model_errors),
        "proxy_mae": average(proxy_errors),
        "model_rmse": round(math.sqrt(average(error * error for error in model_errors)), 6),
        "proxy_rmse": round(math.sqrt(average(error * error for error in proxy_errors)), 6),
        "model_beats_proxy": 1.0 if average(model_errors) < average(proxy_errors) else 0.0,
    }


def ranked_weights(model: LinearCalibrationModel, reverse: bool) -> list[dict[str, float | str]]:
    rows = [
        {"feature": name, "weight": weight}
        for name, weight in model.weights.items()
        if (weight > 0 if reverse else weight < 0)
    ]
    return sorted(rows, key=lambda row: abs(float(row["weight"])), reverse=True)


def target_source_name(examples: list[CalibrationExample]) -> str:
    sources = sorted({example.target_source for example in examples})
    return "+".join(sources)


def counts(values) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        result[str(value)] = result.get(str(value), 0) + 1
    return dict(sorted(result.items()))


def clamp(value: float) -> float:
    return max(0.0, min(1.0, round(value, 6)))


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit transparent calibration models over CANON labels/proxies.")
    parser.add_argument("--mode", default="dry_run")
    parser.add_argument("--source", choices=["technical", "labels"], default="technical")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--policies", default="lexical,balanced,semantic,rag")
    parser.add_argument("--tasks", default=None)
    parser.add_argument("--ridge-lambda", type=float, default=0.75)
    args = parser.parse_args()
    if args.source == "labels":
        report = build_modeling_report_from_labels(
            mode=args.mode,
            tasks_path=Path(args.tasks) if args.tasks else None,
            ridge_lambda=args.ridge_lambda,
        )
    else:
        policies = [policy.strip() for policy in args.policies.split(",") if policy.strip()]
        report = build_modeling_report_from_technical(
            mode=args.mode,
            policies=policies,
            top_k=args.top_k,
            ridge_lambda=args.ridge_lambda,
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
