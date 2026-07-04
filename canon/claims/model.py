from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from canon.config import load_settings
from canon.retrieval.tokenize import tokenize


@dataclass(frozen=True)
class ClaimPrediction:
    model_id: str
    is_claim: bool
    confidence: float
    stance: str
    features: dict[str, float]
    explanation: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


class ClaimModel:
    def __init__(self, artifact: dict[str, Any]) -> None:
        self.artifact = artifact
        self.model_id = artifact["model_id"]
        self.claim_threshold = float(artifact["claim_threshold"])
        self.confidence_intercept = float(artifact["confidence_intercept"])
        self.feature_weights = {
            key: float(value)
            for key, value in artifact["feature_weights"].items()
        }
        self.marker_sets = {
            "positive": set(artifact["positive_markers"]),
            "negative": set(artifact["negative_markers"]),
            "finding": set(artifact["finding_markers"]),
            "evidence": set(artifact["evidence_markers"]),
            "method": set(artifact["method_markers"]),
            "hedge": set(artifact["hedge_markers"]),
        }

    def predict(self, text: str) -> ClaimPrediction:
        features = self.features(text)
        raw_score = self.confidence_intercept + sum(
            features.get(name, 0.0) * weight
            for name, weight in self.feature_weights.items()
        )
        confidence = round(max(0.0, min(1.0, raw_score)), 6)
        stance = self.stance(features)
        return ClaimPrediction(
            model_id=self.model_id,
            is_claim=confidence >= self.claim_threshold,
            confidence=confidence,
            stance=stance,
            features=features,
            explanation=self.explain(features, confidence, stance),
        )

    def features(self, text: str) -> dict[str, float]:
        tokens = tokenize(text)
        token_set = set(tokens)
        return {
            "claim_marker_count": float(
                len(
                    token_set
                    & (
                        self.marker_sets["positive"]
                        | self.marker_sets["negative"]
                        | self.marker_sets["finding"]
                        | self.marker_sets["evidence"]
                    )
                )
            ),
            "finding_marker_count": float(len(token_set & self.marker_sets["finding"])),
            "evidence_marker_count": float(len(token_set & self.marker_sets["evidence"])),
            "method_marker_count": float(len(token_set & self.marker_sets["method"])),
            "hedge_marker_count": float(len(token_set & self.marker_sets["hedge"])),
            "positive_marker_count": float(len(token_set & self.marker_sets["positive"])),
            "negative_marker_count": float(len(token_set & self.marker_sets["negative"])),
            "sentence_token_count": float(min(len(tokens), 60)),
        }

    @staticmethod
    def stance(features: dict[str, float]) -> str:
        positive = features.get("positive_marker_count", 0.0)
        negative = features.get("negative_marker_count", 0.0)
        if positive > negative:
            return "supportive"
        if negative > positive:
            return "critical"
        if positive or negative:
            return "mixed"
        return "descriptive"

    def explain(self, features: dict[str, float], confidence: float, stance: str) -> list[str]:
        explanation = [f"stance={stance}", f"confidence={confidence:.3f}"]
        for name, weight in sorted(self.feature_weights.items()):
            value = features.get(name, 0.0)
            if value:
                explanation.append(f"{name}={value:g} * {weight:g}")
        return explanation


def load_claim_model(path: Path | None = None) -> ClaimModel:
    settings = load_settings()
    model_path = path or settings.root / "conf" / "models" / "claim_model_baseline.json"
    artifact = json.loads(model_path.read_text(encoding="utf-8"))
    return ClaimModel(artifact)


def model_report(texts: list[str]) -> dict:
    model = load_claim_model()
    predictions = [model.predict(text).to_dict() | {"text": text} for text in texts]
    return {
        "model_id": model.model_id,
        "model_type": model.artifact["model_type"],
        "claim_threshold": model.claim_threshold,
        "predictions": predictions,
    }


def write_model_report(mode: str, texts: list[str]) -> dict:
    settings = load_settings()
    report = model_report(texts)
    report["mode"] = mode
    report["candidate_count"] = len(texts)
    report["claim_count"] = sum(
        1 for prediction in report["predictions"] if prediction["is_claim"]
    )
    output_path = settings.reports_dir / f"claim_model_{mode}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect claim model predictions.")
    parser.add_argument("--mode", default="inspect")
    parser.add_argument("--write-report", action="store_true")
    parser.add_argument("text", nargs="*")
    args = parser.parse_args()
    texts = args.text or [
        "We find robust support for the democratic peace.",
        "This article provides background on international relations theory.",
    ]
    report = write_model_report(args.mode, texts) if args.write_report else model_report(texts)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
