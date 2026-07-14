from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from typing import Any

from canon.config import load_settings
from canon.eval.contracts import CONTRACT_VERSION, validate_report


@dataclass(frozen=True)
class EvaluationAnchor:
    anchor_id: str
    anchor_type: str
    domain: str
    status: str
    purpose: str
    current_artifact: str | None
    limitation: str
    next_step: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def anchor_registry() -> list[EvaluationAnchor]:
    return [
        EvaluationAnchor(
            anchor_id="human_evidence_role_labels_v1",
            anchor_type="human_labeling",
            domain="cross_domain",
            status="task_scaffold_ready",
            purpose="Validate method, limitation, claim, context, contradiction, and provenance role labels.",
            current_artifact="reports/unstructured_human_label_tasks_v1.json",
            limitation="The task scaffold exists, but no completed expert/human labels have been collected yet.",
            next_step="Assign the stratified task pack for expert labeling and compare completed labels against deterministic evidence-role signals.",
        ),
        EvaluationAnchor(
            anchor_id="human_source_trust_labels_v1",
            anchor_type="human_labeling",
            domain="cross_domain",
            status="task_scaffold_ready",
            purpose="Validate source trust tiers and domain-specific trust profiles.",
            current_artifact="reports/unstructured_human_label_tasks_v1.json",
            limitation="The task scaffold exists, but source-trust labels are still empty and not expert-calibrated.",
            next_step="Collect expert labels for strong, moderate, limited, low, and blocked source trust across domains and source forms.",
        ),
        EvaluationAnchor(
            anchor_id="human_chunking_quality_labels_v1",
            anchor_type="human_labeling",
            domain="cross_domain",
            status="task_scaffold_ready",
            purpose="Validate whether chunking strategies avoid harmful merges, preserve qualifiers, control fragmentation, and support safe evidence use.",
            current_artifact="reports/unstructured_human_label_tasks_v1.json",
            limitation="The task scaffold exists, but chunking-quality labels are still empty and deterministic variant scores are only proxy diagnostics.",
            next_step="Collect stratified labels comparing fixed-window, boundary-aware, atomic sentence, semantic-cohesion, and paragraph-or-sentence variants across domains and document forms.",
        ),
        EvaluationAnchor(
            anchor_id="public_beir_scifact_smoke",
            anchor_type="external_benchmark",
            domain="scientific_retrieval",
            status="smoke_only",
            purpose="Exercise qrels parsing and BEIR/TREC-style metric plumbing.",
            current_artifact="reports/qrels_validation_public_beir_scifact_smoke.json",
            limitation="Smoke fixture does not support public benchmark performance claims.",
            next_step="Run a full public BEIR/TREC-style benchmark with qrels.",
        ),
        EvaluationAnchor(
            anchor_id="adversarial_rag_security_v1",
            anchor_type="adversarial_benchmark",
            domain="rag_security",
            status="active_deterministic",
            purpose="Measure whether prompt-injection attacks reach candidate retrieval, final results, context, and generated answer.",
            current_artifact="reports/adversarial_rag_security_v1.json",
            limitation="Synthetic deterministic fixture; no live adaptive LLM attacker or HTML/PDF extraction attacks yet.",
            next_step="Add public prompt-injection benchmarks, obfuscation variants, and model-based attack-following evaluation.",
        ),
        EvaluationAnchor(
            anchor_id="adversarial_corroboration_v1",
            anchor_type="adversarial_benchmark",
            domain="corroboration",
            status="active_deterministic",
            purpose="Detect dependent echo and distinguish support count from independent support count.",
            current_artifact="reports/adversarial_corroboration_v1.json",
            limitation="Synthetic fixture with lexical claim grouping and simple source fingerprints.",
            next_step="Add paraphrase, multilingual, and hidden-source-dependence attacks.",
        ),
        EvaluationAnchor(
            anchor_id="hard_negative_anchor_preferences_v1",
            anchor_type="hard_negative_preference_set",
            domain="cross_domain",
            status="active_deterministic",
            purpose="Stress-test ranking models against known bad preferences such as prompt-injection exposure, echo-count inflation, and source accessibility masquerading as quality.",
            current_artifact="reports/hard_negative_anchor_preferences_v1.json",
            limitation="Small deterministic anchor set; useful for regression and failure-mode checks, not sufficient for broad quality claims.",
            next_step="Expand with expert-labeled hard negatives, public prompt-injection corpora, legal/market authority cases, and domain-specific source-quality comparisons.",
        ),
    ]


def build_anchor_report() -> dict[str, Any]:
    anchors = [anchor.to_dict() for anchor in anchor_registry()]
    missing = [anchor for anchor in anchors if anchor["status"] in {"planned", "smoke_only", "task_scaffold_ready"}]
    active = [anchor for anchor in anchors if anchor["status"].startswith("active")]
    report = {
        "contract_version": CONTRACT_VERSION,
        "report_id": "evaluation_anchor_registry_v1",
        "anchor_count": len(anchors),
        "active_anchor_count": len(active),
        "missing_or_limited_anchor_count": len(missing),
        "anchors": anchors,
        "claim_limits": [
            "Do not claim validated source trust until human/expert trust labels exist.",
            "Do not claim validated evidence-role classification until cross-domain human labels exist.",
            "Do not claim validated chunking quality until chunk-level human labels confirm harmful-merge, qualifier-preservation, and fragmentation judgments.",
            "Do not claim public retrieval benchmark performance from smoke qrels.",
            "Do not claim prompt-injection robustness beyond the tested deterministic fixture.",
        ],
    }
    report["contract_validation"] = validate_report(report, "evaluation_anchors")
    return report


def write_report() -> dict[str, Any]:
    settings = load_settings()
    report = build_anchor_report()
    path = settings.reports_dir / "evaluation_anchor_registry_v1.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build evaluation anchor registry.")
    parser.add_argument("--write-report", action="store_true")
    args = parser.parse_args()
    report = write_report() if args.write_report else build_anchor_report()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
