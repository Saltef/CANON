from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from canon.config import load_settings
from canon.eval.contracts import CONTRACT_VERSION, validate_report


DEFAULT_PUBLIC_OPINION_QUERY = "public opinion battery storage safety concerns"


def build_unstructured_portfolio_report(
    document_type_mode: str = "unstructured_demo_corpus",
    domain_mode: str = "mixed_domain_demo_corpus",
    public_opinion_mode: str = "social_public_opinion_demo_corpus",
    public_opinion_query: str = DEFAULT_PUBLIC_OPINION_QUERY,
    write_report: bool = True,
) -> dict[str, Any]:
    settings = load_settings()
    artifacts = load_unstructured_portfolio_artifacts(
        reports_dir=settings.reports_dir,
        document_type_mode=document_type_mode,
        domain_mode=domain_mode,
        public_opinion_mode=public_opinion_mode,
        public_opinion_query=public_opinion_query,
    )
    checks = portfolio_checks(artifacts)
    report = {
        "contract_version": CONTRACT_VERSION,
        "report_id": "unstructured_experiment_portfolio_v1",
        "status": "pass" if all(check["passed"] for check in checks) else "fail",
        "modes": {
            "document_type_mode": document_type_mode,
            "domain_mode": domain_mode,
            "public_opinion_mode": public_opinion_mode,
        },
        "checks": checks,
        "artifact_summary": artifact_summary(artifacts),
        "claim_boundaries": claim_boundaries(),
        "recommended_next_steps": recommended_next_steps(checks),
        "limitations": limitations(),
        "artifact_paths": artifact_paths(settings.reports_dir, document_type_mode, domain_mode, public_opinion_mode, public_opinion_query),
    }
    report["contract_validation"] = validate_report(report, "unstructured_experiment_portfolio")
    if write_report:
        path = settings.reports_dir / "unstructured_experiment_portfolio_v1.json"
        write_json(path, report)
        write_markdown(path.with_suffix(".md"), render_markdown(report))
    return report


def load_unstructured_portfolio_artifacts(
    reports_dir: Path,
    document_type_mode: str,
    domain_mode: str,
    public_opinion_mode: str,
    public_opinion_query: str,
) -> dict[str, Any]:
    paths = artifact_paths(reports_dir, document_type_mode, domain_mode, public_opinion_mode, public_opinion_query)
    return {key: load_json(Path(path)) for key, path in paths.items()}


def artifact_paths(
    reports_dir: Path,
    document_type_mode: str,
    domain_mode: str,
    public_opinion_mode: str,
    public_opinion_query: str,
) -> dict[str, str]:
    return {
        "unstructured_readiness": str(
            reports_dir / f"heterogeneous_unstructured_readiness_{document_type_mode}_{domain_mode}.json"
        ),
        "phase_gate": str(reports_dir / "importance_phase_gate_v1.json"),
        "adversarial_security": str(reports_dir / "adversarial_rag_security_v1.json"),
        "chunking": str(reports_dir / "chunking_strategy_eval_ct14_ov0.json"),
        "chunking_large_window": str(reports_dir / "chunking_strategy_eval_ct1000_ov100.json"),
        "chunking_variants": str(reports_dir / "chunking_variant_eval_ct1000_ov100.json"),
        "coverage_matrix": str(reports_dir / "unstructured_experiment_coverage_matrix_v1.json"),
        "public_opinion": str(reports_dir / f"public_opinion_analysis_{public_opinion_mode}.json"),
        "evidence_committee": str(reports_dir / f"evidence_committee_{public_opinion_mode}.json"),
        "committee_gate_comparison": str(reports_dir / "committee_gate_policy_comparison_v1.json"),
        "committee_gate_usefulness": str(reports_dir / "committee_gate_usefulness_audit_v1.json"),
        "syntax_retrieval_ablation": str(reports_dir / "syntax_retrieval_ablation_v1.json"),
        "unstructured_label_tasks": str(reports_dir / "unstructured_human_label_tasks_v1.json"),
        "unstructured_label_calibration": str(reports_dir / "unstructured_label_calibration_v1.json"),
        "public_opinion_synthesis": str(
            reports_dir / f"synthesis_{public_opinion_mode}_rag_{slug(public_opinion_query)}.json"
        ),
        "evaluation_anchors": str(reports_dir / "evaluation_anchor_registry_v1.json"),
    }


def portfolio_checks(artifacts: dict[str, Any]) -> list[dict[str, Any]]:
    readiness = artifacts.get("unstructured_readiness") or {}
    phase_gate = artifacts.get("phase_gate") or {}
    security = artifacts.get("adversarial_security") or {}
    chunking = artifacts.get("chunking") or {}
    chunking_large = artifacts.get("chunking_large_window") or {}
    chunking_variants = artifacts.get("chunking_variants") or {}
    matrix = artifacts.get("coverage_matrix") or {}
    public_opinion = artifacts.get("public_opinion") or {}
    committee = artifacts.get("evidence_committee") or {}
    committee_comparison = artifacts.get("committee_gate_comparison") or {}
    usefulness = artifacts.get("committee_gate_usefulness") or {}
    syntax_ablation = artifacts.get("syntax_retrieval_ablation") or {}
    label_tasks = artifacts.get("unstructured_label_tasks") or {}
    label_calibration = artifacts.get("unstructured_label_calibration") or {}
    synthesis = artifacts.get("public_opinion_synthesis") or {}
    anchors = artifacts.get("evaluation_anchors") or {}
    chunking_summary = chunking.get("summary") or {}
    boundary = chunking_summary.get("boundary_aware") or {}
    fixed = chunking_summary.get("fixed_window") or {}
    large_summary = chunking_large.get("summary") or {}
    large_boundary = large_summary.get("boundary_aware") or {}
    large_fixed = large_summary.get("fixed_window") or {}
    variant_summary = chunking_variants.get("summary") or {}
    variant_fixed = variant_summary.get("fixed_window") or {}
    variant_boundary = variant_summary.get("boundary_aware") or {}
    variant_semantic = variant_summary.get("semantic_cohesion") or {}
    variant_embedding = variant_summary.get("embedding_semantic") or {}
    variant_atomic = variant_summary.get("atomic_sentence") or {}
    public_aggregation = public_opinion.get("aggregation_readiness") or {}
    committee_summary = committee.get("summary") or {}
    committee_gate_runs = [run for run in committee.get("runs") or [] if run.get("policy") == "committee_rag"]
    synthesis_context = synthesis.get("context_packet") or {}
    return [
        check(
            "heterogeneous_readiness_passes",
            readiness.get("status") == "pass"
            and (readiness.get("contract_validation") or validate_report(readiness, "heterogeneous_unstructured_readiness")).get("status") == "pass",
            "Heterogeneous unstructured readiness gate passes and satisfies its contract.",
            readiness.get("artifact_summary"),
        ),
        check(
            "importance_phase_gate_passes",
            phase_gate.get("status") == "pass"
            and (phase_gate.get("contract_validation") or validate_report(phase_gate, "phase_gate")).get("status") == "pass",
            "Importance phases A-H pass their deterministic coverage gate.",
            phase_gate.get("phases"),
        ),
        check(
            "adversarial_security_blocks_context_exposure",
            security.get("status") == "pass"
            and int((security.get("metrics") or {}).get("attack_context_exposure_count") or 0) == 0,
            "Adversarial prompt-injection benchmark has zero attack context exposure.",
            security.get("metrics"),
        ),
        check(
            "boundary_chunking_beats_fixed_windows_on_span_containment",
            float(boundary.get("average_span_containment_rate") or 0.0)
            >= float(fixed.get("average_span_containment_rate") or 0.0),
            "Boundary-aware chunking preserves expected evidence spans at least as well as fixed windows.",
            {"boundary_aware": boundary, "fixed_window": fixed},
        ),
        check(
            "boundary_chunking_isolates_unsafe_social_span",
            int(boundary.get("unsafe_isolated_count") or 0) >= 1
            and int(fixed.get("unsafe_isolated_count") or 0) == 0,
            "Boundary-aware chunking isolates the unsafe social prompt-injection span in the fixture.",
            {"boundary_unsafe_isolated_count": boundary.get("unsafe_isolated_count"), "fixed_unsafe_isolated_count": fixed.get("unsafe_isolated_count")},
        ),
        check(
            "large_window_chunking_isolates_mixed_unsafe_social_span",
            chunking_large.get("report_id") == "chunking_strategy_eval_v1"
            and int(chunking_large.get("chunk_tokens") or 0) >= 1000
            and int(large_boundary.get("unsafe_isolated_count") or 0) >= 1
            and int(large_fixed.get("unsafe_isolated_count") or 0) == 0,
            "Large-window chunking probe catches mixed benign-plus-injection context that fixed windows fail to isolate.",
            {
                "chunk_tokens": chunking_large.get("chunk_tokens"),
                "overlap_tokens": chunking_large.get("overlap_tokens"),
                "boundary_aware": large_boundary,
                "fixed_window": large_fixed,
            },
        ),
        check(
            "chunking_variant_comparison_tracks_tradeoffs",
            chunking_variants.get("report_id") == "chunking_variant_eval_v1"
            and (chunking_variants.get("contract_validation") or validate_report(chunking_variants, "chunking_variant_eval")).get("status") == "pass"
            and int(variant_fixed.get("forbidden_merge_count") or 0) > 0
            and int(variant_boundary.get("forbidden_merge_count") or 0) == 0
            and int(variant_semantic.get("forbidden_merge_count") or 0) == 0
            and int(variant_embedding.get("forbidden_merge_count") or 0) == 0
            and int(variant_atomic.get("forbidden_merge_count") or 0) == 0
            and float(((variant_boundary.get("objective_scores") or {}).get("balanced_utility")) or 0.0)
            >= float(((variant_fixed.get("objective_scores") or {}).get("balanced_utility")) or 0.0),
            "Chunking variant benchmark compares fixed, production, atomic, lexical semantic-surrogate, and local embedding-semantic strategies and exposes safety/usefulness tradeoffs.",
            {
                "best_variant_by_objective": chunking_variants.get("best_variant_by_objective"),
                "fixed_window": variant_fixed,
                "boundary_aware": variant_boundary,
                "semantic_cohesion": variant_semantic,
                "embedding_semantic": variant_embedding,
                "atomic_sentence": variant_atomic,
            },
        ),
        check(
            "coverage_matrix_exposes_expansion_and_label_gaps",
            matrix.get("report_id") == "unstructured_experiment_coverage_matrix_v1"
            and matrix.get("contract_validation", {}).get("status") == "pass"
            and matrix.get("status") in {"needs_expansion", "needs_human_labels", "fixture_ready_needs_scale", "ready_for_broad_validation"}
            and (matrix.get("gap_summary") or {}).get("ready_for_broad_validity_claims") is False,
            "Coverage matrix is present and explicitly blocks broad validity claims while gaps remain.",
            matrix.get("gap_summary"),
        ),
        check(
            "public_opinion_report_keeps_inference_boundary",
            public_opinion.get("report_id") == "public_opinion_analysis_v1"
            and public_aggregation.get("status") in {"fixture_or_insufficient_for_inference", "ready_for_exploratory_aggregation"},
            "Public-opinion analysis declares aggregation readiness instead of treating posts as factual authority.",
            public_aggregation,
        ),
        check(
            "public_opinion_collection_audit_present",
            (public_opinion.get("collection_audit") or {}).get("status")
            in {
                "fixture_only_or_needs_collection_metadata",
                "ready_for_exploratory_public_ingest",
                "needs_privacy_review",
                "needs_visibility_review",
            },
            "Public-opinion analysis exposes collection metadata, privacy, visibility, and permission boundaries.",
            public_opinion.get("collection_audit"),
        ),
        check(
            "committee_marks_public_opinion_aggregate_only",
            int((committee_summary.get("committee_decision_counts") or {}).get("aggregate_only") or 0) > 0,
            "Evidence committee marks public-opinion candidates as aggregate-only.",
            committee_summary,
        ),
        check(
            "committee_rag_gates_final_context",
            bool(committee_gate_runs)
            and all(int(run.get("selected_context_policy_violation_count") or 0) == 0 for run in committee_gate_runs),
            "Committee-gated retrieval excludes non-generator-eligible evidence from selected final context.",
            {
                "run_count": len(committee_gate_runs),
                "selected_policy_violations": [
                    int(run.get("selected_context_policy_violation_count") or 0) for run in committee_gate_runs
                ],
                "final_context_counts": [int(run.get("final_context_count") or 0) for run in committee_gate_runs],
            },
        ),
        check(
            "committee_gate_comparison_reduces_policy_violations",
            committee_comparison.get("report_id") == "committee_gate_policy_comparison_v1"
            and (committee_comparison.get("contract_validation") or validate_report(committee_comparison, "committee_gate_comparison")).get("status") == "pass"
            and int((committee_comparison.get("summary") or {}).get("gated_selected_context_policy_violation_count") or 0)
            <= int((committee_comparison.get("summary") or {}).get("baseline_selected_context_policy_violation_count") or 0),
            "Paired committee-gate comparison shows the gated policy does not increase selected-context policy violations.",
            committee_comparison.get("summary"),
        ),
        check(
            "committee_gate_usefulness_audit_ready_for_labels",
            usefulness.get("report_id") == "committee_gate_usefulness_audit_v1"
            and (usefulness.get("contract_validation") or validate_report(usefulness, "committee_gate_usefulness")).get("status") == "pass"
            and int((usefulness.get("summary") or {}).get("human_label_task_count") or 0) > 0,
            "Committee-gate usefulness audit creates human-label-ready tasks for lost context and abstentions.",
            usefulness.get("summary"),
        ),
        check(
            "syntax_retrieval_ablation_tracks_form_signal",
            syntax_ablation.get("report_id") == "syntax_retrieval_ablation_v1"
            and (syntax_ablation.get("contract_validation") or validate_report(syntax_ablation, "syntax_retrieval_ablation")).get("status") == "pass"
            and (syntax_ablation.get("status") in {"pass", "review"})
            and bool(((syntax_ablation.get("summary") or {}).get("by_variant") or {}).get("syntax_candidate_and_rerank"))
            and int((((syntax_ablation.get("summary") or {}).get("by_variant") or {}).get("syntax_candidate_and_rerank") or {}).get("unsafe_final_count") or 0) == 0,
            "Syntax/form relevance ablation compares no-syntax, candidate-only syntax, and candidate-plus-rerank syntax without unsafe final exposure.",
            {
                "status": syntax_ablation.get("status"),
                "summary": syntax_ablation.get("summary"),
                "checks": syntax_ablation.get("checks"),
            },
        ),
        check(
            "unstructured_human_label_tasks_ready_for_annotation",
            label_tasks.get("report_id") == "unstructured_human_label_tasks_v1"
            and (label_tasks.get("contract_validation") or validate_report(label_tasks, "unstructured_label_tasks")).get("status") == "pass"
            and int(label_tasks.get("task_count") or 0) > 0
            and (label_tasks.get("coverage_summary") or {}).get("ready_for_annotation") is True,
            "Stratified unstructured human-label task pack covers required label families and is ready for annotation.",
            label_tasks.get("coverage_summary"),
        ),
        check(
            "unstructured_label_calibration_honest_status",
            label_calibration.get("report_id") == "unstructured_label_calibration_v1"
            and (label_calibration.get("contract_validation") or validate_report(label_calibration, "unstructured_label_calibration")).get("status") == "pass"
            and (
                (
                    label_calibration.get("status") == "needs_labels"
                    and int(label_calibration.get("completed_label_count") or 0) == 0
                )
                or (
                    label_calibration.get("status") == "calibrated"
                    and int(label_calibration.get("completed_label_count") or 0) > 0
                    and int((label_calibration.get("agreement") or {}).get("overall_comparison_count") or 0) > 0
                )
            ),
            "Unstructured label calibration artifact honestly distinguishes empty labels from calibrated agreement.",
            {
                "status": label_calibration.get("status"),
                "task_count": label_calibration.get("task_count"),
                "completed_task_count": label_calibration.get("completed_task_count"),
                "completed_label_count": label_calibration.get("completed_label_count"),
                "overall_comparison_count": (label_calibration.get("agreement") or {}).get("overall_comparison_count"),
            },
        ),
        check(
            "public_opinion_synthesis_excludes_generator_context",
            synthesis.get("contract_validation", {}).get("status") == "pass"
            and len(synthesis_context.get("items") or []) == 0
            and int((synthesis_context.get("assembly_summary") or {}).get("aggregate_only_count") or 0) > 0,
            "Public-opinion synthesis smoke test excludes aggregate-only posts from ordinary generator context.",
            synthesis_context.get("assembly_summary"),
        ),
        check(
            "evaluation_anchor_claim_limits_present",
            anchors.get("report_id") == "evaluation_anchor_registry_v1"
            and len(anchors.get("claim_limits") or []) >= 4,
            "Evaluation anchor registry includes overclaiming limits.",
            anchors.get("claim_limits"),
        ),
    ]


def artifact_summary(artifacts: dict[str, Any]) -> dict[str, Any]:
    readiness = artifacts.get("unstructured_readiness") or {}
    phase_gate = artifacts.get("phase_gate") or {}
    security = artifacts.get("adversarial_security") or {}
    public_opinion = artifacts.get("public_opinion") or {}
    committee = artifacts.get("evidence_committee") or {}
    committee_comparison = artifacts.get("committee_gate_comparison") or {}
    usefulness = artifacts.get("committee_gate_usefulness") or {}
    syntax_ablation = artifacts.get("syntax_retrieval_ablation") or {}
    matrix = artifacts.get("coverage_matrix") or {}
    synthesis = artifacts.get("public_opinion_synthesis") or {}
    committee_gate_runs = [run for run in committee.get("runs") or [] if run.get("policy") == "committee_rag"]
    return {
        "unstructured_readiness_status": readiness.get("status"),
        "phase_gate_status": phase_gate.get("status"),
        "adversarial_security_status": security.get("status"),
        "attack_context_exposure_count": (security.get("metrics") or {}).get("attack_context_exposure_count"),
        "chunking_large_window_boundary_unsafe_isolated_count": (
            ((artifacts.get("chunking_large_window") or {}).get("summary") or {}).get("boundary_aware") or {}
        ).get("unsafe_isolated_count"),
        "chunking_large_window_fixed_unsafe_isolated_count": (
            ((artifacts.get("chunking_large_window") or {}).get("summary") or {}).get("fixed_window") or {}
        ).get("unsafe_isolated_count"),
        "chunking_variant_balanced_utility_winner": (
            (artifacts.get("chunking_variants") or {}).get("best_variant_by_objective") or {}
        ).get("balanced_utility"),
        "chunking_variant_fixed_forbidden_merge_count": (
            (((artifacts.get("chunking_variants") or {}).get("summary") or {}).get("fixed_window") or {})
        ).get("forbidden_merge_count"),
        "chunking_variant_boundary_forbidden_merge_count": (
            (((artifacts.get("chunking_variants") or {}).get("summary") or {}).get("boundary_aware") or {})
        ).get("forbidden_merge_count"),
        "chunking_variant_embedding_forbidden_merge_count": (
            (((artifacts.get("chunking_variants") or {}).get("summary") or {}).get("embedding_semantic") or {})
        ).get("forbidden_merge_count"),
        "public_opinion_aggregation_status": (public_opinion.get("aggregation_readiness") or {}).get("status"),
        "public_opinion_collection_status": (public_opinion.get("collection_audit") or {}).get("status"),
        "public_opinion_generator_context_count": len((synthesis.get("context_packet") or {}).get("items") or []),
        "public_opinion_excluded_count": len((synthesis.get("context_packet") or {}).get("excluded") or []),
        "committee_rag_run_count": len(committee_gate_runs),
        "committee_rag_policy_violation_count": sum(
            int(run.get("selected_context_policy_violation_count") or 0) for run in committee_gate_runs
        ),
        "committee_gate_comparison_policy_violation_reduction": (
            committee_comparison.get("summary") or {}
        ).get("policy_violation_reduction"),
        "committee_gate_comparison_retention_rate": (
            committee_comparison.get("summary") or {}
        ).get("final_context_retention_rate"),
        "committee_gate_usefulness_human_label_task_count": (
            usefulness.get("summary") or {}
        ).get("human_label_task_count"),
        "committee_gate_usefulness_potentially_useful_loss_count": (
            usefulness.get("summary") or {}
        ).get("potentially_useful_loss_count"),
        "committee_gate_usefulness_review_needed_loss_count": (
            usefulness.get("summary") or {}
        ).get("review_needed_loss_count"),
        "syntax_ablation_status": syntax_ablation.get("status"),
        "syntax_ablation_mrr_delta": (
            (syntax_ablation.get("summary") or {}).get("syntax_vs_no_syntax") or {}
        ).get("mean_reciprocal_final_rank_delta"),
        "syntax_ablation_category_match_delta": (
            (syntax_ablation.get("summary") or {}).get("syntax_vs_no_syntax") or {}
        ).get("syntax_category_match_rate_delta"),
        "unstructured_label_task_count": (artifacts.get("unstructured_label_tasks") or {}).get("task_count"),
        "unstructured_label_ready_for_annotation": (
            ((artifacts.get("unstructured_label_tasks") or {}).get("coverage_summary") or {})
        ).get("ready_for_annotation"),
        "unstructured_label_missing_families": (
            ((artifacts.get("unstructured_label_tasks") or {}).get("coverage_summary") or {})
        ).get("missing_label_families", []),
        "unstructured_label_calibration_status": (artifacts.get("unstructured_label_calibration") or {}).get("status"),
        "unstructured_label_completed_label_count": (artifacts.get("unstructured_label_calibration") or {}).get("completed_label_count"),
        "coverage_matrix_status": matrix.get("status"),
        "coverage_matrix_missing_document_types": (matrix.get("gap_summary") or {}).get("missing_document_types", []),
        "coverage_matrix_limited_label_families": ((matrix.get("gap_summary") or {}).get("human_label_status") or {}).get("planned_or_limited_label_families", []),
    }


def claim_boundaries() -> list[str]:
    return [
        "Allowed: CANON can run a deterministic heterogeneous-unstructured experiment bundle across document types, domains, public opinion, safety, chunking, and typed evidence decisions.",
        "Allowed: current fixtures demonstrate that prompt-injection text can be retrieved, rejected, quarantined, or excluded before ordinary generation context.",
        "Allowed: the chunking benchmark includes a large-window adversarial probe showing why safety-sensitive source forms need boundary-aware atomic chunks, not only token windows.",
        "Allowed: the chunking variant benchmark compares deterministic technical baselines and records when span containment alone hides harmful merges.",
        "Allowed: the chunking variant benchmark now includes local hashed embedding-semantic grouping as a stronger no-network technical baseline.",
        "Allowed: the experiment now produces a stratified human-label task scaffold for document type, evidence role, source trust, source independence, safety disposition, and answer usefulness.",
        "Allowed: public social/forum records can be analyzed as aggregate expression evidence when visibility, platform, sampling, safety, and representativeness boundaries are explicit.",
        "Allowed: public-opinion acquisition audits can distinguish fixture-only corpora from exploratory public-ingest candidates.",
        "Allowed: syntax/form relevance can be evaluated as a separate retrieval signal rather than being folded into semantic similarity.",
        "Not allowed: claim production-grade prompt-injection robustness from deterministic fixtures alone.",
        "Not allowed: claim public opinion, market sentiment, legal truth, or disciplinary source quality without larger external corpora and human/domain-expert anchors.",
        "Not allowed: treat generated label tasks as completed human validation.",
        "Not allowed: claim calibrated deterministic quality decisions while label calibration status is `needs_labels`.",
        "Not allowed: collapse source trust, relevance, safety, corroboration, and evidence role into one universal importance score.",
    ]


def recommended_next_steps(checks: list[dict[str, Any]]) -> list[str]:
    failed = [item["id"] for item in checks if not item["passed"]]
    if failed:
        return [
            f"Regenerate or fix failed portfolio artifacts: {', '.join(failed)}.",
            "Rerun the unstructured portfolio after the upstream report contracts pass.",
        ]
    return [
        "Replace fixtures with larger domain-sliced corpora for economics, psychology, anthropology, history, cultural studies, legal, market, and public-opinion tasks.",
        "Add human labels for document type, evidence role, source trust, source independence, safety disposition, and answer usefulness.",
        "Add public external retrieval/qrels anchors and domain-specific expert review sets.",
        "Compare deterministic gates against learned arbiters, model-based safety screeners, semantic chunkers, late chunking, and hierarchical parent-child retrieval.",
        "Compare local hashed embedding-semantic chunking against external neural embedding chunkers, late chunking, and hierarchical parent-child retrieval on the same artifact contract.",
        "Add harder syntax/form retrieval qrels with near-duplicate topical distractors, so syntax ablations measure incremental value beyond easy lexical/semantic matches.",
        "For public-opinion corpora, add platform terms/robots/license/consent-basis metadata, sampling frames, and privacy review before broad aggregation.",
        "Compare `committee_rag` against audit-only `rag` on recall loss, answer usefulness, safety exposure, and aggregate-only abstention.",
    ]


def limitations() -> list[str]:
    return [
        "This portfolio proves experiment coverage and artifact coherence, not final scientific validity.",
        "Most current corpora are fixtures or small named corpora; broad-domain claims require larger external datasets.",
        "Safety, source trust, public-opinion stance/theme labels, and chunking policy are deterministic or local-hashed baselines that need adversarial and human validation.",
        "The syntax/form ablation is currently fixture-saturated; neutral deltas mean the easy anchors are already solved by lexical/semantic retrieval, not that syntax is fully validated.",
        "The portfolio intentionally allows abstention when evidence is aggregate-only or unsafe for ordinary RAG generation.",
        "Public-opinion acquisition audits are metadata checks, not legal advice or proof of platform compliance.",
    ]


def check(identifier: str, passed: bool, description: str, evidence: Any = None) -> dict[str, Any]:
    row = {"id": identifier, "passed": bool(passed), "description": description}
    if evidence is not None:
        row["evidence"] = evidence
    return row


def slug(query: str) -> str:
    return "".join(character if character.isalnum() else "-" for character in query.lower())[:60].strip("-")


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def render_markdown(report: dict[str, Any]) -> str:
    checks = "\n".join(
        f"- [{'x' if item['passed'] else ' '}] `{item['id']}`: {item['description']}"
        for item in report["checks"]
    )
    boundaries = "\n".join(f"- {item}" for item in report["claim_boundaries"])
    steps = "\n".join(f"- {item}" for item in report["recommended_next_steps"])
    limitations_text = "\n".join(f"- {item}" for item in report["limitations"])
    return f"""# CANON Unstructured Experiment Portfolio

- Status: {report["status"]}
- Document-type mode: {report["modes"]["document_type_mode"]}
- Domain mode: {report["modes"]["domain_mode"]}
- Public-opinion mode: {report["modes"]["public_opinion_mode"]}

## Checks

{checks}

## Artifact Summary

```json
{json.dumps(report["artifact_summary"], indent=2)}
```

## Claim Boundaries

{boundaries}

## Recommended Next Steps

{steps}

## Limitations

{limitations_text}
"""


def write_markdown(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the CANON heterogeneous unstructured experiment portfolio.")
    parser.add_argument("--document-type-mode", default="unstructured_demo_corpus")
    parser.add_argument("--domain-mode", default="mixed_domain_demo_corpus")
    parser.add_argument("--public-opinion-mode", default="social_public_opinion_demo_corpus")
    parser.add_argument("--public-opinion-query", default=DEFAULT_PUBLIC_OPINION_QUERY)
    parser.add_argument("--no-write-report", action="store_true")
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = build_unstructured_portfolio_report(
        document_type_mode=args.document_type_mode,
        domain_mode=args.domain_mode,
        public_opinion_mode=args.public_opinion_mode,
        public_opinion_query=args.public_opinion_query,
        write_report=not args.no_write_report,
    )
    print(json.dumps(report, indent=2))
    if report["status"] != "pass" and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
