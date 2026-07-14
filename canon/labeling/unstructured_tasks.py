from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from canon.config import load_settings
from canon.eval.contracts import CONTRACT_VERSION, validate_report
from canon.quality.domain_profiles import domain_fit_vector
from canon.quality.document_types import document_type_profile
from canon.quality.signals import source_trust_vector
from canon.retrieval.corpus import work_model_from_dict
from canon.safety.scanner import scan_text


DEFAULT_MODES = [
    "unstructured_demo_corpus",
    "mixed_domain_demo_corpus",
    "social_public_opinion_demo_corpus",
]

UNSTRUCTURED_LABEL_SCHEMA = {
    "document_type_correctness": ["wrong", "partly_correct", "correct", "uncertain"],
    "primary_evidence_role": [
        "claim",
        "method",
        "result",
        "limitation",
        "definition",
        "context",
        "contradiction",
        "provenance",
        "public_expression",
        "instruction_like_text",
        "other",
    ],
    "source_trust_tier": ["blocked", "low", "limited", "moderate", "strong", "uncertain"],
    "source_independence": ["dependent_echo", "unclear", "partly_independent", "independent"],
    "safety_disposition": ["allow", "sanitize", "quarantine", "block", "uncertain"],
    "generator_use": ["allow_context", "aggregate_only", "inspect_only", "exclude", "uncertain"],
    "answer_usefulness": ["harmful", "not_useful", "useful_with_caution", "useful", "essential", "uncertain"],
    "exclusion_correctness": ["wrongly_excluded", "uncertain", "correctly_excluded"],
    "chunking_harmful_merge": ["no_harmful_merge", "possible_harmful_merge", "harmful_merge", "uncertain"],
    "qualifier_preservation": ["missing_qualifier", "partly_preserved", "preserved", "uncertain"],
    "fragmentation": ["under_fragmented", "appropriate", "over_fragmented", "uncertain"],
    "preferred_chunking_variant": [
        "fixed_window",
        "boundary_aware",
        "atomic_sentence",
        "semantic_cohesion",
        "embedding_semantic",
        "paragraph_or_sentence",
        "uncertain",
    ],
}

LABEL_FAMILY_TO_FIELDS = {
    "document_type": ["document_type_correctness"],
    "evidence_role": ["primary_evidence_role"],
    "source_trust": ["source_trust_tier", "generator_use"],
    "source_independence": ["source_independence"],
    "safety_disposition": ["safety_disposition", "generator_use"],
    "answer_usefulness": ["answer_usefulness", "exclusion_correctness"],
    "chunking_quality": [
        "chunking_harmful_merge",
        "qualifier_preservation",
        "fragmentation",
        "preferred_chunking_variant",
    ],
}


def build_unstructured_label_tasks(
    modes: list[str] | None = None,
    per_stratum: int = 2,
    max_tasks: int = 80,
    write_report: bool = True,
) -> dict[str, Any]:
    settings = load_settings()
    resolved_modes = modes or DEFAULT_MODES
    works, chunks = load_corpus_rows(settings.data_dir, resolved_modes)
    chunk_tasks = stratified_chunk_tasks(works, chunks, per_stratum=per_stratum, max_tasks=max_tasks)
    usefulness_tasks = committee_usefulness_tasks(settings.reports_dir / "committee_gate_usefulness_audit_v1.json")
    chunking_tasks = chunking_variant_review_tasks(settings.reports_dir / "chunking_variant_eval_ct1000_ov100.json")
    tasks = dedupe_tasks(chunking_tasks + chunk_tasks + usefulness_tasks, max_tasks=max_tasks)
    report = {
        "contract_version": CONTRACT_VERSION,
        "report_id": "unstructured_human_label_tasks_v1",
        "modes": resolved_modes,
        "schema": UNSTRUCTURED_LABEL_SCHEMA,
        "label_family_to_fields": LABEL_FAMILY_TO_FIELDS,
        "task_count": len(tasks),
        "tasks": tasks,
        "coverage_summary": coverage_summary(tasks),
        "claim_boundaries": claim_boundaries(),
        "limitations": limitations(),
    }
    report["contract_validation"] = validate_report(report, "unstructured_label_tasks")
    if write_report:
        path = settings.reports_dir / "unstructured_human_label_tasks_v1.json"
        write_json(path, report)
        write_markdown(path.with_suffix(".md"), render_markdown(report))
    return report


def load_corpus_rows(data_dir: Path, modes: list[str]) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    works: dict[str, dict[str, Any]] = {}
    chunks: list[dict[str, Any]] = []
    for mode in modes:
        works_path = data_dir / "processed" / f"works_{mode}.json"
        chunks_path = data_dir / "processed" / f"chunks_{mode}.json"
        mode_works = {row.get("id"): {**row, "label_source_mode": mode} for row in load_json(works_path, [])}
        works.update({str(key): value for key, value in mode_works.items() if key})
        for chunk in load_json(chunks_path, []):
            if chunk.get("work_id") in mode_works:
                chunks.append({**chunk, "label_source_mode": mode})
    return works, chunks


def stratified_chunk_tasks(
    works: dict[str, dict[str, Any]],
    chunks: list[dict[str, Any]],
    per_stratum: int,
    max_tasks: int,
) -> list[dict[str, Any]]:
    by_stratum: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for chunk in chunks:
        work = works.get(str(chunk.get("work_id") or ""))
        if not work:
            continue
        row = chunk_task(work, chunk)
        by_stratum[(row["domain"], row["document_type"])].append(row)
    selected: list[dict[str, Any]] = []
    for stratum in sorted(by_stratum):
        rows = sorted(by_stratum[stratum], key=lambda row: (-float(row["priority_score"]), row["id"]))
        selected.extend(rows[: max(1, per_stratum)])
    return sorted(selected, key=lambda row: (-float(row["priority_score"]), row["id"]))[:max_tasks]


def chunk_task(work: dict[str, Any], chunk: dict[str, Any]) -> dict[str, Any]:
    work_model = work_model_from_dict(work)
    domain = str(work.get("domain") or (work.get("raw") or {}).get("domain") or "general_academic")
    document_profile = document_type_profile(work_model).to_dict()
    source_trust = source_trust_vector(work_model, domain=domain)
    domain_fit = domain_fit_vector(work_model, domain, document_profile, source_trust)
    provenance = str(work.get("provenance") or (work.get("raw") or {}).get("provenance") or "unknown")
    safety = scan_text(str(chunk.get("text") or ""), provenance=provenance).as_dict()
    importance = chunk.get("importance") or {}
    label_families = chunk_label_families(document_profile, source_trust, domain_fit, safety, importance)
    return {
        "id": f"chunk::{chunk.get('label_source_mode')}::{chunk.get('id')}",
        "task_family": "chunk_evidence_review",
        "label_families": label_families,
        "mode": chunk.get("label_source_mode"),
        "domain": domain,
        "document_type": document_profile["document_type"],
        "work_id": work.get("id"),
        "chunk_id": chunk.get("id"),
        "title": work.get("title"),
        "source_name": work.get("source_name"),
        "year": work.get("year"),
        "provenance": provenance,
        "text": chunk.get("text"),
        "deterministic_signals": {
            "document_type_profile": document_profile,
            "source_trust": source_trust,
            "domain_fit": domain_fit,
            "safety": safety,
            "chunk_importance": selected_importance(importance),
        },
        "priority_score": priority_score(document_profile, source_trust, domain_fit, safety, importance),
        "priority_reasons": priority_reasons(document_profile, source_trust, domain_fit, safety, importance),
        "human_labels": empty_labels(label_families),
        "review_instructions": review_instructions(label_families),
    }


def chunk_label_families(
    document_profile: dict[str, Any],
    source_trust: dict[str, Any],
    domain_fit: dict[str, Any],
    safety: dict[str, Any],
    importance: dict[str, Any],
) -> list[str]:
    families = {"document_type", "evidence_role", "source_trust", "source_independence", "safety_disposition"}
    if safety.get("decision") != "allow":
        families.add("answer_usefulness")
    if document_profile.get("document_type") in {"social_media_post", "forum_comment", "web_page"}:
        families.add("answer_usefulness")
    if domain_fit.get("domain_fit_tier") in {"weak_fit", "low_fit", "quarantine_review", "blocked"}:
        families.add("answer_usefulness")
    if float(importance.get("chunk_safety_contamination_risk", 0.0) or 0.0) >= 0.6:
        families.add("safety_disposition")
    return sorted(families)


def committee_usefulness_tasks(path: Path) -> list[dict[str, Any]]:
    report = load_json(path, {})
    tasks: list[dict[str, Any]] = []
    for mode in report.get("modes") or []:
        mode_name = mode.get("mode")
        for query in mode.get("queries") or []:
            query_id = query.get("id")
            for candidate in query.get("lost_candidates") or []:
                tasks.append(usefulness_loss_task(mode_name, query, candidate))
            if query.get("gated_abstained") and query.get("abstention_review"):
                tasks.append(abstention_task(mode_name, query, query["abstention_review"]))
    return tasks


def chunking_variant_review_tasks(path: Path) -> list[dict[str, Any]]:
    report = load_json(path, {})
    if not report:
        return []
    rows = report.get("cases") or []
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_case[str(row.get("case_id") or "")].append(row)
    summary = report.get("summary") or {}
    best = report.get("best_variant_by_objective") or {}
    tasks = []
    for case_id, case_rows in sorted(by_case.items()):
        if not case_id:
            continue
        fixed = next((row for row in case_rows if row.get("strategy") == "fixed_window"), {})
        boundary = next((row for row in case_rows if row.get("strategy") == "boundary_aware"), {})
        semantic = next((row for row in case_rows if row.get("strategy") == "semantic_cohesion"), {})
        embedding = next((row for row in case_rows if row.get("strategy") == "embedding_semantic"), {})
        atomic = next((row for row in case_rows if row.get("strategy") == "atomic_sentence"), {})
        fixed_merges = int(((fixed.get("forbidden_merge") or {}).get("forbidden_merge_count")) or 0)
        interesting = fixed_merges > 0 or case_has_fragmentation_tradeoff(case_rows)
        if not interesting:
            continue
        document_type = str((fixed or boundary or {}).get("document_type") or "unknown")
        domain = str((fixed or boundary or {}).get("domain") or "unknown")
        label_families = ["chunking_quality", "answer_usefulness", "safety_disposition"]
        tasks.append(
            {
                "id": f"chunking_variant::{case_id}",
                "task_family": "chunking_variant_review",
                "label_families": label_families,
                "mode": "chunking_variant_eval_ct1000_ov100",
                "domain": domain,
                "document_type": document_type,
                "work_id": f"chunking_variant:{case_id}",
                "chunk_id": None,
                "title": f"Chunking variant review: {case_id}",
                "source_name": "chunking_variant_fixture",
                "year": 2026,
                "provenance": "fixture",
                "text": chunking_case_text(case_rows),
                "deterministic_signals": {
                    "chunking_variant_comparison": {
                        "case_id": case_id,
                        "best_variant_by_objective": best,
                        "fixed_window": compact_variant_case(fixed, summary.get("fixed_window") or {}),
                        "boundary_aware": compact_variant_case(boundary, summary.get("boundary_aware") or {}),
                        "semantic_cohesion": compact_variant_case(semantic, summary.get("semantic_cohesion") or {}),
                        "embedding_semantic": compact_variant_case(embedding, summary.get("embedding_semantic") or {}),
                        "atomic_sentence": compact_variant_case(atomic, summary.get("atomic_sentence") or {}),
                    },
                    "safety": chunking_variant_safety_signal(fixed, boundary),
                },
                "priority_score": chunking_variant_priority(fixed, boundary, semantic, embedding, atomic),
                "priority_reasons": chunking_variant_priority_reasons(fixed, boundary, semantic, embedding, atomic),
                "human_labels": empty_labels(label_families),
                "review_instructions": review_instructions(label_families),
            }
        )
    return sorted(tasks, key=lambda row: (-float(row["priority_score"]), row["id"]))


def case_has_fragmentation_tradeoff(case_rows: list[dict[str, Any]]) -> bool:
    fixed = next((row for row in case_rows if row.get("strategy") == "fixed_window"), {})
    atomic = next((row for row in case_rows if row.get("strategy") == "atomic_sentence"), {})
    fixed_count = int(((fixed.get("chunking_profile") or {}).get("chunk_count")) or 0)
    atomic_count = int(((atomic.get("chunking_profile") or {}).get("chunk_count")) or 0)
    return atomic_count >= max(3, fixed_count + 2)


def chunking_case_text(case_rows: list[dict[str, Any]]) -> str:
    for row in case_rows:
        chunks = row.get("chunks") or []
        if chunks:
            return " ".join(str(chunk.get("text") or "") for chunk in chunks)
    return ""


def compact_variant_case(row: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    chunks = row.get("chunks") or []
    return {
        "span_containment_rate": row.get("span_containment_rate"),
        "missing_spans": row.get("missing_spans") or [],
        "forbidden_merge": row.get("forbidden_merge") or {},
        "unsafe_isolation": row.get("unsafe_isolation") or {},
        "chunk_count": (row.get("chunking_profile") or {}).get("chunk_count"),
        "objective_scores": summary.get("objective_scores") or {},
        "chunks": [
            {
                "token_start": chunk.get("token_start"),
                "token_end": chunk.get("token_end"),
                "text": chunk.get("text"),
            }
            for chunk in chunks[:6]
        ],
    }


def chunking_variant_safety_signal(fixed: dict[str, Any], boundary: dict[str, Any]) -> dict[str, Any]:
    fixed_merge_count = int(((fixed.get("forbidden_merge") or {}).get("forbidden_merge_count")) or 0)
    fixed_unsafe = bool((fixed.get("unsafe_isolation") or {}).get("unsafe_span_present"))
    boundary_unsafe_isolated = bool((boundary.get("unsafe_isolation") or {}).get("unsafe_isolated"))
    if fixed_merge_count > 0 and fixed_unsafe and boundary_unsafe_isolated:
        return {"decision": "block", "reason": "fixed_window_harmful_merge"}
    if fixed_merge_count > 0:
        return {"decision": "quarantine", "reason": "fixed_window_forbidden_merge"}
    return {"decision": "allow", "reason": "variant_review"}


def chunking_variant_priority(
    fixed: dict[str, Any],
    boundary: dict[str, Any],
    semantic: dict[str, Any],
    embedding: dict[str, Any],
    atomic: dict[str, Any],
) -> float:
    score = 0.55
    if int(((fixed.get("forbidden_merge") or {}).get("forbidden_merge_count")) or 0) > 0:
        score += 0.25
    if (fixed.get("unsafe_isolation") or {}).get("unsafe_isolated") is False and (boundary.get("unsafe_isolation") or {}).get("unsafe_isolated"):
        score += 0.1
    fixed_count = int(((fixed.get("chunking_profile") or {}).get("chunk_count")) or 0)
    atomic_count = int(((atomic.get("chunking_profile") or {}).get("chunk_count")) or 0)
    semantic_count = int(((semantic.get("chunking_profile") or {}).get("chunk_count")) or 0)
    embedding_count = int(((embedding.get("chunking_profile") or {}).get("chunk_count")) or 0)
    if max(atomic_count, semantic_count, embedding_count) >= max(3, fixed_count + 2):
        score += 0.1
    return round(min(1.0, score), 6)


def chunking_variant_priority_reasons(
    fixed: dict[str, Any],
    boundary: dict[str, Any],
    semantic: dict[str, Any],
    embedding: dict[str, Any],
    atomic: dict[str, Any],
) -> list[str]:
    reasons = []
    if int(((fixed.get("forbidden_merge") or {}).get("forbidden_merge_count")) or 0) > 0:
        reasons.append("fixed_window_forbidden_merge")
    if (fixed.get("unsafe_isolation") or {}).get("unsafe_isolated") is False and (boundary.get("unsafe_isolation") or {}).get("unsafe_isolated"):
        reasons.append("boundary_aware_isolates_unsafe_text")
    fixed_count = int(((fixed.get("chunking_profile") or {}).get("chunk_count")) or 0)
    atomic_count = int(((atomic.get("chunking_profile") or {}).get("chunk_count")) or 0)
    semantic_count = int(((semantic.get("chunking_profile") or {}).get("chunk_count")) or 0)
    embedding_count = int(((embedding.get("chunking_profile") or {}).get("chunk_count")) or 0)
    if max(atomic_count, semantic_count, embedding_count) >= max(3, fixed_count + 2):
        reasons.append("fragmentation_tradeoff")
    return reasons or ["chunking_variant_tradeoff_review"]


def usefulness_loss_task(mode: str, query: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    label_families = ["answer_usefulness", "safety_disposition", "source_trust"]
    return {
        "id": f"usefulness_loss::{mode}::{query.get('id')}::{candidate.get('chunk_id')}",
        "task_family": "committee_gate_loss_review",
        "label_families": label_families,
        "mode": mode,
        "domain": candidate.get("domain") or "unknown",
        "document_type": candidate.get("document_type") or "unknown",
        "work_id": candidate.get("work_id"),
        "chunk_id": candidate.get("chunk_id"),
        "title": candidate.get("title"),
        "source_name": candidate.get("source_name"),
        "year": candidate.get("year"),
        "provenance": candidate.get("provenance") or "unknown",
        "query": query.get("query"),
        "query_intent": query.get("intent"),
        "text": candidate.get("preview"),
        "deterministic_signals": {
            "committee_decision": candidate.get("committee_decision"),
            "allowed_uses": candidate.get("allowed_uses"),
            "conflict_tags": candidate.get("conflict_tags"),
            "proxy_usefulness_risk": candidate.get("proxy_usefulness_risk"),
        },
        "priority_score": usefulness_priority(candidate),
        "priority_reasons": ["committee_gate_lost_baseline_selected_context", str(candidate.get("proxy_usefulness_risk") or "unknown_risk")],
        "human_labels": empty_labels(label_families),
        "review_instructions": review_instructions(label_families),
    }


def abstention_task(mode: str, query: dict[str, Any], abstention: dict[str, Any]) -> dict[str, Any]:
    label_families = ["answer_usefulness", "source_trust", "safety_disposition"]
    return {
        "id": f"abstention::{mode}::{query.get('id')}",
        "task_family": "committee_gate_abstention_review",
        "label_families": label_families,
        "mode": mode,
        "domain": "unknown",
        "document_type": "none_selected",
        "work_id": None,
        "chunk_id": None,
        "title": "Committee-gated abstention",
        "source_name": None,
        "year": None,
        "provenance": "retrieval_decision",
        "query": query.get("query"),
        "query_intent": query.get("intent"),
        "text": None,
        "deterministic_signals": {
            "baseline_final_context_count": abstention.get("baseline_final_context_count"),
            "gated_final_context_count": abstention.get("gated_final_context_count"),
            "baseline_had_context": abstention.get("baseline_final_context_count", 0) > 0,
        },
        "priority_score": 0.95,
        "priority_reasons": ["committee_gate_abstention_needs_usefulness_review"],
        "human_labels": empty_labels(label_families),
        "review_instructions": review_instructions(label_families),
    }


def selected_importance(importance: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "claim_density",
        "method_signal",
        "limitation_signal",
        "evidence_specificity",
        "uncertainty_signal",
        "chunk_strategy",
        "chunk_resolution",
        "generation_context_role",
        "context_expansion_policy",
        "chunk_safety_contamination_risk",
        "chunk_quality_tier",
    ]
    return {key: importance.get(key) for key in keys if key in importance}


def priority_score(
    document_profile: dict[str, Any],
    source_trust: dict[str, Any],
    domain_fit: dict[str, Any],
    safety: dict[str, Any],
    importance: dict[str, Any],
) -> float:
    score = 0.2
    if safety.get("decision") != "allow":
        score += 0.35
    if document_profile.get("document_type") in {"web_page", "social_media_post", "forum_comment", "unknown_unstructured_text"}:
        score += 0.15
    if domain_fit.get("domain_fit_tier") in {"weak_fit", "low_fit", "quarantine_review", "blocked"}:
        score += 0.15
    if float(source_trust.get("source_trust", 0.0) or 0.0) < 0.4:
        score += 0.1
    if float(importance.get("limitation_signal", 0.0) or 0.0) > 0.0 or float(importance.get("method_signal", 0.0) or 0.0) > 0.0:
        score += 0.05
    return round(min(1.0, score), 6)


def usefulness_priority(candidate: dict[str, Any]) -> float:
    risk = str(candidate.get("proxy_usefulness_risk") or "")
    if risk == "potentially_useful_loss":
        return 1.0
    if risk == "review_needed_loss":
        return 0.9
    return 0.75


def priority_reasons(
    document_profile: dict[str, Any],
    source_trust: dict[str, Any],
    domain_fit: dict[str, Any],
    safety: dict[str, Any],
    importance: dict[str, Any],
) -> list[str]:
    reasons = []
    if safety.get("decision") != "allow":
        reasons.append(f"safety_{safety.get('decision')}")
    if document_profile.get("document_type") in {"web_page", "social_media_post", "forum_comment", "unknown_unstructured_text"}:
        reasons.append(f"high_risk_document_type:{document_profile.get('document_type')}")
    if domain_fit.get("domain_fit_tier") in {"weak_fit", "low_fit", "quarantine_review", "blocked"}:
        reasons.append(f"domain_fit:{domain_fit.get('domain_fit_tier')}")
    if float(source_trust.get("source_trust", 0.0) or 0.0) < 0.4:
        reasons.append("low_or_limited_source_trust")
    if float(importance.get("limitation_signal", 0.0) or 0.0) > 0.0:
        reasons.append("limitation_signal")
    if float(importance.get("method_signal", 0.0) or 0.0) > 0.0:
        reasons.append("method_signal")
    return reasons or ["stratified_domain_document_type_sample"]


def empty_labels(label_families: list[str]) -> dict[str, None]:
    fields = []
    for family in label_families:
        fields.extend(LABEL_FAMILY_TO_FIELDS.get(family, []))
    return {field: None for field in sorted(set(fields))}


def review_instructions(label_families: list[str]) -> list[str]:
    instructions = []
    if "document_type" in label_families:
        instructions.append("Judge whether the inferred document type matches the text and metadata.")
    if "evidence_role" in label_families:
        instructions.append("Identify the primary role of the chunk for the query-independent evidence store.")
    if "source_trust" in label_families:
        instructions.append("Rate source trust for evidentiary use, ignoring open access, PDF availability, citation count, or prestige as direct reliability shortcuts.")
    if "source_independence" in label_families:
        instructions.append("Flag whether the source appears independent or dependent on another source chain.")
    if "safety_disposition" in label_families:
        instructions.append("Decide whether the text should be allowed, sanitized, quarantined, or blocked before generation.")
    if "answer_usefulness" in label_families:
        instructions.append("Judge whether excluding or using this evidence would help answer quality for the given query or task context.")
    if "chunking_quality" in label_families:
        instructions.append("Compare the candidate chunking variants for harmful merges, missing qualifiers, excessive fragmentation, and the best strategy for safe evidence use.")
    return instructions


def dedupe_tasks(tasks: list[dict[str, Any]], max_tasks: int) -> list[dict[str, Any]]:
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    for task in sorted(tasks, key=lambda row: (-float(row.get("priority_score") or 0.0), str(row.get("id") or ""))):
        task_id = str(task.get("id") or "")
        if task_id in seen:
            continue
        seen.add(task_id)
        rows.append(task)
    return rows[:max_tasks]


def coverage_summary(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    label_families = Counter()
    domains = Counter()
    document_types = Counter()
    task_families = Counter()
    safety = Counter()
    for task in tasks:
        task_families[str(task.get("task_family") or "unknown")] += 1
        domains[str(task.get("domain") or "unknown")] += 1
        document_types[str(task.get("document_type") or "unknown")] += 1
        for family in task.get("label_families") or []:
            label_families[str(family)] += 1
        decision = (((task.get("deterministic_signals") or {}).get("safety") or {}).get("decision"))
        if decision:
            safety[str(decision)] += 1
    required = set(LABEL_FAMILY_TO_FIELDS)
    return {
        "task_family_counts": dict(sorted(task_families.items())),
        "label_family_counts": dict(sorted(label_families.items())),
        "domain_counts": dict(sorted(domains.items())),
        "document_type_counts": dict(sorted(document_types.items())),
        "safety_decision_counts": dict(sorted(safety.items())),
        "covered_label_families": sorted(label_families),
        "missing_label_families": sorted(required - set(label_families)),
        "ready_for_annotation": bool(tasks) and not (required - set(label_families)),
    }


def claim_boundaries() -> list[str]:
    return [
        "Allowed: use these tasks to collect human labels for deterministic document-type, evidence-role, source-trust, safety, independence, and usefulness decisions.",
        "Allowed: use completed labels to calibrate or challenge deterministic gates and ranking policies.",
        "Not allowed: treat task creation as completed human validation.",
        "Not allowed: infer public-opinion representativeness, legal truth, market truth, or disciplinary source quality before labels and external anchors are completed.",
    ]


def limitations() -> list[str]:
    return [
        "The task pack is a sampling and review scaffold; all labels are empty until a human or approved expert workflow fills them.",
        "Stratification uses current processed fixture metadata and deterministic profiles, so it can inherit their classification errors.",
        "Answer-usefulness tasks are currently anchored mainly by committee-gate loss and abstention cases.",
    ]


def render_markdown(report: dict[str, Any]) -> str:
    coverage = json.dumps(report["coverage_summary"], indent=2)
    boundaries = "\n".join(f"- {item}" for item in report["claim_boundaries"])
    limitations_text = "\n".join(f"- {item}" for item in report["limitations"])
    return f"""# Unstructured Human Label Tasks

- Tasks: {report["task_count"]}
- Modes: {", ".join(report["modes"])}
- Ready for annotation: {report["coverage_summary"]["ready_for_annotation"]}

## Coverage

```json
{coverage}
```

## Claim Boundaries

{boundaries}

## Limitations

{limitations_text}
"""


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_markdown(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build stratified human-label tasks for heterogeneous unstructured evidence.")
    parser.add_argument("--modes", default=",".join(DEFAULT_MODES))
    parser.add_argument("--per-stratum", type=int, default=2)
    parser.add_argument("--max-tasks", type=int, default=80)
    parser.add_argument("--no-write-report", action="store_true")
    args = parser.parse_args()
    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    report = build_unstructured_label_tasks(
        modes=modes,
        per_stratum=args.per_stratum,
        max_tasks=args.max_tasks,
        write_report=not args.no_write_report,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
