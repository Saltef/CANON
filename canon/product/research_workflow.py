from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from canon.config import load_settings
from canon.product.frame_enrichment import enrich_frame
from canon.product import report_io, service


QUESTION_TYPES = {
    "comparative": ("compare", "difference", "versus", "vs", "tradeoff", "better"),
    "mechanism_map": ("why", "how", "mechanism", "cause", "driver", "pathway"),
    "evidence_check": ("evidence", "support", "claim", "prove", "true", "does"),
    "policy_scan": ("policy", "regulation", "law", "government", "permit", "approval"),
    "risk_report": ("risk", "threat", "exposure", "impact", "vulnerability"),
    "controversy_map": ("debate", "controversy", "disagree", "criticism", "opposition"),
}

DISCIPLINE_HINTS = {
    "computer_science": ("ai", "model", "compute", "chip", "gpu", "data center", "cloud"),
    "energy_policy": ("grid", "electricity", "energy", "transmission", "power", "utility"),
    "water_environment": ("water", "cooling", "environment", "drought", "emissions"),
    "public_policy": ("policy", "regulation", "permit", "government", "public", "official"),
    "economics_business": ("market", "investment", "company", "cost", "supply", "demand"),
    "social_political": ("community", "opposition", "public opinion", "labor", "resident"),
    "biomedical": ("clinical", "health", "disease", "nutrition", "patient", "study"),
}

SUBDOMAIN_HINTS = {
    "data center expansion": ("data center", "datacenter", "expansion"),
    "energy": ("energy", "electricity", "power", "grid", "transmission"),
    "water": ("water", "cooling", "drought"),
    "cloud dependency": ("cloud", "dependency", "hyperscaler"),
    "sovereign AI": ("sovereign", "sovereignty"),
    "local opposition": ("opposition", "resident", "community", "protest"),
    "public opinion": ("public opinion", "public", "hearing"),
    "supply chain": ("supply", "chip", "gpu", "semiconductor"),
    "regulation": ("policy", "regulation", "permit", "approval"),
}

METHOD_HINTS = {
    "official_or_regulatory": ("official", "regulator", "government", "permit", "filing"),
    "journalistic": ("news", "media", "reported", "interview"),
    "academic_or_policy": ("study", "journal", "paper", "policy", "report"),
    "company_source": ("company", "corporate", "press release", "investor"),
    "public_signal": ("hearing", "resident", "community", "public opinion", "opposition"),
}


def run_research_workflow(payload: dict[str, Any]) -> dict[str, Any]:
    question = service.optional_text(payload, "question") or service.require_text(payload, "query")
    mode = str(payload.get("mode") or service.DEFAULT_MODE)
    policy = str(payload.get("policy") or service.DEFAULT_POLICY)
    supplied_frame = service.optional_dict(payload.get("research_frame"), "research_frame")
    requirements = service.optional_dict(payload.get("evidence_requirements"), "evidence_requirements")
    research_question = construct_research_question(question)
    research_frame = construct_research_frame(
        question=question,
        supplied_frame=supplied_frame,
        requirements=requirements,
        research_question=research_question,
    )
    enrichment = enrich_frame(
        question=question,
        deterministic_frame=research_frame,
        provider=payload.get("frame_enrichment_provider"),
        model=payload.get("frame_enrichment_model"),
    )
    research_frame = dict(enrichment["research_frame"])
    packet_response = service.evidence_packets(
        {
            **payload,
            "question": question,
            "mode": mode,
            "policy": policy,
            "research_frame": research_frame,
            "evidence_requirements": requirements,
        }
    )
    evidence = packet_evidence(packet_response)
    neighborhood = evidence_neighborhood(packet_response, evidence)
    concepts = concept_discipline_analysis(question, research_frame, evidence)
    guidance = research_guidance(
        research_question=research_question,
        research_frame=research_frame,
        packet_response=packet_response,
        neighborhood=neighborhood,
        concepts=concepts,
    )
    report = {
        "report_id": "research_workflow_layers_1_6_v1",
        "status": workflow_status(packet_response, guidance),
        "mode": mode,
        "policy": policy,
        "project_id": str(payload.get("project_id") or ""),
        "layers": {
            "research_question": research_question,
            "research_frame_constructor": {
                "status": "ready_for_review",
                "research_frame": research_frame,
                "constructed_fields": constructed_fields(supplied_frame, research_frame),
                "enrichment": {
                    "status": enrichment["status"],
                    "note": enrichment["note"],
                    "model_suggestion": enrichment["model_suggestion"],
                    "human_review_required": enrichment["human_review_required"],
                },
                "human_review_required": True,
            },
            "semantic_retrieval_engine": semantic_retrieval_summary(packet_response),
            "evidence_neighborhood_mapper": neighborhood,
            "concept_and_discipline_analyzer": concepts,
            "research_guidance_layer": guidance,
        },
        "evidence_packet_response": compact_packet_response(packet_response),
        "production_boundary": [
            "Layers 1-6 prepare and diagnose evidence; they do not finalize claims.",
            "Human-reviewed qrels and brief labels are still required for production quality claims.",
        ],
    }
    if service.optional_bool(payload.get("write_report"), default=True):
        write_report(report)
    return report


def construct_research_question(question: str) -> dict[str, Any]:
    normalized = normalize(question)
    question_type = first_matching_type(normalized)
    terms = content_terms(question)
    return {
        "status": "parsed",
        "question": question,
        "question_type": question_type,
        "risk_level": risk_level(normalized),
        "terms": terms,
        "unknown_terms": [],
        "requires_human_review": True,
        "routing": {
            "retrieval_lens": retrieval_lens(question_type),
            "default_top_k": 12 if question_type in {"risk_report", "controversy_map"} else 8,
        },
    }


def construct_research_frame(
    question: str,
    supplied_frame: dict[str, Any],
    requirements: dict[str, Any],
    research_question: dict[str, Any],
) -> dict[str, Any]:
    frame = dict(supplied_frame or {})
    frame.setdefault("topic", topic_from_question(question))
    frame.setdefault("question_type", research_question["question_type"])
    frame.setdefault("subdomains", inferred_subdomains(question, frame))
    frame.setdefault("regions", inferred_regions(question, frame))
    frame.setdefault("languages", inferred_languages(frame))
    frame.setdefault("disciplines", inferred_disciplines(question))
    frame.setdefault("methods", inferred_methods_from_requirements(requirements))
    frame.setdefault("representation_goals", inferred_representation_goals(research_question, requirements))
    frame.setdefault("exclusion_criteria", [])
    frame.setdefault("review_risks", review_risks(research_question, frame))
    return frame


def evidence_neighborhood(packet_response: dict[str, Any], evidence: list[dict[str, Any]]) -> dict[str, Any]:
    parent_counts = Counter(str(item.get("document_id") or item.get("chunk_id")) for item in evidence)
    source_counts = Counter(str(item.get("source_name") or "unknown") for item in evidence)
    cluster_counts = Counter(str(item.get("cluster_id") or "unknown") for item in evidence)
    diversity = ((packet_response.get("evidence_packets") or [{}])[0].get("source_diversity") or {})
    return {
        "status": "mapped" if evidence else "empty",
        "evidence_count": len(evidence),
        "distinct_parent_documents": len(parent_counts),
        "distinct_sources": len(source_counts),
        "distinct_clusters": len(cluster_counts),
        "dominant_source": most_common(source_counts),
        "dominant_parent": most_common(parent_counts),
        "dominant_cluster": most_common(cluster_counts),
        "parent_duplicate_count": sum(count - 1 for count in parent_counts.values() if count > 1),
        "source_diversity": diversity,
        "coverage_gaps": packet_response.get("coverage_gaps") or [],
        "frame_coverage": packet_response.get("frame_coverage") or {},
        "human_review_required": bool(evidence),
    }


def concept_discipline_analysis(
    question: str,
    research_frame: dict[str, Any],
    evidence: list[dict[str, Any]],
) -> dict[str, Any]:
    text = " ".join([question, json.dumps(research_frame, ensure_ascii=False), json.dumps(evidence, ensure_ascii=False)])
    disciplines = scored_hints(text, DISCIPLINE_HINTS)
    methods = scored_hints(text, METHOD_HINTS)
    concepts = sorted(set(content_terms(text)) & set(content_terms(" ".join(research_frame_terms(research_frame)))))
    topics = topic_signals(evidence)
    missing_disciplines = [
        discipline
        for discipline in research_frame.get("disciplines", [])
        if discipline not in {row["id"] for row in disciplines}
    ]
    return {
        "status": "analyzed",
        "concepts": concepts[:20],
        "topic_signals": topics,
        "disciplines": disciplines,
        "methods": methods,
        "missing_requested_disciplines": missing_disciplines,
        "slices": {
            "by_source_type": counted_field(evidence, "source_type"),
            "by_language": counted_field(evidence, "language"),
            "by_domain": counted_field(evidence, "domain"),
        },
        "human_review_required": True,
    }


def research_guidance(
    research_question: dict[str, Any],
    research_frame: dict[str, Any],
    packet_response: dict[str, Any],
    neighborhood: dict[str, Any],
    concepts: dict[str, Any],
) -> dict[str, Any]:
    actions = []
    frame_coverage = packet_response.get("frame_coverage") or {}
    diagnostics = packet_response.get("query_diagnostics") or {}
    retrieval_metrics = packet_response.get("retrieval_metrics") or {}
    if frame_coverage.get("status") in {"missing", "partial"}:
        actions.append(action("review_frame_gaps", "high", "Inspect missing research-frame dimensions before synthesis."))
    if diagnostics.get("weak_terms"):
        actions.append(action("approve_query_variants", "high", "Review weak terms and approve targeted query variants."))
    if neighborhood.get("distinct_sources", 0) < 3:
        actions.append(action("increase_source_breadth", "medium", "Retrieve more independent sources or relax source filters."))
    if neighborhood.get("parent_duplicate_count", 0) > 0:
        actions.append(action("apply_parent_diversification", "medium", "Cap final evidence to avoid repeated chunks from one parent."))
    if concepts.get("missing_requested_disciplines"):
        actions.append(action("fill_discipline_slice", "medium", "Run follow-up retrieval for missing discipline slices."))
    if retrieval_metrics.get("human_review_status") == "not_reviewed":
        actions.append(action("prepare_human_review", "high", "Prepare qrels and brief-level review before final claims."))
    if not actions:
        actions.append(action("proceed_to_review", "medium", "Evidence packet is ready for human inspection."))
    return {
        "status": "needs_revision" if any(row["priority"] == "high" for row in actions) else "ready_for_review",
        "next_actions": actions,
        "recommended_iteration": recommended_iteration(actions, research_question, research_frame),
        "synthesis_gate": {
            "status": "blocked_pending_human_review",
            "reason": "Layers 1-6 are automated diagnostics; synthesis should wait for human review or explicit acceptance of limitations.",
        },
    }


def semantic_retrieval_summary(packet_response: dict[str, Any]) -> dict[str, Any]:
    metrics = packet_response.get("retrieval_metrics") or {}
    packets = packet_response.get("evidence_packets") or []
    evidence_count = sum(len(packet.get("supporting_evidence") or []) for packet in packets)
    return {
        "status": "retrieved" if evidence_count else "no_evidence",
        "evidence_count": evidence_count,
        "estimated_confidence": metrics.get("estimated_confidence"),
        "source_diversity_status": metrics.get("source_diversity_status"),
        "human_review_status": metrics.get("human_review_status"),
        "support_confidence": metrics.get("support_confidence"),
    }


def workflow_status(packet_response: dict[str, Any], guidance: dict[str, Any]) -> str:
    if packet_response.get("status") != "complete":
        return "blocked"
    if guidance.get("status") == "needs_revision":
        return "needs_revision_before_synthesis"
    return "ready_for_human_review"


def compact_packet_response(packet_response: dict[str, Any]) -> dict[str, Any]:
    return {
        "report_id": packet_response.get("report_id"),
        "status": packet_response.get("status"),
        "contract_validation": packet_response.get("contract_validation"),
        "query_diagnostics": packet_response.get("query_diagnostics"),
        "frame_coverage": packet_response.get("frame_coverage"),
        "coverage_gaps": packet_response.get("coverage_gaps"),
        "retrieval_metrics": packet_response.get("retrieval_metrics"),
        "evidence_packets": [
            {
                "packet_id": packet.get("packet_id"),
                "supporting_evidence": packet.get("supporting_evidence") or [],
                "conflicting_evidence": packet.get("conflicting_evidence") or [],
                "limitations": packet.get("limitations") or [],
                "source_diversity": packet.get("source_diversity") or {},
            }
            for packet in packet_response.get("evidence_packets") or []
        ],
    }


def packet_evidence(packet_response: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        item
        for packet in packet_response.get("evidence_packets", [])
        for item in packet.get("supporting_evidence", [])
    ]


def first_matching_type(normalized: str) -> str:
    for question_type, hints in QUESTION_TYPES.items():
        if any(hint in normalized for hint in hints):
            return question_type
    return "exploratory"


def retrieval_lens(question_type: str) -> str:
    return {
        "comparative": "research_lens_comparison",
        "mechanism_map": "mechanism_map",
        "evidence_check": "canonical_depth",
        "policy_scan": "source_type_balanced",
        "risk_report": "broad_discovery_with_conflicts",
        "controversy_map": "conflict_aware",
    }.get(question_type, "broad_discovery")


def risk_level(normalized: str) -> str:
    if any(term in normalized for term in ("medical", "health", "legal", "safety", "policy", "risk")):
        return "high"
    if any(term in normalized for term in ("claim", "evidence", "compare")):
        return "medium"
    return "low"


def topic_from_question(question: str) -> str:
    terms = content_terms(question)
    return " ".join(terms[:8]) or question.strip()


def inferred_subdomains(question: str, frame: dict[str, Any]) -> list[str]:
    normalized = normalize(question)
    inferred = [
        subdomain
        for subdomain, hints in SUBDOMAIN_HINTS.items()
        if any(hint in normalized for hint in hints)
    ]
    if inferred:
        return inferred[:8]
    terms = [
        term
        for term in content_terms(question)
        if term not in {"emerging", "risk", "risks", "around", "main", "current"}
    ]
    return terms[:6] or list(frame.get("subdomains") or [])


def inferred_regions(question: str, frame: dict[str, Any]) -> list[str]:
    known = [
        "Latin America",
        "Brazil",
        "Chile",
        "Mexico",
        "United States",
        "Europe",
        "China",
        "India",
        "Africa",
    ]
    normalized = normalize(question)
    found = [region for region in known if normalize(region) in normalized]
    return found or list(frame.get("regions") or [])


def inferred_languages(frame: dict[str, Any]) -> list[str]:
    return list(frame.get("languages") or ["English"])


def inferred_disciplines(question: str) -> list[str]:
    return [row["id"] for row in scored_hints(question, DISCIPLINE_HINTS)[:4]]


def inferred_methods_from_requirements(requirements: dict[str, Any]) -> list[str]:
    source_types = requirements.get("minimum_source_types") if isinstance(requirements, dict) else []
    if not source_types:
        return []
    return [str(source_type) for source_type in source_types]


def inferred_representation_goals(research_question: dict[str, Any], requirements: dict[str, Any]) -> list[str]:
    goals = []
    if research_question["question_type"] in {"risk_report", "controversy_map", "comparative"}:
        goals.extend(["source-type diversity", "viewpoint diversity"])
    if isinstance(requirements, dict) and requirements.get("include_source_diversity"):
        goals.append("source diversity")
    return sorted(set(goals))


def review_risks(research_question: dict[str, Any], frame: dict[str, Any]) -> list[str]:
    risks = ["human_review_required"]
    if research_question["risk_level"] == "high":
        risks.append("high_stakes_question")
    if not frame.get("regions"):
        risks.append("scope_region_unspecified")
    if research_question["question_type"] == "exploratory":
        risks.append("broad_question_may_need_refinement")
    return risks


def scored_hints(text: str, hint_map: dict[str, tuple[str, ...]]) -> list[dict[str, Any]]:
    normalized = normalize(text)
    rows = []
    for identifier, hints in hint_map.items():
        matched = [hint for hint in hints if hint in normalized]
        if matched:
            rows.append({"id": identifier, "score": round(len(matched) / len(hints), 6), "matched_terms": matched})
    return sorted(rows, key=lambda row: (row["score"], row["id"]), reverse=True)


def counted_field(evidence: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts = Counter(str(item.get(field) or "unknown") for item in evidence)
    return dict(sorted(counts.items()))


def topic_signals(evidence: list[dict[str, Any]], topic_count: int = 6) -> list[dict[str, Any]]:
    documents = [
        content_terms(" ".join(str(item.get(field) or "") for field in ("title", "text", "domain", "source_type")))
        for item in evidence
    ]
    document_frequency = Counter(term for terms in documents for term in set(terms))
    total = len(documents) or 1
    scored = Counter()
    for terms in documents:
        for term in terms:
            if len(term) < 4:
                continue
            idf = 1.0 + (total / max(document_frequency[term], 1))
            scored[term] += idf
    return [
        {"term": term, "score": round(score, 6), "document_frequency": document_frequency[term]}
        for term, score in scored.most_common(topic_count)
    ]


def research_frame_terms(frame: dict[str, Any]) -> list[str]:
    values = []
    for value in frame.values():
        if isinstance(value, list):
            values.extend(str(item) for item in value)
        else:
            values.append(str(value))
    return values


def constructed_fields(supplied_frame: dict[str, Any], research_frame: dict[str, Any]) -> list[str]:
    supplied = set((supplied_frame or {}).keys())
    return sorted(key for key in research_frame if key not in supplied)


def action(identifier: str, priority: str, description: str) -> dict[str, str]:
    return {"id": identifier, "priority": priority, "description": description}


def recommended_iteration(
    actions: list[dict[str, str]],
    research_question: dict[str, Any],
    research_frame: dict[str, Any],
) -> dict[str, Any]:
    approved_terms = research_question.get("terms", [])[:5]
    frame_terms = [
        term
        for field in ("subdomains", "regions", "disciplines")
        for term in research_frame.get(field, [])
    ][:6]
    base = " ".join(approved_terms)
    return {
        "query_variants_to_review": sorted(
            set([f"{base} {term}".strip() for term in frame_terms if term and normalize(term) not in normalize(base)])
        )[:5],
        "blocking_actions": [row["id"] for row in actions if row["priority"] == "high"],
    }


def most_common(counts: Counter) -> dict[str, Any]:
    if not counts:
        return {"id": "", "count": 0}
    identifier, count = counts.most_common(1)[0]
    return {"id": identifier, "count": count}


def content_terms(text: str) -> list[str]:
    stop = {
        "about",
        "after",
        "against",
        "and",
        "are",
        "around",
        "can",
        "current",
        "does",
        "emerging",
        "for",
        "from",
        "how",
        "into",
        "main",
        "risk",
        "risks",
        "the",
        "this",
        "what",
        "when",
        "where",
        "which",
        "with",
    }
    return [token for token in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]+", normalize(text)) if token not in stop and len(token) >= 3]


def normalize(text: Any) -> str:
    return str(text or "").lower().replace("_", " ").replace("-", " ")


def write_report(report: dict[str, Any]) -> None:
    settings = load_settings()
    slug = service.safe_report_slug(report["layers"]["research_question"]["question"])
    json_path = settings.reports_dir / f"research_workflow_{report['mode']}_{slug}.json"
    md_path = settings.reports_dir / f"research_workflow_{report['mode']}_{slug}.md"
    report_io.write_json(json_path, report)
    report_io.write_markdown(md_path, render_markdown(report))


def render_markdown(report: dict[str, Any]) -> str:
    actions = "\n".join(
        f"- `{row['priority']}` `{row['id']}`: {row['description']}"
        for row in report["layers"]["research_guidance_layer"]["next_actions"]
    )
    question = report["layers"]["research_question"]
    retrieval = report["layers"]["semantic_retrieval_engine"]
    concepts = report["layers"]["concept_and_discipline_analyzer"]
    return f"""# Research Workflow Layers 1-6

Status: `{report['status']}`

Question type: `{question['question_type']}`

Retrieval evidence count: `{retrieval['evidence_count']}`

Disciplines: `{', '.join(row['id'] for row in concepts['disciplines']) or 'none'}`

## Next Actions

{actions}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CANON research workflow layers 1-6.")
    parser.add_argument("question")
    parser.add_argument("--mode", default=service.DEFAULT_MODE)
    parser.add_argument("--policy", default=service.DEFAULT_POLICY)
    parser.add_argument("--project-id", default="")
    parser.add_argument("--frame-json")
    parser.add_argument("--requirements-json")
    parser.add_argument("--frame-enrichment-provider", default=None)
    parser.add_argument("--frame-enrichment-model", default=None)
    parser.add_argument("--no-write-report", action="store_true")
    args = parser.parse_args()
    payload = {
        "question": args.question,
        "mode": args.mode,
        "policy": args.policy,
        "project_id": args.project_id,
        "write_report": not args.no_write_report,
        "frame_enrichment_provider": args.frame_enrichment_provider,
        "frame_enrichment_model": args.frame_enrichment_model,
    }
    if args.frame_json:
        payload["research_frame"] = json.loads(Path(args.frame_json).read_text(encoding="utf-8"))
    if args.requirements_json:
        payload["evidence_requirements"] = json.loads(Path(args.requirements_json).read_text(encoding="utf-8"))
    print(json.dumps(run_research_workflow(payload), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
