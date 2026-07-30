from __future__ import annotations

import argparse
import json
import os
import re
import urllib.request
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Literal
from typing import Any

from canon.config import load_settings
from canon.product import report_io, service
from canon.product.research_workflow import run_research_workflow
from canon.retrieval.tokenize import tokenize
from canon.secrets import load_local_env


Stance = Literal["supports", "contradicts", "qualifies", "neutral", "undetermined"]


@dataclass(frozen=True)
class EvidenceLink:
    evidence_id: str
    stance: Stance
    entailment_score: float
    hedge_score: float
    excerpt_span: tuple[int, int]
    excerpt: str


@dataclass(frozen=True)
class SynthesizedClaim:
    claim_id: str
    claim: str
    links: list[EvidenceLink]
    net_support: float
    contested: bool
    disagreement_axis: str
    confidence: float


def run_stage2_synthesis(payload: dict[str, Any]) -> dict[str, Any]:
    question = service.optional_text(payload, "question") or service.require_text(payload, "query")
    workflow = service.optional_dict(payload.get("research_workflow"), "research_workflow")
    if not workflow:
        workflow = run_research_workflow({**payload, "question": question, "write_report": False})
    evidence = synthesis_evidence(workflow)
    max_claims = service.optional_int(payload.get("max_claims")) or 6
    model_provider = service.optional_text(payload, "stage2_model_provider") or service.optional_text(payload, "synthesis_provider")
    model_name = service.optional_text(payload, "stage2_model") or service.optional_text(payload, "synthesis_model")
    allow_external_data = service.optional_bool(payload.get("allow_external_stage2_data"), default=False)
    model_status = {"status": "deterministic_fallback", "provider": "deterministic", "model": "lexical-stance-v1"}
    if model_provider:
        try:
            claims = build_model_cited_claims(
                question=question,
                evidence=evidence,
                max_claims=max_claims,
                provider=model_provider,
                model=model_name,
                allow_external_data=allow_external_data,
            )
            model_status = {
                "status": "model_backed",
                "provider": model_provider,
                "model": model_name or default_stage2_model(model_provider),
            }
        except Exception as exc:  # noqa: BLE001
            claims = []
            model_status = {
                "status": "model_unavailable",
                "provider": model_provider,
                "model": model_name or default_stage2_model(model_provider),
                "reason": str(exc),
            }
    else:
        claims = build_cited_claims(question, evidence, max_claims=max_claims)
    disagreement_map = build_disagreement_map(claims, evidence)
    synthesis = build_synthesis(question, claims, evidence, workflow)
    gate = stage2_quality_gate(
        claims=claims,
        evidence=evidence,
        workflow=workflow,
        min_evidence_count=service.optional_int(payload.get("min_evidence_count")) or 2,
        min_distinct_sources=service.optional_int(payload.get("min_distinct_sources")) or 2,
    )
    report = {
        "report_id": "stage2_evidence_synthesis_v1",
        "status": stage2_status(gate, evidence),
        "question": question,
        "mode": workflow.get("mode") or str(payload.get("mode") or service.DEFAULT_MODE),
        "policy": workflow.get("policy") or str(payload.get("policy") or service.DEFAULT_POLICY),
        "stage1_report_id": workflow.get("report_id"),
        "stage1_status": workflow.get("status"),
        "stage2_model": model_status,
        "synthesis": synthesis,
        "cited_claims": claims,
        "disagreement_map": disagreement_map,
        "stage2_metrics": stage2_metrics(claims, disagreement_map),
        "evidence": evidence,
        "quality_gate": gate,
        "human_review_required": True,
        "production_boundary": [
            "Stage 2 creates a review-ready synthesis from retrieved evidence.",
            "Automated checks verify citation integrity and evidence overlap, not final factual correctness.",
            "Human review is still required before release-quality claims.",
        ],
    }
    if service.optional_bool(payload.get("write_report"), default=True):
        write_report(report)
    return report


def synthesis_evidence(workflow: dict[str, Any]) -> list[dict[str, Any]]:
    packet_response = workflow.get("evidence_packet_response") or {}
    packets = packet_response.get("evidence_packets") or []
    if not packets:
        return []
    rows = []
    for item in packets[0].get("supporting_evidence") or []:
        evidence_id = str(item.get("evidence_id") or item.get("citation_id") or f"C{len(rows) + 1}")
        rows.append(
            {
                "evidence_id": evidence_id,
                "chunk_id": str(item.get("chunk_id") or ""),
                "document_id": str(item.get("document_id") or item.get("work_id") or ""),
                "title": str(item.get("title") or ""),
                "source_name": str(item.get("source_name") or "unknown"),
                "source_type": str(item.get("source_type") or "unknown"),
                "language": str(item.get("language") or "unknown"),
                "domain": str(item.get("domain") or "unknown"),
                "cluster_id": str(item.get("cluster_id") or "unknown"),
                "text": evidence_text(item),
                "metadata": {
                    "evidence_scope": item.get("evidence_scope"),
                    "retrieval_stage": item.get("retrieval_stage"),
                    "rank": item.get("rank"),
                },
            }
        )
    return rows


def evidence_text(item: dict[str, Any]) -> str:
    text = item.get("text") or item.get("preview") or item.get("claim") or ""
    return compact_whitespace(str(text))[:900]


def build_cited_claims(question: str, evidence: list[dict[str, Any]], max_claims: int = 6) -> list[dict[str, Any]]:
    question_terms = content_terms(question)
    candidates = []
    for item in evidence[:max_claims]:
        focus = best_evidence_sentence(item["text"], question_terms) or item["text"][:220]
        claim_text = compact_sentence(focus)
        if not claim_text:
            claim_text = f"{item['title']} provides contextual evidence."
        candidates.append((claim_text, item))
    clusters = cluster_claim_candidates(candidates)
    claims = []
    for cluster in clusters[:max_claims]:
        claim_text = representative_claim([claim for claim, _item in cluster])
        links = [
            build_evidence_link(claim_text, item)
            for _claim, item in cluster
        ]
        synthesized = SynthesizedClaim(
            claim_id=f"S2C{len(claims) + 1}",
            claim=claim_text,
            links=links,
            net_support=net_support(links),
            contested=is_contested(links),
            disagreement_axis=disagreement_axis(claim_text, links, cluster),
            confidence=round(average([link.entailment_score for link in links]), 6),
        )
        claims.append(claim_to_dict(synthesized, cluster))
    return claims


def build_model_cited_claims(
    question: str,
    evidence: list[dict[str, Any]],
    max_claims: int,
    provider: str,
    model: str | None = None,
    allow_external_data: bool = False,
) -> list[dict[str, Any]]:
    if not evidence:
        return []
    if provider.lower() in {"openrouter", "openai"} and not allow_external_data:
        raise RuntimeError(
            "Model-backed Stage 2 would send evidence text to an external provider. "
            "Set allow_external_stage2_data=true only for synthetic, public, or explicitly approved evidence."
        )
    payload = call_stage2_model(
        provider=provider,
        model=model or default_stage2_model(provider),
        prompt=stage2_model_prompt(question, evidence, max_claims),
    )
    claims = sanitize_model_claims(payload, evidence)
    if not claims:
        raise ValueError("Stage 2 model returned no usable cited claims.")
    return claims[:max_claims]


def stage2_model_prompt(question: str, evidence: list[dict[str, Any]], max_claims: int) -> dict[str, Any]:
    return {
        "task": "Create a structured scholarly evidence synthesis. Return strict JSON only.",
        "question": question,
        "instructions": [
            "Extract atomic synthesis claims, not one summary per chunk.",
            "Cluster near-duplicate assertions across evidence items into one claim.",
            "When two evidence items make opposite assertions about the same proposition, create one claim for the proposition and attach one supports link plus one contradicts link.",
            "For example, evidence saying 'X increases Y' and evidence saying 'X does not increase Y' should become one contested claim, not two separate claims.",
            "For every claim, attach one or more evidence links.",
            "Each link stance must be one of supports, contradicts, qualifies, neutral, undetermined.",
            "Use qualifies for hedged or scope-limited support.",
            "Use contradicts only when the cited evidence challenges or negates the claim.",
            "Do not cite evidence IDs that are not supplied.",
            "Do not create final facts beyond the supplied evidence.",
        ],
        "schema": {
            "claims": [
                {
                    "claim": "string",
                    "confidence": "0..1",
                    "disagreement_axis": "measurement_or_operationalization | population_or_scope | mechanism_or_theory | temporal_scope | qualification_or_boundary_condition | not_contested | unspecified_contestation",
                    "links": [
                        {
                            "evidence_id": "string",
                            "stance": "supports | contradicts | qualifies | neutral | undetermined",
                            "entailment_score": "0..1",
                            "hedge_score": "0..1",
                            "excerpt": "short exact or near-exact evidence excerpt",
                        }
                    ],
                }
            ]
        },
        "max_claims": max_claims,
        "evidence": [
            {
                "evidence_id": item["evidence_id"],
                "title": item["title"],
                "source_name": item["source_name"],
                "source_type": item["source_type"],
                "text": item["text"][:1200],
            }
            for item in evidence
        ],
    }


def call_stage2_model(provider: str, model: str, prompt: dict[str, Any]) -> dict[str, Any]:
    normalized = provider.lower()
    if normalized == "openrouter":
        return call_openrouter_json(model, prompt)
    if normalized == "openai":
        return call_openai_json(model, prompt)
    raise ValueError(f"Unsupported Stage 2 model provider: {provider}")


def call_openrouter_json(model: str, prompt: dict[str, Any]) -> dict[str, Any]:
    load_local_env()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is required for model-backed Stage 2 synthesis.")
    request = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=json.dumps(
            {
                "model": model,
                "temperature": 0,
                "messages": [
                    {"role": "system", "content": "You return only strict JSON for evidence synthesis."},
                    {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
                ],
                "response_format": {"type": "json_object"},
            }
        ).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return json.loads(payload["choices"][0]["message"]["content"])


def call_openai_json(model: str, prompt: dict[str, Any]) -> dict[str, Any]:
    load_local_env()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for OpenAI Stage 2 synthesis.")
    request = urllib.request.Request(
        "https://api.openai.com/v1/responses",
        data=json.dumps(
            {
                "model": model,
                "input": [
                    {"role": "system", "content": "You return only strict JSON for evidence synthesis."},
                    {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
                ],
                "text": {"format": {"type": "json_object"}},
                "temperature": 0,
            }
        ).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return json.loads(extract_openai_text(payload))


def extract_openai_text(payload: dict[str, Any]) -> str:
    if payload.get("output_text"):
        return str(payload["output_text"])
    parts = []
    for item in payload.get("output", []):
        for content in item.get("content", []):
            if content.get("type") in {"output_text", "text"}:
                parts.append(str(content.get("text") or ""))
    return "\n".join(parts).strip()


def sanitize_model_claims(payload: dict[str, Any], evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    evidence_by_id = {item["evidence_id"]: item for item in evidence}
    rows = []
    for raw_claim in payload.get("claims") or []:
        if not isinstance(raw_claim, dict):
            continue
        claim_text = compact_sentence(str(raw_claim.get("claim") or ""), limit=260)
        if not claim_text:
            continue
        links = []
        for raw_link in raw_claim.get("links") or []:
            link = sanitize_model_link(raw_link, evidence_by_id)
            if link:
                links.append(link)
        if not links:
            continue
        synthesized = SynthesizedClaim(
            claim_id=f"S2C{len(rows) + 1}",
            claim=claim_text,
            links=links,
            net_support=net_support(links),
            contested=is_contested(links),
            disagreement_axis=sanitize_axis(str(raw_claim.get("disagreement_axis") or ""), links),
            confidence=clamped_float(raw_claim.get("confidence"), default=average([link.entailment_score for link in links])),
        )
        rows.append(claim_to_dict_from_links(synthesized, evidence_by_id))
    return rows


def sanitize_model_link(raw_link: Any, evidence_by_id: dict[str, dict[str, Any]]) -> EvidenceLink | None:
    if not isinstance(raw_link, dict):
        return None
    evidence_id = str(raw_link.get("evidence_id") or "")
    evidence = evidence_by_id.get(evidence_id)
    if not evidence:
        return None
    excerpt = compact_sentence(str(raw_link.get("excerpt") or ""), limit=260)
    if not excerpt:
        excerpt = best_evidence_sentence(evidence["text"], content_terms(evidence["text"])) or evidence["text"][:220]
    start = evidence["text"].find(excerpt)
    if start < 0:
        start = 0
    return EvidenceLink(
        evidence_id=evidence_id,
        stance=sanitize_stance(str(raw_link.get("stance") or "")),
        entailment_score=clamped_float(raw_link.get("entailment_score"), default=0.5),
        hedge_score=clamped_float(raw_link.get("hedge_score"), default=hedge_score(excerpt)),
        excerpt_span=(start, start + len(excerpt)),
        excerpt=excerpt,
    )


def claim_to_dict_from_links(claim: SynthesizedClaim, evidence_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    evidence_ids = [link.evidence_id for link in claim.links]
    stance_counts = Counter(link.stance for link in claim.links)
    return {
        "claim_id": claim.claim_id,
        "claim": claim.claim,
        "links": [asdict(link) for link in claim.links],
        "evidence_ids": evidence_ids,
        "support_status": "contested" if claim.contested else dominant_stance(stance_counts),
        "stance_counts": dict(sorted(stance_counts.items())),
        "net_support": claim.net_support,
        "contested": claim.contested,
        "disagreement_axis": claim.disagreement_axis,
        "confidence": claim.confidence,
        "source_names": sorted({evidence_by_id[evidence_id]["source_name"] for evidence_id in evidence_ids if evidence_id in evidence_by_id}),
        "source_name": evidence_by_id[evidence_ids[0]]["source_name"] if evidence_ids and evidence_ids[0] in evidence_by_id else "unknown",
    }


def sanitize_stance(value: str) -> Stance:
    normalized = value.strip().lower()
    if normalized in {"supports", "contradicts", "qualifies", "neutral", "undetermined"}:
        return normalized  # type: ignore[return-value]
    return "undetermined"


def sanitize_axis(value: str, links: list[EvidenceLink]) -> str:
    allowed = {
        "measurement_or_operationalization",
        "population_or_scope",
        "mechanism_or_theory",
        "temporal_scope",
        "qualification_or_boundary_condition",
        "not_contested",
        "unspecified_contestation",
    }
    normalized = value.strip().lower()
    if normalized in allowed:
        return normalized
    if is_contested(links):
        return "unspecified_contestation"
    if any(link.stance == "qualifies" for link in links):
        return "qualification_or_boundary_condition"
    return "not_contested"


def clamped_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = float(default)
    return round(max(0.0, min(1.0, number)), 6)


def default_stage2_model(provider: str) -> str:
    if provider.lower() == "openrouter":
        return "openai/gpt-4.1-mini"
    if provider.lower() == "openai":
        return "gpt-4.1-mini"
    return "unknown"


def cluster_claim_candidates(candidates: list[tuple[str, dict[str, Any]]]) -> list[list[tuple[str, dict[str, Any]]]]:
    clusters: list[list[tuple[str, dict[str, Any]]]] = []
    for claim, item in candidates:
        placed = False
        for cluster in clusters:
            cluster_claim = representative_claim([row[0] for row in cluster])
            if should_cluster_claims(claim, cluster_claim):
                cluster.append((claim, item))
                placed = True
                break
        if not placed:
            clusters.append([(claim, item)])
    clusters.sort(key=lambda cluster: (len(cluster), average([len(content_terms(row[0])) for row in cluster])), reverse=True)
    return clusters


def should_cluster_claims(left_claim: str, right_claim: str) -> bool:
    left = content_terms(left_claim)
    right = content_terms(right_claim)
    if jaccard(left, right) >= 0.42:
        return True
    left_core = claim_core_terms(left_claim)
    right_core = claim_core_terms(right_claim)
    if jaccard(left_core, right_core) >= 0.34 and (
        contains_contradiction_marker(left_claim.lower())
        or contains_contradiction_marker(right_claim.lower())
    ):
        return True
    return False


def claim_core_terms(text: str) -> set[str]:
    generic = {
        "after",
        "data",
        "does",
        "load",
        "review",
        "says",
        "show",
        "shows",
        "the",
    }
    return {canonical_term(token) for token in content_terms(text) if token not in generic}


def canonical_term(token: str) -> str:
    aliases = {
        "centers": "center",
        "increased": "increase",
        "increases": "increase",
        "logs": "log",
    }
    if token in aliases:
        return aliases[token]
    if token.endswith("s") and len(token) > 4:
        return token[:-1]
    return token


def representative_claim(claims: list[str]) -> str:
    if not claims:
        return ""
    candidates = [
        claim
        for claim in claims
        if not contains_contradiction_marker(claim.lower())
    ] or claims
    return sorted(candidates, key=lambda claim: (len(content_terms(claim)), len(claim)), reverse=True)[0]


def build_evidence_link(claim: str, evidence: dict[str, Any]) -> EvidenceLink:
    excerpt = best_evidence_sentence(evidence["text"], content_terms(claim)) or evidence["text"][:220]
    start = max(0, evidence["text"].find(excerpt))
    if start < 0:
        start = 0
    stance, entailment = classify_stance(claim, evidence)
    return EvidenceLink(
        evidence_id=evidence["evidence_id"],
        stance=stance,
        entailment_score=entailment,
        hedge_score=hedge_score(excerpt),
        excerpt_span=(start, start + len(excerpt)),
        excerpt=compact_sentence(excerpt, limit=260),
    )


def classify_stance(claim: str, evidence: dict[str, Any]) -> tuple[Stance, float]:
    claim_terms = content_terms(claim)
    evidence_text_value = " ".join([evidence.get("title", ""), evidence.get("text", "")])
    evidence_terms = content_terms(evidence_text_value)
    overlap = safe_ratio(len(claim_terms & evidence_terms), len(claim_terms))
    normalized = evidence_text_value.lower()
    if contains_contradiction_marker(normalized):
        return "contradicts", round(max(0.55, overlap), 6)
    if contains_qualification_marker(normalized) or hedge_score(evidence_text_value) >= 0.25:
        return "qualifies", round(max(0.5, overlap), 6)
    if overlap >= 0.5:
        return "supports", round(overlap, 6)
    if overlap >= 0.2:
        return "neutral", round(overlap, 6)
    return "undetermined", round(overlap, 6)


def contains_contradiction_marker(text: str) -> bool:
    markers = ("does not", "did not", "not attribute", "conflicting", "contradict", "opposite", "however")
    return contains_marker(text, markers)


def contains_qualification_marker(text: str) -> bool:
    markers = ("but", "although", "while", "under certain", "limited", "not representative", "depends")
    return contains_marker(text, markers)


def contains_marker(text: str, markers: tuple[str, ...]) -> bool:
    tokens = set(tokenize(text))
    for marker in markers:
        if " " in marker:
            if marker in text:
                return True
        elif marker in tokens or any(token.startswith(marker) for token in tokens if len(marker) >= 6):
            return True
    return False


def hedge_score(text: str) -> float:
    markers = {
        "may",
        "might",
        "could",
        "suggest",
        "possible",
        "likely",
        "appears",
        "limited",
        "uncertain",
        "estimate",
        "not representative",
    }
    normalized = text.lower()
    hits = sum(1 for marker in markers if marker in normalized)
    return round(min(1.0, hits / 4.0), 6)


def net_support(links: list[EvidenceLink]) -> float:
    weights = {"supports": 1.0, "qualifies": 0.45, "neutral": 0.0, "undetermined": 0.0, "contradicts": -1.0}
    return round(sum(weights[link.stance] * link.entailment_score for link in links), 6)


def is_contested(links: list[EvidenceLink]) -> bool:
    stances = {link.stance for link in links}
    return bool(stances & {"supports", "qualifies"} and "contradicts" in stances)


def disagreement_axis(
    claim: str,
    links: list[EvidenceLink],
    cluster: list[tuple[str, dict[str, Any]]],
) -> str:
    text = " ".join([claim, *(item.get("title", "") + " " + item.get("text", "") for _claim, item in cluster)]).lower()
    if any(term in text for term in ("measure", "coding", "indicator", "attribute", "quantify")):
        return "measurement_or_operationalization"
    if any(term in text for term in ("resident", "community", "sample", "population", "local")):
        return "population_or_scope"
    if any(term in text for term in ("mechanism", "cause", "driver", "because")):
        return "mechanism_or_theory"
    if any(term in text for term in ("temporal", "staged", "future", "trend", "over time")):
        return "temporal_scope"
    if is_contested(links):
        return "unspecified_contestation"
    if any(link.stance == "qualifies" for link in links):
        return "qualification_or_boundary_condition"
    return "not_contested"


def claim_to_dict(claim: SynthesizedClaim, cluster: list[tuple[str, dict[str, Any]]]) -> dict[str, Any]:
    evidence_ids = [link.evidence_id for link in claim.links]
    stance_counts = Counter(link.stance for link in claim.links)
    return {
        "claim_id": claim.claim_id,
        "claim": claim.claim,
        "links": [asdict(link) for link in claim.links],
        "evidence_ids": evidence_ids,
        "support_status": "contested" if claim.contested else dominant_stance(stance_counts),
        "stance_counts": dict(sorted(stance_counts.items())),
        "net_support": claim.net_support,
        "contested": claim.contested,
        "disagreement_axis": claim.disagreement_axis,
        "confidence": claim.confidence,
        "source_names": sorted({item["source_name"] for _claim, item in cluster}),
        "source_name": cluster[0][1]["source_name"] if cluster else "unknown",
    }


def dominant_stance(counts: Counter[str]) -> str:
    if not counts:
        return "undetermined"
    return counts.most_common(1)[0][0]


def best_evidence_sentence(text: str, question_terms: set[str]) -> str:
    sentences = split_sentences(text)
    if not sentences:
        return ""
    ranked = sorted(
        sentences,
        key=lambda sentence: (len(content_terms(sentence) & question_terms), len(content_terms(sentence))),
        reverse=True,
    )
    return ranked[0]


def claim_confidence(claim: str, evidence: dict[str, Any], question_terms: set[str]) -> float:
    claim_terms = content_terms(claim)
    evidence_terms = content_terms(" ".join([evidence["title"], evidence["text"]]))
    support_overlap = safe_ratio(len(claim_terms & evidence_terms), len(claim_terms))
    focus_overlap = safe_ratio(len(question_terms & evidence_terms), len(question_terms))
    return round(min(0.95, 0.35 + 0.45 * support_overlap + 0.20 * focus_overlap), 6)


def build_synthesis(
    question: str,
    claims: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
    workflow: dict[str, Any],
) -> dict[str, Any]:
    answer_sentences = [
        render_claim_sentence(claim)
        for claim in claims[:4]
    ]
    if not answer_sentences:
        answer_sentences.append("The current evidence packet does not contain enough evidence to synthesize an answer.")
    return {
        "answer": " ".join(answer_sentences),
        "summary_claim_count": len(claims),
        "citation_count": sum(len(claim["evidence_ids"]) for claim in claims),
        "contested_claim_count": sum(1 for claim in claims if claim.get("contested")),
        "evidence_scope": evidence_scope(evidence),
        "source_summary": source_summary(evidence),
        "contradictions_and_uncertainty": uncertainty_notes(workflow, evidence),
        "next_review_questions": next_review_questions(question, workflow, evidence),
    }


def render_claim_sentence(claim: dict[str, Any]) -> str:
    citations = ", ".join(claim.get("evidence_ids") or [])
    status = claim.get("support_status") or "cited"
    if status == "contested":
        return f"{claim['claim']} The retrieved evidence is contested on `{claim['disagreement_axis']}` [{citations}]."
    if status == "qualifies":
        return f"{claim['claim']} This claim is qualified by the cited evidence [{citations}]."
    if status == "contradicts":
        return f"The evidence challenges this proposition: {claim['claim']} [{citations}]."
    return f"{claim['claim']} [{citations}]"


def build_disagreement_map(claims: list[dict[str, Any]], evidence: list[dict[str, Any]]) -> dict[str, Any]:
    evidence_by_id = {row["evidence_id"]: row for row in evidence}
    clusters = []
    for claim in claims:
        links = claim.get("links") or []
        stance_groups: dict[str, list[dict[str, Any]]] = {}
        for link in links:
            evidence_row = evidence_by_id.get(str(link.get("evidence_id")), {})
            stance_groups.setdefault(str(link.get("stance") or "undetermined"), []).append(
                {
                    "evidence_id": link.get("evidence_id"),
                    "source_name": evidence_row.get("source_name", "unknown"),
                    "entailment_score": link.get("entailment_score", 0.0),
                    "hedge_score": link.get("hedge_score", 0.0),
                    "excerpt": link.get("excerpt", ""),
                }
            )
        clusters.append(
            {
                "claim_id": claim["claim_id"],
                "claim": claim["claim"],
                "contested": bool(claim.get("contested")),
                "disagreement_axis": claim.get("disagreement_axis", "not_contested"),
                "net_support": claim.get("net_support", 0.0),
                "stance_groups": dict(sorted(stance_groups.items())),
            }
        )
    return {
        "status": "mapped" if clusters else "empty",
        "claim_cluster_count": len(clusters),
        "contested_cluster_count": sum(1 for cluster in clusters if cluster["contested"]),
        "axes": dict(sorted(Counter(cluster["disagreement_axis"] for cluster in clusters).items())),
        "claim_clusters": clusters,
    }


def stage2_quality_gate(
    claims: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
    workflow: dict[str, Any],
    min_evidence_count: int = 2,
    min_distinct_sources: int = 2,
) -> dict[str, Any]:
    evidence_by_id = {row["evidence_id"]: row for row in evidence}
    checks = [
        check("evidence_present", len(evidence) >= min_evidence_count, {"evidence_count": len(evidence)}),
        check("claims_present", len(claims) > 0, {"claim_count": len(claims)}),
        check(
            "source_breadth",
            len({row["source_name"] for row in evidence}) >= min_distinct_sources,
            {"distinct_sources": len({row["source_name"] for row in evidence})},
        ),
        check("citation_integrity", all_known_citations(claims, evidence_by_id), {}),
        check("claim_evidence_overlap", unsupported_claim_count(claims, evidence_by_id) == 0, {
            "unsupported_claim_count": unsupported_claim_count(claims, evidence_by_id),
        }),
        check(
            "stage1_completed",
            workflow.get("status") not in {"failed", "empty"},
            {"stage1_status": workflow.get("status")},
        ),
    ]
    blockers = [
        row
        for row in checks
        if row["status"] == "fail"
        and row["id"] in {"evidence_present", "claims_present", "citation_integrity", "claim_evidence_overlap"}
    ]
    return {
        "status": "pass_pending_human_review" if not blockers else "blocked",
        "checks": checks,
        "unsupported_claims": unsupported_claims(claims, evidence_by_id),
        "human_review_required": True,
    }


def all_known_citations(claims: list[dict[str, Any]], evidence_by_id: dict[str, dict[str, Any]]) -> bool:
    return all(
        evidence_id in evidence_by_id
        for claim in claims
        for evidence_id in claim_evidence_ids(claim)
    )


def unsupported_claims(claims: list[dict[str, Any]], evidence_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for claim in claims:
        evidence_terms = set()
        for evidence_id in claim_evidence_ids(claim):
            evidence = evidence_by_id.get(str(evidence_id), {})
            evidence_terms |= content_terms(" ".join([str(evidence.get("title") or ""), str(evidence.get("text") or "")]))
        claim_terms = content_terms(str(claim.get("claim") or ""))
        overlap = safe_ratio(len(claim_terms & evidence_terms), len(claim_terms))
        if overlap < 0.5:
            rows.append({"claim_id": claim.get("claim_id"), "overlap": round(overlap, 6)})
    return rows


def unsupported_claim_count(claims: list[dict[str, Any]], evidence_by_id: dict[str, dict[str, Any]]) -> int:
    return len(unsupported_claims(claims, evidence_by_id))


def claim_evidence_ids(claim: dict[str, Any]) -> list[str]:
    if claim.get("links"):
        return [str(link.get("evidence_id")) for link in claim.get("links", []) if link.get("evidence_id")]
    return [str(evidence_id) for evidence_id in claim.get("evidence_ids", [])]


def stage2_metrics(claims: list[dict[str, Any]], disagreement_map: dict[str, Any]) -> dict[str, Any]:
    claim_count = len(claims)
    linked_claim_count = sum(1 for claim in claims if claim_evidence_ids(claim))
    contested_count = sum(1 for claim in claims if claim.get("contested"))
    multi_source_count = sum(1 for claim in claims if len(set(claim_evidence_ids(claim))) > 1)
    hedge_scores = [
        float(link.get("hedge_score", 0.0))
        for claim in claims
        for link in claim.get("links", [])
    ]
    return {
        "claim_count": claim_count,
        "linked_claim_ratio": round(safe_ratio(linked_claim_count, claim_count), 6),
        "multi_evidence_claim_count": multi_source_count,
        "contested_claim_count": contested_count,
        "disagreement_recall_proxy": round(safe_ratio(contested_count, claim_count), 6),
        "mean_hedge_score": round(average(hedge_scores), 6),
        "disagreement_axes": disagreement_map.get("axes", {}),
    }


def stage2_status(gate: dict[str, Any], evidence: list[dict[str, Any]]) -> str:
    if not evidence:
        return "blocked_insufficient_evidence"
    if gate["status"] == "blocked":
        return "blocked_quality_gate"
    return "ready_for_human_review"


def evidence_scope(evidence: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(str(item.get("metadata", {}).get("evidence_scope") or "unknown") for item in evidence)
    return dict(sorted(counts.items()))


def source_summary(evidence: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "distinct_sources": len({item["source_name"] for item in evidence}),
        "distinct_source_types": len({item["source_type"] for item in evidence}),
        "distinct_domains": len({item["domain"] for item in evidence}),
        "source_counts": dict(sorted(Counter(item["source_name"] for item in evidence).items())),
    }


def uncertainty_notes(workflow: dict[str, Any], evidence: list[dict[str, Any]]) -> list[str]:
    notes = []
    packet_response = workflow.get("evidence_packet_response") or {}
    for gap in packet_response.get("coverage_gaps") or []:
        notes.append(str(gap.get("gap") or gap))
    if len(evidence) < 3:
        notes.append("The synthesis is based on fewer than three evidence items.")
    if len({item["source_name"] for item in evidence}) < 2:
        notes.append("The evidence packet has limited independent source breadth.")
    guidance = ((workflow.get("layers") or {}).get("research_guidance_layer") or {})
    gate = guidance.get("synthesis_gate") or {}
    if gate.get("status"):
        notes.append(f"Stage 1 synthesis gate: {gate.get('status')}.")
    return dedupe(notes)[:8]


def next_review_questions(question: str, workflow: dict[str, Any], evidence: list[dict[str, Any]]) -> list[str]:
    questions = []
    guidance = ((workflow.get("layers") or {}).get("research_guidance_layer") or {})
    for action in guidance.get("next_actions") or []:
        questions.append(str(action.get("description") or action.get("id") or "Review evidence gap."))
    if evidence:
        questions.append("Do the cited evidence items directly support the synthesis claims?")
    else:
        questions.append(f"What additional retrieval is needed before answering: {question}")
    return dedupe(questions)[:6]


def check(check_id: str, passed: bool, details: dict[str, Any]) -> dict[str, Any]:
    return {"id": check_id, "status": "pass" if passed else "fail", "details": details}


def content_terms(text: str) -> set[str]:
    stop = {
        "about",
        "and",
        "are",
        "based",
        "does",
        "from",
        "into",
        "the",
        "this",
        "with",
    }
    return {token for token in tokenize(text) if len(token) >= 3 and token not in stop}


def split_sentences(text: str) -> list[str]:
    return [piece.strip() for piece in re.split(r"(?<=[.!?])\s+", text) if piece.strip()]


def compact_sentence(text: str, limit: int = 220) -> str:
    value = compact_whitespace(text)
    if len(value) <= limit:
        return value
    return value[:limit].rsplit(" ", 1)[0].rstrip(".,;:") + "."


def compact_whitespace(text: str) -> str:
    return " ".join(str(text or "").split())


def jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def dedupe(values: list[str]) -> list[str]:
    seen = set()
    rows = []
    for value in values:
        normalized = compact_whitespace(value).lower()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        rows.append(value)
    return rows


def safe_ratio(numerator: int | float, denominator: int | float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def average(values: list[int | float]) -> float:
    return sum(float(value) for value in values) / len(values) if values else 0.0


def write_report(report: dict[str, Any]) -> None:
    settings = load_settings()
    slug = "".join(character if character.isalnum() else "-" for character in report["question"].lower())[:60].strip("-")
    output = settings.reports_dir / f"stage2_synthesis_{report['mode']}_{slug or 'query'}.json"
    report_io.write_json(output, report)
    report_io.write_markdown(output.with_suffix(".md"), render_markdown(report))
    report["output_path"] = str(output).replace("\\", "/")


def render_markdown(report: dict[str, Any]) -> str:
    claims = "\n".join(
        f"- `{claim['claim_id']}` {claim['claim']} "
        f"({claim['support_status']}; {', '.join(claim['evidence_ids'])})"
        for claim in report["cited_claims"]
    )
    map_rows = "\n".join(
        f"- `{cluster['claim_id']}` contested=`{cluster['contested']}` axis=`{cluster['disagreement_axis']}`"
        for cluster in report.get("disagreement_map", {}).get("claim_clusters", [])
    )
    checks = "\n".join(f"- `{row['id']}`: `{row['status']}`" for row in report["quality_gate"]["checks"])
    return f"""# Stage 2 Evidence Synthesis

Status: `{report['status']}`

Question: {report['question']}

## Answer

{report['synthesis']['answer']}

## Cited Claims

{claims}

## Disagreement Map

{map_rows}

## Quality Gate

{checks}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 2 evidence synthesis.")
    parser.add_argument("query")
    parser.add_argument("--mode", default=service.DEFAULT_MODE)
    parser.add_argument("--policy", default=service.DEFAULT_POLICY)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--stage2-model-provider", default=None)
    parser.add_argument("--stage2-model", default=None)
    parser.add_argument("--allow-external-stage2-data", action="store_true")
    parser.add_argument("--write-report", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            run_stage2_synthesis(
                {
                    "query": args.query,
                    "mode": args.mode,
                    "policy": args.policy,
                    "top_k": args.top_k,
                    "stage2_model_provider": args.stage2_model_provider,
                    "stage2_model": args.stage2_model,
                    "allow_external_stage2_data": args.allow_external_stage2_data,
                    "write_report": args.write_report,
                }
            ),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
