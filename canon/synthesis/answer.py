from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from canon.claims.conflict import load_or_detect_conflicts
from canon.claims.extract import load_or_extract_claims
from canon.config import load_settings
from canon.eval.contracts import CONTRACT_VERSION, validate_report
from canon.evidence.chunk_context import expand_evidence_with_parent_context
from canon.evidence.context import assemble_context_packet
from canon.evidence.corroboration import assess_corroboration
from canon.generation.providers import get_generation_provider
from canon.retrieval.experiment import run as run_retrieval
from canon.retrieval.query_diagnostics import content_terms, term_matches_tokens
from canon.retrieval.tokenize import tokenize
from canon.synthesis.prompting import (
    build_generation_payload,
    compose_cited_template_answer,
    compose_disciplined_prompt,
)
from canon.synthesis.verification import verify_generation


GENERIC_QUERY_TERMS = {
    "about",
    "answer",
    "corpus",
    "does",
    "evidence",
    "literature",
    "research",
    "science",
    "says",
    "say",
    "social",
    "studies",
    "this",
    "what",
    "where",
    "which",
}

CONFLICT_NOTE_LIMIT = 12


def synthesize(
    query: str,
    policy: str,
    mode: str,
    top_k: int | None = None,
    generator_name: str = "template",
    generator_model: str | None = None,
) -> dict:
    settings = load_settings()
    retrieval = run_retrieval(query, policy, mode, top_k)
    claims_report = load_or_extract_claims(mode)
    conflicts_report = load_or_detect_conflicts(mode)
    if policy == "conflict_aware":
        retrieval = augment_retrieval_with_conflicts(
            query=query,
            mode=mode,
            retrieval=retrieval,
            claims=claims_report["claims"],
            conflicts=conflicts_report["conflicts"],
        )
    claims_by_chunk = claims_for_chunks(claims_report["claims"], retrieval["results"])
    processed_chunks = load_processed_chunks(settings.data_dir, mode)
    candidate_evidence = build_evidence(
        retrieval["results"],
        claims_by_chunk,
        chunks=processed_chunks or None,
    )
    corroboration = assess_corroboration(candidate_evidence)
    conflict_note_rows = conflict_notes(conflicts_report["conflicts"], claims_by_chunk, query)
    context_packet = assemble_context_packet(
        query=query,
        evidence=candidate_evidence,
        corroboration=corroboration,
        conflicts=conflict_note_rows,
        max_items=top_k or retrieval["top_k"],
    )
    evidence = [item["evidence"] for item in context_packet["items"]]
    generation_payload = build_generation_payload(query, context_packet)
    prompt = compose_answer_prompt(query, context_packet, generation_payload, generator_name)
    generator = get_generation_provider(generator_name, generator_model)
    generated = generator.generate(prompt)
    support = support_assessment(query, evidence, retrieval["results"], corroboration)
    answer = grounded_answer_with_support(generated.text, support)
    citations = build_citations(evidence)
    verification = verify_generation(answer, citations, evidence, context_packet)
    report = {
        "contract_version": CONTRACT_VERSION,
        "query": query,
        "policy": policy,
        "mode": mode,
        "generator": {"provider": generated.provider, "model": generated.model},
        "top_k": retrieval["top_k"],
        "answer": answer,
        "citations": citations,
        "evidence": evidence,
        "candidate_evidence": candidate_evidence,
        "context_packet": context_packet,
        "generation_payload": generation_payload,
        "post_generation_verification": verification,
        "importance_summary": summarize_importance(retrieval["results"]),
        "support_assessment": support,
        "corroboration_assessment": corroboration,
        "conflict_notes": conflict_note_rows,
        "limitations": limitations(evidence, support, corroboration),
        "retrieval": retrieval,
    }
    report["contract_validation"] = validate_report(report, "synthesis")
    slug = "".join(character if character.isalnum() else "-" for character in query.lower())[:60].strip("-")
    write_json(settings.reports_dir / f"synthesis_{mode}_{policy}_{slug}.json", report)
    return report


def claims_for_chunks(claims: list[dict], results: list[dict]) -> dict[str, list[dict]]:
    result_ids = {result["chunk_id"] for result in results}
    grouped: dict[str, list[dict]] = {chunk_id: [] for chunk_id in result_ids}
    for claim in claims:
        chunk_id = claim["chunk_id"]
        if chunk_id in grouped:
            grouped[chunk_id].append(claim)
    for chunk_claims in grouped.values():
        chunk_claims.sort(key=lambda claim: claim["confidence"], reverse=True)
    return grouped


def build_evidence(results: list[dict], claims_by_chunk: dict[str, list[dict]], chunks: list[dict] | None = None) -> list[dict]:
    evidence = []
    for result in results:
        claims = claims_by_chunk.get(result["chunk_id"], [])
        selected_claim = claims[0] if claims else None
        evidence.append(
            {
                "citation_id": f"C{len(evidence) + 1}",
                "chunk_id": result["chunk_id"],
                "work_id": result["work_id"],
                "title": result["title"],
                "source_name": result["source_name"],
                "year": result["year"],
                "rank": result["rank"],
                "cluster_id": result["cluster_id"],
                "final_score": result["final_score"],
                "components": result["components"],
                "decision": result.get("decision", {}),
                "explanation": result.get("explanation", {}),
                "conflict_injected": bool(result.get("conflict_injected")),
                "claim": selected_claim,
                "preview": result["preview"],
            }
        )
    if chunks is None:
        return evidence
    return expand_evidence_with_parent_context(evidence, chunks)


def load_processed_chunks(data_dir: Path, mode: str) -> list[dict]:
    path = data_dir / "processed" / f"chunks_{mode}.json"
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def augment_retrieval_with_conflicts(
    query: str,
    mode: str,
    retrieval: dict,
    claims: list[dict],
    conflicts: list[dict],
) -> dict:
    results = [dict(result) for result in retrieval["results"]]
    if not results or not conflicts:
        return retrieval
    top_k = retrieval["top_k"]
    query_tokens = set(tokenize(query))
    claims_by_id = {claim["id"]: claim for claim in claims}
    selected_chunk_ids = {result["chunk_id"] for result in results}
    selected_claim_ids = {
        claim["id"]
        for claim in claims
        if claim["chunk_id"] in selected_chunk_ids
    }
    candidate_chunk_ids = conflict_candidate_chunk_ids(
        conflicts=conflicts,
        claims_by_id=claims_by_id,
        selected_claim_ids=selected_claim_ids,
        query_tokens=query_tokens,
    )
    missing_chunk_ids = [chunk_id for chunk_id in candidate_chunk_ids if chunk_id not in selected_chunk_ids]
    if not missing_chunk_ids:
        return retrieval
    broad = run_retrieval(query, retrieval["policy"], mode, top_k=500)
    broad_by_chunk = {result["chunk_id"]: dict(result) for result in broad["results"]}
    injected = []
    for chunk_id in missing_chunk_ids:
        if chunk_id not in broad_by_chunk:
            continue
        result = broad_by_chunk[chunk_id]
        result["conflict_injected"] = True
        injected.append(result)
    if not injected:
        return retrieval
    injected.sort(key=lambda result: result["rank"])
    focused_base_count = count_focused_results(query, results)
    keep_count = max(1, min(top_k - 1, focused_base_count or (top_k - max(1, top_k // 2))))
    injection_limit = min(len(injected), top_k - keep_count)
    kept = conflict_aware_base_results(query, results, keep_count)
    selected_injected = injected[:injection_limit]
    augmented = kept + selected_injected
    for rank, result in enumerate(augmented, start=1):
        result["rank"] = rank
    updated = dict(retrieval)
    updated["results"] = augmented[:top_k]
    updated["conflict_augmentation"] = {
        "candidate_chunk_count": len(candidate_chunk_ids),
        "injected_chunk_ids": [result["chunk_id"] for result in selected_injected],
    }
    return updated


def conflict_aware_base_results(query: str, results: list[dict], keep_count: int) -> list[dict]:
    if keep_count <= 0:
        return []
    focus_terms = content_terms(query)
    if not focus_terms:
        return results[:keep_count]
    scored = [
        (result_focus_coverage(result, focus_terms), result["final_score"], index, result)
        for index, result in enumerate(results)
    ]
    ranked = sorted(scored, key=lambda row: (row[0], row[1], -row[2]), reverse=True)
    selected = sorted(ranked[:keep_count], key=lambda row: row[2])
    return [row[3] for row in selected]


def count_focused_results(query: str, results: list[dict]) -> int:
    focus_terms = content_terms(query)
    if not focus_terms:
        return len(results)
    return sum(1 for result in results if result_focus_coverage(result, focus_terms) > 0.0)


def result_focus_coverage(result: dict, focus_terms: list[str]) -> float:
    text = " ".join([result.get("title") or "", result.get("preview") or ""])
    tokens = set(tokenize(text))
    if not focus_terms:
        return 0.0
    return len([term for term in focus_terms if term_matches_tokens(term, tokens)]) / len(focus_terms)


def conflict_candidate_chunk_ids(
    conflicts: list[dict],
    claims_by_id: dict[str, dict],
    selected_claim_ids: set[str],
    query_tokens: set[str],
) -> list[str]:
    chunk_ids = []
    seen = set()
    focus_tokens = focused_query_tokens(query_tokens)
    for conflict in conflicts:
        left_id = conflict["left_claim_id"]
        right_id = conflict["right_claim_id"]
        selected_match = left_id in selected_claim_ids or right_id in selected_claim_ids
        topic_match = conflict_topic_overlap(conflict, focus_tokens) >= 2
        if not selected_match and not topic_match:
            continue
        for claim_id in (left_id, right_id):
            claim = claims_by_id.get(claim_id)
            if not claim:
                continue
            chunk_id = claim["chunk_id"]
            if chunk_id not in seen:
                seen.add(chunk_id)
                chunk_ids.append(chunk_id)
    return chunk_ids


def compose_answer(query: str, evidence: list[dict]) -> str:
    if not evidence:
        return "The current corpus does not provide enough retrieved evidence to answer this query."
    query_terms = content_terms(query)
    opening = (
        "Across the retrieved literature, the strongest grounded answer is that the evidence "
        "is mixed and should be read through the specific claims and methods represented here."
    )
    if query_terms:
        opening += f" The retrieval centered on: {', '.join(query_terms[:8])}."
    sentences = [opening]
    for item in evidence[:3]:
        claim = item["claim"]
        if claim:
            sentences.append(
                f"{item['citation_id']} reports a {claim['stance']} claim: {claim['text']}"
            )
        else:
            sentences.append(
                f"{item['citation_id']} contributes contextual evidence from {item['title']}."
            )
    return " ".join(sentences)


def compose_answer_prompt(
    query: str,
    context_or_evidence: dict | list[dict],
    generation_payload: dict | None = None,
    generator_name: str = "template",
) -> str:
    if isinstance(context_or_evidence, list):
        return compose_answer(query, context_or_evidence)
    if generator_name in {"template", "local"}:
        return compose_cited_template_answer(query, context_or_evidence)
    return compose_disciplined_prompt(generation_payload or build_generation_payload(query, context_or_evidence))


def grounded_answer_with_support(answer: str, support: dict) -> str:
    if support["support_level"] != "weak":
        return answer
    prefix = (
        "The retrieved corpus provides weak support for this query, so this should be treated "
        "as an abstention rather than a grounded answer."
    )
    return f"{prefix} {answer}"


def build_citations(evidence: list[dict]) -> list[dict]:
    return [
        {
            "citation_id": item["citation_id"],
            "title": item["title"],
            "source_name": item["source_name"],
            "year": item["year"],
            "chunk_id": item["chunk_id"],
            "rank": item["rank"],
        }
        for item in evidence
    ]


def summarize_importance(results: list[dict]) -> dict:
    if not results:
        return {
            "distinct_works": 0,
            "distinct_sources": 0,
            "distinct_clusters": 0,
            "average_components": {},
        }
    component_names = sorted({name for result in results for name in result["components"]})
    averages = {}
    for name in component_names:
        averages[name] = round(
            sum(float(result["components"].get(name, 0.0)) for result in results) / len(results),
            6,
        )
    return {
        "distinct_works": len({result["work_id"] for result in results}),
        "distinct_sources": len({result["source_name"] for result in results}),
        "distinct_clusters": len(
            {result["cluster_id"] for result in results if result.get("cluster_id") is not None}
        ),
        "average_components": averages,
        "top_sources": dict(Counter(result["source_name"] or "UNKNOWN" for result in results).most_common(5)),
    }


def support_assessment(
    query: str,
    evidence: list[dict],
    results: list[dict],
    corroboration: dict | None = None,
) -> dict:
    focus_terms = content_terms(query)
    if not evidence:
        return {
            "support_level": "none",
            "support_confidence": 0.0,
            "query_term_coverage": 0.0,
            "covered_focus_terms": [],
            "missing_focus_terms": focus_terms,
            "focus_terms": focus_terms,
            "corroboration": (corroboration or {}).get("summary", {}),
        }
    evidence_tokens = set()
    for item in evidence:
        evidence_tokens.update(tokenize(item.get("title") or ""))
        evidence_tokens.update(tokenize(item.get("preview") or ""))
        claim = item.get("claim")
        if claim:
            evidence_tokens.update(tokenize(claim.get("text") or ""))
    covered = [token for token in focus_terms if term_matches_tokens(token, evidence_tokens)]
    term_coverage = round(len(covered) / len(focus_terms), 6) if focus_terms else 0.0
    components = average_components(results)
    context_relevance = float(components.get("relevance", 0.0))
    semantic_alignment = float(components.get("semantic_similarity", 0.0))
    confidence = round((term_coverage * 0.65) + (context_relevance * 0.2) + (semantic_alignment * 0.15), 6)
    corroboration_summary = (corroboration or {}).get("summary", {})
    max_independent_support = int(corroboration_summary.get("max_independent_support") or 0)
    max_echo_risk = float(corroboration_summary.get("max_echo_risk") or 0.0)
    if max_independent_support >= 2:
        confidence = min(1.0, round(confidence + 0.08, 6))
    elif evidence and max_independent_support <= 1:
        confidence = min(confidence, 0.62)
    if max_echo_risk >= 0.45:
        confidence = min(confidence, 0.42)
    if int(corroboration_summary.get("total_contradictions") or 0) > 0:
        confidence = min(confidence, 0.58)
    if focus_terms and term_coverage == 0.0:
        confidence = min(confidence, 0.3)
    elif focus_terms and term_coverage < 0.25:
        confidence = min(confidence, 0.34)
    if focus_terms and not covered:
        support_level = "weak"
    elif confidence < 0.35:
        support_level = "weak"
    elif confidence < 0.55:
        support_level = "moderate"
    else:
        support_level = "strong"
    return {
        "support_level": support_level,
        "support_confidence": confidence,
        "query_term_coverage": term_coverage,
        "covered_focus_terms": covered,
        "missing_focus_terms": [token for token in focus_terms if token not in covered],
        "focus_terms": focus_terms,
        "context_relevance": round(context_relevance, 6),
        "semantic_alignment": round(semantic_alignment, 6),
        "corroboration": corroboration_summary,
    }


def average_components(results: list[dict]) -> dict[str, float]:
    if not results:
        return {}
    component_names = sorted({name for result in results for name in result.get("components", {})})
    return {
        name: round(
            sum(float(result.get("components", {}).get(name, 0.0)) for result in results) / len(results),
            6,
        )
        for name in component_names
    }


def conflict_notes(
    conflicts: list[dict],
    claims_by_chunk: dict[str, list[dict]],
    query: str | None = None,
) -> list[dict]:
    selected_claim_ids = {
        claim["id"]
        for claims in claims_by_chunk.values()
        for claim in claims[:1]
    }
    query_tokens = focused_query_tokens(set(tokenize(query or "")))
    scored_notes = []
    for conflict in conflicts:
        selected_match = (
            conflict["left_claim_id"] in selected_claim_ids
            or conflict["right_claim_id"] in selected_claim_ids
        )
        overlap = conflict_topic_overlap(conflict, query_tokens)
        if not selected_match and overlap < 2:
            continue
        score = (2.0 if selected_match else 0.0) + overlap + float(conflict.get("score") or 0.0)
        scored_notes.append((score, conflict.get("id") or "", conflict))
    ranked = sorted(scored_notes, key=lambda row: (-row[0], row[1]))
    return [conflict for _score, _id, conflict in ranked[:CONFLICT_NOTE_LIMIT]]


def focused_query_tokens(tokens: set[str]) -> set[str]:
    return {
        token
        for token in tokens
        if len(token) >= 4 and token not in GENERIC_QUERY_TERMS
    }


def conflict_topic_overlap(conflict: dict, query_tokens: set[str]) -> int:
    return len(query_tokens & set(conflict.get("shared_topic_tokens") or []))


def limitations(
    evidence: list[dict],
    support: dict | None = None,
    corroboration: dict | None = None,
) -> list[str]:
    limits = []
    if len(evidence) < 3:
        limits.append("The answer is based on fewer than three retrieved evidence items.")
    if not any(item["claim"] for item in evidence):
        limits.append("No extracted claim was attached to the retrieved chunks.")
    if len({item["cluster_id"] for item in evidence if item["cluster_id"] is not None}) < 2:
        limits.append("The retrieved evidence has limited graph-cluster diversity.")
    if support and support.get("support_level") == "weak":
        limits.append("The retrieved evidence weakly covers the query focus terms.")
    summary = (corroboration or {}).get("summary", {})
    if summary.get("overall_verdict") == "adversarial_echo_risk":
        limits.append("Corroboration is vulnerable to dependent repetition or adversarial echo.")
    elif summary.get("overall_verdict") == "single_source_or_dependent":
        limits.append("The retrieved support appears to come from a single source or dependent source chain.")
    if summary.get("overall_verdict") == "conflicted":
        limits.append("The retrieved evidence contains conflicting claims.")
    return limits


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a grounded CANON answer.")
    parser.add_argument("query")
    parser.add_argument("--policy", default="rag")
    parser.add_argument("--mode", default="dry_run")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--generator", default="template")
    parser.add_argument("--generator-model", default=None)
    args = parser.parse_args()
    print(
        json.dumps(
            synthesize(
                args.query,
                args.policy,
                args.mode,
                args.top_k,
                args.generator,
                args.generator_model,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
