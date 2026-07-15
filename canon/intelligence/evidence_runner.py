from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from canon.config import load_settings
from canon.product import service
from canon.product.report_io import write_json, write_markdown


REPORT_ID = "mock_evidence_intelligence_brief_v1"

DEFAULT_AI_INFRA_FRAME = {
    "topic": "AI data center expansion",
    "subdomains": [
        "energy",
        "water",
        "cloud dependency",
        "sovereign AI",
        "local opposition",
        "public opinion",
    ],
    "regions": ["Latin America", "Brazil", "Chile", "Mexico"],
    "languages": ["English", "Spanish", "Portuguese"],
    "question_type": "regional risk report",
    "representation_goals": ["regional diversity", "language diversity", "source-type diversity"],
}

DEFAULT_EVIDENCE_REQUIREMENTS = {
    "top_k": 8,
    "include_conflicts": True,
    "include_source_diversity": True,
    "include_query_diagnostics": True,
    "include_public_opinion": True,
    "minimum_source_types": ["official", "local_media", "company", "policy", "public_opinion"],
}


@dataclass(frozen=True)
class AgentSpec:
    agent_id: str
    name: str
    issue_keywords: tuple[str, ...]
    role: str


AGENTS = [
    AgentSpec(
        "technical_infrastructure",
        "Technical Infrastructure Agent",
        ("data center", "cloud", "grid", "substation", "interconnection", "region"),
        "Map infrastructure exposure and operational constraints.",
    ),
    AgentSpec(
        "energy_water",
        "Energy and Water Agent",
        ("energy", "electricity", "water", "cooling", "grid", "transmission"),
        "Identify energy, water, and cooling risks.",
    ),
    AgentSpec(
        "policy_controls",
        "Policy and Export Controls Agent",
        ("policy", "regulator", "permit", "sovereign", "public", "approval"),
        "Interpret regulatory, permitting, and sovereignty implications.",
    ),
    AgentSpec(
        "geopolitical",
        "Geopolitical Agent",
        ("foreign", "dependency", "sovereign", "latin america", "government", "negotiation"),
        "Surface cross-border dependency and strategic-exposure issues.",
    ),
    AgentSpec(
        "public_opinion",
        "Public Opinion and Social Signal Agent",
        ("resident", "public opinion", "hearing", "community", "opposition", "tariff"),
        "Treat public signals with caveats and avoid representativeness claims.",
    ),
]


def run_intelligence_brief(
    question: str,
    mode: str,
    policy: str = "rag",
    project_id: str = "ai_infra_geo_risk",
    research_frame: dict[str, Any] | None = None,
    evidence_requirements: dict[str, Any] | None = None,
    write_report: bool = True,
) -> dict[str, Any]:
    packet_response = service.evidence_packets(
        {
            "request_id": stable_request_id(question),
            "project_id": project_id,
            "question": question,
            "mode": mode,
            "policy": policy,
            "research_frame": research_frame or DEFAULT_AI_INFRA_FRAME,
            "evidence_requirements": evidence_requirements or DEFAULT_EVIDENCE_REQUIREMENTS,
        }
    )
    return run_from_packet_response(
        packet_response=packet_response,
        question=question,
        mode=mode,
        policy=policy,
        project_id=project_id,
        write_report=write_report,
    )


def run_from_packet_response(
    packet_response: dict[str, Any],
    question: str | None = None,
    mode: str | None = None,
    policy: str | None = None,
    project_id: str | None = None,
    write_report: bool = True,
) -> dict[str, Any]:
    evidence = evidence_index(packet_response)
    plan = build_agent_plan(packet_response)
    outputs = [run_agent(agent, evidence) for agent in plan]
    grounding = evaluate_grounding(outputs, evidence)
    duplicate_report = duplicate_agent_output_report(outputs)
    red_team = red_team_review(outputs, grounding, duplicate_report, packet_response)
    brief = synthesize_brief(
        question=question or packet_response.get("query", ""),
        packet_response=packet_response,
        agent_outputs=outputs,
        red_team=red_team,
        duplicate_report=duplicate_report,
    )
    report = {
        "report_id": REPORT_ID,
        "project_id": project_id or packet_response.get("project_id", ""),
        "mode": mode or packet_response.get("mode", ""),
        "policy": policy or packet_response.get("policy", ""),
        "question": question or packet_response.get("query", ""),
        "status": "blocked" if red_team["blocking_issue_count"] else "ready_for_human_review",
        "evidence_packet_request_id": packet_response.get("request_id", ""),
        "agent_plan": [agent_to_dict(agent) for agent in plan],
        "agent_outputs": outputs,
        "grounding_report": grounding,
        "duplicate_agent_output": duplicate_report,
        "red_team": red_team,
        "brief": brief,
        "human_review_status": "not_reviewed",
        "claim_boundaries": [
            "Every factual claim must cite evidence IDs from the packet response.",
            "This deterministic runner interprets retrieved evidence; it does not discover new evidence.",
            "The brief is ready for human review, not publication or operational use.",
        ],
    }
    if write_report:
        write_intelligence_report(report)
    return report


def build_agent_plan(packet_response: dict[str, Any]) -> list[AgentSpec]:
    evidence = list(evidence_index(packet_response).values())
    active_agents = [agent for agent in AGENTS if agent_has_matching_evidence(agent, evidence)]
    if len(active_agents) < 5:
        active_ids = {agent.agent_id for agent in active_agents}
        active_agents.extend(agent for agent in AGENTS if agent.agent_id not in active_ids)
    return active_agents[:5]


def run_agent(agent: AgentSpec, evidence: dict[str, dict[str, Any]]) -> dict[str, Any]:
    selected = select_agent_evidence(agent, evidence)
    if not selected:
        return {
            "agent_id": agent.agent_id,
            "agent_name": agent.name,
            "role": agent.role,
            "status": "insufficient_evidence",
            "claims": [],
            "insufficient_evidence": f"No packet evidence matched the {agent.name} lens.",
        }
    claims = [claim_from_evidence(agent, item) for item in selected[:2]]
    return {
        "agent_id": agent.agent_id,
        "agent_name": agent.name,
        "role": agent.role,
        "status": "complete",
        "claims": claims,
        "insufficient_evidence": "",
    }


def claim_from_evidence(agent: AgentSpec, item: dict[str, Any]) -> dict[str, Any]:
    evidence_id = str(item.get("evidence_id") or "")
    text = clean_text(item.get("text") or item.get("title") or "")
    source = item.get("source_name") or item.get("title") or "packet evidence"
    issue = first_matching_keyword(agent.issue_keywords, text) or agent.issue_keywords[0]
    return {
        "claim_id": f"{agent.agent_id}:{evidence_id}",
        "claim": f"{agent.name} flags {issue} based on {source}: {summarize_text(text)}",
        "evidence_ids": [evidence_id] if evidence_id else [],
        "issue_categories": [issue],
        "confidence": item_confidence(item),
    }


def select_agent_evidence(agent: AgentSpec, evidence: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    scored = []
    for item in evidence.values():
        haystack = evidence_haystack(item)
        score = sum(1 for keyword in agent.issue_keywords if keyword in haystack)
        if agent.agent_id == "public_opinion" and item.get("provenance") == "public_opinion":
            score += 2
        if score > 0:
            scored.append((score, int(item.get("rank") or 999), str(item.get("evidence_id")), item))
    return [item for _score, _rank, _id, item in sorted(scored, key=lambda row: (-row[0], row[1], row[2]))]


def evaluate_grounding(agent_outputs: list[dict[str, Any]], evidence: dict[str, dict[str, Any]]) -> dict[str, Any]:
    evidence_ids = set(evidence)
    unsupported = []
    claim_count = 0
    for output in agent_outputs:
        for claim in output.get("claims", []):
            claim_count += 1
            ids = [str(evidence_id) for evidence_id in claim.get("evidence_ids") or []]
            missing = [evidence_id for evidence_id in ids if evidence_id not in evidence_ids]
            if not ids:
                unsupported.append(
                    {
                        "agent_id": output.get("agent_id"),
                        "claim_id": claim.get("claim_id"),
                        "reason": "missing_evidence_ids",
                    }
                )
            elif missing:
                unsupported.append(
                    {
                        "agent_id": output.get("agent_id"),
                        "claim_id": claim.get("claim_id"),
                        "reason": "unknown_evidence_ids",
                        "unknown_evidence_ids": missing,
                    }
                )
    supported = claim_count - len(unsupported)
    return {
        "status": "pass" if not unsupported else "fail",
        "claim_count": claim_count,
        "supported_claim_count": supported,
        "unsupported_claim_count": len(unsupported),
        "grounded_claim_ratio": round(safe_ratio(supported, claim_count), 4),
        "unsupported_claims": unsupported,
    }


def red_team_review(
    agent_outputs: list[dict[str, Any]],
    grounding: dict[str, Any],
    duplicate_report: dict[str, Any],
    packet_response: dict[str, Any],
) -> dict[str, Any]:
    objections = []
    if grounding["unsupported_claim_count"]:
        objections.append(
            {
                "severity": "blocker",
                "issue": "unsupported_claims",
                "message": "At least one factual claim lacks valid evidence IDs.",
            }
        )
    if duplicate_report["duplicate_agent_output_rate"] > 0.2:
        objections.append(
            {
                "severity": "review",
                "issue": "duplicate_agent_output",
                "message": "Agent outputs are too similar; synthesis should avoid overstating independent support.",
            }
        )
    if packet_response.get("frame_coverage", {}).get("status") in {"partial", "missing"}:
        objections.append(
            {
                "severity": "review",
                "issue": "frame_coverage_gap",
                "message": "The evidence packet reports missing or partial research-frame coverage.",
            }
        )
    incomplete_agents = [output["agent_id"] for output in agent_outputs if output.get("status") != "complete"]
    if incomplete_agents:
        objections.append(
            {
                "severity": "review",
                "issue": "incomplete_agents",
                "message": f"Some agents had insufficient evidence: {', '.join(incomplete_agents)}.",
            }
        )
    return {
        "status": "block" if any(item["severity"] == "blocker" for item in objections) else "pass_with_review",
        "blocking_issue_count": sum(1 for item in objections if item["severity"] == "blocker"),
        "review_issue_count": sum(1 for item in objections if item["severity"] == "review"),
        "objections": objections,
    }


def duplicate_agent_output_report(agent_outputs: list[dict[str, Any]]) -> dict[str, Any]:
    signatures = []
    for output in agent_outputs:
        claim_text = " ".join(claim.get("claim", "") for claim in output.get("claims", []))
        signatures.append((output.get("agent_id"), normalize_claim_signature(claim_text)))
    counts = Counter(signature for _agent_id, signature in signatures if signature)
    duplicate_agents = [
        agent_id
        for agent_id, signature in signatures
        if signature and counts[signature] > 1
    ]
    return {
        "agent_count": len(agent_outputs),
        "duplicate_agent_count": len(duplicate_agents),
        "duplicate_agent_ids": duplicate_agents,
        "duplicate_agent_output_rate": round(safe_ratio(len(duplicate_agents), len(agent_outputs)), 4),
    }


def synthesize_brief(
    question: str,
    packet_response: dict[str, Any],
    agent_outputs: list[dict[str, Any]],
    red_team: dict[str, Any],
    duplicate_report: dict[str, Any],
) -> dict[str, Any]:
    supported_claims = [
        claim
        for output in agent_outputs
        for claim in output.get("claims", [])
        if claim.get("evidence_ids")
    ]
    citation_ids = sorted({evidence_id for claim in supported_claims for evidence_id in claim.get("evidence_ids", [])})
    developments = [
        {
            "agent_id": output["agent_id"],
            "agent_name": output["agent_name"],
            "claims": output.get("claims", []),
        }
        for output in agent_outputs
    ]
    return {
        "title": "Weekly Intelligence Brief",
        "question": question,
        "executive_summary": executive_summary(question, supported_claims, red_team),
        "what_changed": summarize_developments(supported_claims),
        "evidence_backed_developments": developments,
        "contradictions_and_uncertainty": contradictions_and_uncertainty(packet_response, red_team),
        "source_gaps": packet_response.get("coverage_gaps", []),
        "next_signals_to_watch": next_signals_to_watch(packet_response),
        "citation_appendix": citation_appendix(packet_response, citation_ids),
        "agent_duplicate_output_rate": duplicate_report["duplicate_agent_output_rate"],
        "human_review_status": "not_reviewed",
    }


def evidence_index(packet_response: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = {}
    for packet in packet_response.get("evidence_packets", []):
        for item in packet.get("supporting_evidence", []):
            evidence_id = str(item.get("evidence_id") or "")
            if evidence_id:
                rows[evidence_id] = item
    return rows


def write_intelligence_report(report: dict[str, Any]) -> None:
    settings = load_settings()
    stem = f"intelligence_brief_{safe_slug(report['mode'] or 'demo')}_{safe_slug(report['question'])}"
    write_json(settings.reports_dir / f"{stem}.json", report)
    write_markdown(settings.reports_dir / f"{stem}.md", render_markdown(report))


def render_markdown(report: dict[str, Any]) -> str:
    brief = report["brief"]
    developments = "\n".join(
        f"- **{row['agent_name']}**: "
        + "; ".join(f"{claim['claim']} [{', '.join(claim['evidence_ids'])}]" for claim in row["claims"])
        for row in brief["evidence_backed_developments"]
    )
    gaps = "\n".join(f"- {gap.get('gap')}" for gap in brief["source_gaps"]) or "- None reported."
    citations = "\n".join(
        f"- `{row['evidence_id']}`: {row.get('title')} ({row.get('source_name')})"
        for row in brief["citation_appendix"]
    )
    objections = "\n".join(
        f"- {item['severity']}: {item['issue']} - {item['message']}"
        for item in report["red_team"]["objections"]
    ) or "- No blocking objections."
    return f"""# {brief['title']}

Status: **{report['status']}**

Question: {brief['question']}

## Executive Summary

{brief['executive_summary']}

## Evidence-Backed Developments

{developments}

## Contradictions And Uncertainty

{brief['contradictions_and_uncertainty']}

## Source Gaps

{gaps}

## Red-Team Review

{objections}

## Citation Appendix

{citations}
"""


def executive_summary(question: str, claims: list[dict[str, Any]], red_team: dict[str, Any]) -> str:
    if red_team["blocking_issue_count"]:
        return "The brief is blocked because at least one factual claim lacks valid packet evidence."
    if not claims:
        return "The packet evidence was insufficient for a grounded intelligence brief."
    return (
        f"For '{question}', the current packet evidence supports a cautious, review-ready brief "
        f"with {len(claims)} grounded claim(s). Human review remains required."
    )


def summarize_developments(claims: list[dict[str, Any]]) -> list[str]:
    return [claim["claim"] for claim in claims[:5]]


def contradictions_and_uncertainty(packet_response: dict[str, Any], red_team: dict[str, Any]) -> str:
    pieces = []
    for packet in packet_response.get("evidence_packets", []):
        if packet.get("conflicting_evidence"):
            pieces.append("The evidence packet contains conflict notes that require analyst review.")
        if packet.get("limitations"):
            pieces.extend(str(limit) for limit in packet["limitations"][:2])
    for objection in red_team.get("objections", []):
        if objection["severity"] == "review":
            pieces.append(objection["message"])
    return " ".join(pieces) if pieces else "No explicit contradiction was detected in the packet, but coverage remains provisional."


def next_signals_to_watch(packet_response: dict[str, Any]) -> list[str]:
    gaps = packet_response.get("coverage_gaps", [])
    signals = [gap.get("suggested_next_query") for gap in gaps if gap.get("suggested_next_query")]
    return signals[:5] or ["New official filings, local reporting, public hearings, and company infrastructure announcements."]


def citation_appendix(packet_response: dict[str, Any], citation_ids: list[str]) -> list[dict[str, Any]]:
    evidence = evidence_index(packet_response)
    return [evidence[citation_id] for citation_id in citation_ids if citation_id in evidence]


def agent_has_matching_evidence(agent: AgentSpec, evidence: list[dict[str, Any]]) -> bool:
    return any(any(keyword in evidence_haystack(item) for keyword in agent.issue_keywords) for item in evidence)


def evidence_haystack(item: dict[str, Any]) -> str:
    values = [
        item.get("title"),
        item.get("text"),
        item.get("source_name"),
        item.get("source_type"),
        item.get("provenance"),
        item.get("domain"),
        item.get("jurisdiction"),
    ]
    return " ".join(clean_text(value).lower() for value in values if value)


def first_matching_keyword(keywords: tuple[str, ...], text: str) -> str | None:
    haystack = text.lower()
    for keyword in keywords:
        if keyword in haystack:
            return keyword
    return None


def item_confidence(item: dict[str, Any]) -> str:
    if item.get("provenance") in {"official", "policy"}:
        return "medium"
    if item.get("provenance") == "public_opinion":
        return "low"
    return "low"


def summarize_text(text: str, max_chars: int = 220) -> str:
    cleaned = clean_text(text)
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[:max_chars].rstrip() + "..."


def clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def normalize_claim_signature(text: str) -> str:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    stop = {"the", "and", "from", "with", "agent", "based", "flags", "on"}
    return " ".join(token for token in tokens if token not in stop)[:160]


def safe_ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def safe_slug(text: str, limit: int = 60) -> str:
    slug = "".join(character if character.isalnum() else "-" for character in text.lower())[:limit].strip("-")
    return slug or "report"


def stable_request_id(text: str) -> str:
    return f"req_{safe_slug(text, limit=24)}"


def agent_to_dict(agent: AgentSpec) -> dict[str, Any]:
    return {
        "agent_id": agent.agent_id,
        "name": agent.name,
        "role": agent.role,
        "issue_keywords": list(agent.issue_keywords),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a deterministic evidence-grounded intelligence brief.")
    parser.add_argument("question")
    parser.add_argument("--mode", default="ai_infra_geo_risk_demo")
    parser.add_argument("--policy", default="rag")
    parser.add_argument("--project-id", default="ai_infra_geo_risk")
    parser.add_argument("--no-write-report", action="store_true")
    args = parser.parse_args()
    report = run_intelligence_brief(
        question=args.question,
        mode=args.mode,
        policy=args.policy,
        project_id=args.project_id,
        write_report=not args.no_write_report,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
