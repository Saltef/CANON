from __future__ import annotations

import argparse
import json
import re
from typing import Any

from canon.config import load_settings
from canon.intelligence.evidence_runner import run_intelligence_brief
from canon.product.report_io import write_json, write_markdown


REPORT_ID = "evidence_alert_digest_v1"


def run_alert_digest(
    question: str,
    mode: str,
    policy: str = "rag",
    project_id: str = "ai_infra_geo_risk",
    write_report: bool = True,
) -> dict[str, Any]:
    brief_report = run_intelligence_brief(
        question=question,
        mode=mode,
        policy=policy,
        project_id=project_id,
        write_report=False,
    )
    digest = build_alert_digest(brief_report)
    if write_report:
        write_alert_digest(digest)
    return digest


def build_alert_digest(brief_report: dict[str, Any]) -> dict[str, Any]:
    alerts = alert_rows(brief_report)
    duplicate = duplicate_alert_report(alerts)
    quality = alert_quality_report(alerts, duplicate)
    return {
        "report_id": REPORT_ID,
        "project_id": brief_report.get("project_id"),
        "mode": brief_report.get("mode"),
        "policy": brief_report.get("policy"),
        "question": brief_report.get("question"),
        "status": "ready_for_human_review" if quality["status"] == "pass" else "review",
        "alert_count": len(alerts),
        "alerts": alerts,
        "duplicate_alerts": duplicate,
        "quality": quality,
        "human_review_status": "not_reviewed",
        "claim_boundaries": [
            "Alerts are evidence-triggered review prompts, not autonomous notifications.",
            "Every alert must carry packet evidence IDs or explicitly remain a source-gap alert.",
            "Human review is required before operational use.",
        ],
    }


def alert_rows(brief_report: dict[str, Any]) -> list[dict[str, Any]]:
    brief = brief_report.get("brief") or {}
    evidence = {str(item.get("evidence_id")): item for item in brief.get("citation_appendix") or [] if item.get("evidence_id")}
    alerts = []
    for claim in supported_claims(brief_report):
        evidence_ids = [str(evidence_id) for evidence_id in claim.get("evidence_ids") or []]
        trigger = first_evidence(evidence, evidence_ids)
        alerts.append(
            {
                "alert_id": f"alert_{len(alerts) + 1:03d}",
                "severity": severity_for_claim(claim),
                "confidence": claim.get("confidence") or "low",
                "issue": first_value(claim.get("issue_categories")) or "evidence_signal",
                "affected_regions": infer_regions([trigger, claim]),
                "affected_entities": infer_entities([trigger, claim]),
                "evidence_trigger": summarize_trigger(trigger, claim),
                "evidence_ids": evidence_ids,
                "uncertainty": uncertainty_from_brief(brief_report),
                "recommended_follow_up": follow_up_for_claim(claim, brief),
            }
        )
    for gap in brief.get("source_gaps") or []:
        if len(alerts) >= 12:
            break
        alerts.append(
            {
                "alert_id": f"alert_{len(alerts) + 1:03d}",
                "severity": "medium",
                "confidence": "low",
                "issue": "source_gap",
                "affected_regions": infer_regions([gap]),
                "affected_entities": [],
                "evidence_trigger": str(gap.get("gap") or "Source gap reported by evidence packet."),
                "evidence_ids": [],
                "uncertainty": "Coverage gap; requires follow-up evidence collection.",
                "recommended_follow_up": gap.get("suggested_next_query") or first_value(brief.get("next_signals_to_watch")) or "",
            }
        )
    return dedupe_alerts(alerts)


def supported_claims(brief_report: dict[str, Any]) -> list[dict[str, Any]]:
    claims = []
    for output in brief_report.get("agent_outputs") or []:
        for claim in output.get("claims") or []:
            if claim.get("evidence_ids"):
                claims.append(claim)
    return claims


def first_evidence(evidence: dict[str, dict[str, Any]], evidence_ids: list[str]) -> dict[str, Any]:
    for evidence_id in evidence_ids:
        if evidence_id in evidence:
            return evidence[evidence_id]
    return {}


def severity_for_claim(claim: dict[str, Any]) -> str:
    issue = " ".join(str(value).lower() for value in claim.get("issue_categories") or [])
    text = str(claim.get("claim") or "").lower()
    high_terms = {"grid", "water", "electricity", "sovereign", "foreign", "permit"}
    if any(term in issue or term in text for term in high_terms):
        return "high"
    return "medium"


def infer_regions(items: list[dict[str, Any]]) -> list[str]:
    known = ["Latin America", "Brazil", "Chile", "Mexico"]
    text = normalized_blob(items)
    regions = [region for region in known if region.lower() in text]
    return regions or ([str(items[0].get("jurisdiction"))] if items and items[0].get("jurisdiction") else [])


def infer_entities(items: list[dict[str, Any]]) -> list[str]:
    known = ["ExampleCloud", "Chile Energy Regulator", "Regional Digital Policy Institute", "Municipal Hearing Sample"]
    text = normalized_blob(items)
    return [entity for entity in known if entity.lower() in text]


def normalized_blob(items: list[dict[str, Any]]) -> str:
    values = []
    for item in items:
        for key in ["claim", "title", "source_name", "jurisdiction", "evidence_trigger", "gap"]:
            if item.get(key):
                values.append(str(item[key]))
    return " ".join(values).lower()


def summarize_trigger(trigger: dict[str, Any], claim: dict[str, Any]) -> str:
    source = trigger.get("source_name") or trigger.get("title") or "packet evidence"
    text = trigger.get("text") or claim.get("claim") or ""
    return f"{source}: {summarize_text(text)}"


def uncertainty_from_brief(brief_report: dict[str, Any]) -> str:
    red_team = brief_report.get("red_team") or {}
    if red_team.get("review_issue_count"):
        return "Review issue present: " + "; ".join(item.get("issue", "review") for item in red_team.get("objections") or [])
    return "No red-team blocker; alert remains provisional until human review."


def follow_up_for_claim(claim: dict[str, Any], brief: dict[str, Any]) -> str:
    issue = first_value(claim.get("issue_categories")) or "evidence"
    for signal in brief.get("next_signals_to_watch") or []:
        if issue.lower() in str(signal).lower():
            return str(signal)
    return f"Collect newer independent evidence for {issue} and verify against official/local sources."


def dedupe_alerts(alerts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    rows = []
    for alert in alerts:
        signature = alert_signature(alert)
        if signature in seen:
            continue
        seen.add(signature)
        alert = dict(alert)
        alert["alert_id"] = f"alert_{len(rows) + 1:03d}"
        rows.append(alert)
    return rows


def duplicate_alert_report(alerts: list[dict[str, Any]]) -> dict[str, Any]:
    signatures = [alert_signature(alert) for alert in alerts]
    duplicate_count = len(signatures) - len(set(signatures))
    return {
        "alert_count": len(alerts),
        "duplicate_alert_count": duplicate_count,
        "duplicate_alert_rate": round(safe_ratio(duplicate_count, len(alerts)), 4),
    }


def alert_quality_report(alerts: list[dict[str, Any]], duplicate: dict[str, Any]) -> dict[str, Any]:
    checks = [
        check("alerts_present", len(alerts) > 0),
        check("duplicate_alert_rate", float(duplicate.get("duplicate_alert_rate") or 0.0) <= 0.10),
        check("all_alerts_have_trigger", all(bool(alert.get("evidence_trigger")) for alert in alerts)),
        check("all_alerts_have_issue", all(bool(alert.get("issue")) for alert in alerts)),
        check("all_alerts_have_uncertainty", all(bool(alert.get("uncertainty")) for alert in alerts)),
        check("all_alerts_have_follow_up", all(bool(alert.get("recommended_follow_up")) for alert in alerts)),
        check(
            "evidence_alerts_have_ids",
            all(alert.get("evidence_ids") or alert.get("issue") == "source_gap" for alert in alerts),
        ),
    ]
    return {
        "status": "pass" if all(item["passed"] for item in checks) else "fail",
        "checks": checks,
        "limitations": [
            "This gate checks alert structure and duplicate rate only.",
            "Human review is required for alert usefulness and severity calibration.",
        ],
    }


def alert_signature(alert: dict[str, Any]) -> str:
    parts = [
        str(alert.get("issue") or "").lower(),
        ",".join(sorted(str(region).lower() for region in alert.get("affected_regions") or [])),
        ",".join(sorted(str(evidence_id).lower() for evidence_id in alert.get("evidence_ids") or [])),
    ]
    return "|".join(parts)


def check(identifier: str, passed: bool) -> dict[str, Any]:
    return {"id": identifier, "passed": bool(passed)}


def first_value(values: Any) -> str:
    if isinstance(values, list) and values:
        return str(values[0])
    if values:
        return str(values)
    return ""


def summarize_text(text: Any, max_chars: int = 220) -> str:
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[:max_chars].rstrip() + "..."


def safe_ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def write_alert_digest(digest: dict[str, Any]) -> None:
    settings = load_settings()
    stem = f"alert_digest_{safe_slug(digest.get('mode') or 'demo')}_{safe_slug(digest.get('question') or 'question')}"
    write_json(settings.reports_dir / f"{stem}.json", digest)
    write_markdown(settings.reports_dir / f"{stem}.md", render_markdown(digest))


def render_markdown(digest: dict[str, Any]) -> str:
    alerts = "\n".join(
        f"- `{alert['alert_id']}` {alert['severity']} {alert['issue']}: "
        f"{alert['evidence_trigger']} (evidence: {', '.join(alert.get('evidence_ids') or ['source_gap'])})"
        for alert in digest["alerts"]
    )
    return f"""# Alert Digest

Status: **{digest['status']}**

Question: {digest['question']}

Alerts: {digest['alert_count']}

Duplicate alert rate: {digest['duplicate_alerts']['duplicate_alert_rate']}

## Alerts

{alerts}
"""


def safe_slug(text: Any, limit: int = 60) -> str:
    slug = "".join(character if character.isalnum() else "-" for character in str(text).lower())[:limit].strip("-")
    return slug or "alert"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an evidence-grounded alert digest.")
    parser.add_argument("question")
    parser.add_argument("--mode", default="ai_infra_geo_risk_demo")
    parser.add_argument("--policy", default="rag")
    parser.add_argument("--project-id", default="ai_infra_geo_risk")
    parser.add_argument("--no-write-report", action="store_true")
    args = parser.parse_args()
    digest = run_alert_digest(
        question=args.question,
        mode=args.mode,
        policy=args.policy,
        project_id=args.project_id,
        write_report=not args.no_write_report,
    )
    print(json.dumps(digest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
