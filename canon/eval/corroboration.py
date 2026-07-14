from __future__ import annotations

import argparse
import json

from canon.config import load_settings
from canon.eval.contracts import CONTRACT_VERSION, validate_report
from canon.evidence.corroboration import assess_corroboration


def adversarial_corroboration_fixture() -> list[dict]:
    return [
        evidence_item(
            "C1",
            "w1",
            "Journal of Market Structure",
            "Independent Event Study",
            "The merger reduced prices in treated markets after the policy change.",
            "supportive",
            source_chain_id="independent-study-1",
        ),
        evidence_item(
            "C2",
            "w2",
            "Regulator Archive",
            "Regulatory Review",
            "The merger reduced prices in treated markets after the policy change.",
            "supportive",
            source_chain_id="independent-regulator-1",
        ),
        evidence_item(
            "C3",
            "w3",
            "Company Blog",
            "Market Update",
            "According to the company press release, the merger reduced prices in treated markets.",
            "supportive",
            source_chain_id="company-pr-2026-01",
            provenance="web",
        ),
        evidence_item(
            "C4",
            "w4",
            "Syndicated News",
            "Market Update Reprint",
            "Reprinted from the company press release: the merger reduced prices in treated markets.",
            "supportive",
            source_chain_id="company-pr-2026-01",
            provenance="web",
        ),
        evidence_item(
            "C5",
            "w5",
            "Competitor Filing",
            "Risk Disclosure",
            "The merger did not reduce prices in treated markets and may have increased switching costs.",
            "critical",
            source_chain_id="competitor-filing-1",
        ),
    ]


def echo_attack_fixture() -> list[dict]:
    return [
        evidence_item(
            f"C{index}",
            f"w{index}",
            source,
            f"Echo Report {index}",
            "According to the same source, the product has no churn risk in enterprise accounts.",
            "supportive",
            source_chain_id="vendor-pr-echo",
            provenance="web",
        )
        for index, source in enumerate(
            ["Vendor Blog", "Partner Blog", "Syndicated News", "SEO Review Site"],
            start=1,
        )
    ]


def evaluate_adversarial_corroboration() -> dict:
    mixed = assess_corroboration(adversarial_corroboration_fixture())
    echo = assess_corroboration(echo_attack_fixture())
    report = {
        "contract_version": CONTRACT_VERSION,
        "benchmark_id": "adversarial_corroboration_v1",
        "cases": {
            "mixed_independent_and_echo": mixed,
            "pure_echo_attack": echo,
        },
        "checks": {
            "mixed_has_independent_support": mixed["summary"]["max_independent_support"] >= 2,
            "mixed_has_conflict": mixed["summary"]["total_contradictions"] >= 1,
            "echo_attack_flagged": echo["summary"]["overall_verdict"] == "adversarial_echo_risk",
            "echo_attack_not_independent": echo["summary"]["max_independent_support"] <= 1,
        },
    }
    report["status"] = "pass" if all(report["checks"].values()) else "fail"
    report["contract_validation"] = validate_report(report, "adversarial_corroboration")
    return report


def evidence_item(
    citation_id: str,
    work_id: str,
    source_name: str,
    title: str,
    claim_text: str,
    stance: str,
    source_chain_id: str,
    provenance: str = "academic",
) -> dict:
    return {
        "citation_id": citation_id,
        "chunk_id": f"chunk:{work_id}",
        "work_id": work_id,
        "title": title,
        "source_name": source_name,
        "source_chain_id": source_chain_id,
        "provenance": provenance,
        "claim": {
            "id": f"claim:{work_id}",
            "text": claim_text,
            "stance": stance,
            "confidence": 0.9,
        },
        "preview": claim_text,
    }


def write_report() -> dict:
    settings = load_settings()
    report = evaluate_adversarial_corroboration()
    path = settings.reports_dir / "adversarial_corroboration_v1.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate adversarial corroboration handling.")
    parser.add_argument("--write-report", action="store_true")
    args = parser.parse_args()
    report = write_report() if args.write_report else evaluate_adversarial_corroboration()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
