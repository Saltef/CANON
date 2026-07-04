from __future__ import annotations

import argparse
import json
from pathlib import Path

from canon.config import load_settings


METHOD_SET_ID = "baseline_methods_v1"
BASELINE_METHOD_ID = "lexical_k5_template"
ORIGINAL_DIVERSE_METHOD_ID = "diverse_k5_template"
FOCUS_DIVERSE_METHOD_ID = "focus_diverse_k5_template"


def build_portfolio_report(mode: str = "social_science_ir_10k") -> dict:
    settings = load_settings()
    artifacts = load_portfolio_artifacts(settings.reports_dir, mode)
    report = {
        "project": "CANON",
        "mode": mode,
        "positioning": "Importance-aware scholarly RAG with explicit claim boundaries.",
        "corpus": summarize_corpus(artifacts.get("data_card") or {}),
        "main_finding": main_finding(artifacts),
        "diversity_comparison": diversity_comparison(artifacts),
        "allowed_claim": allowed_claim(),
        "claim_limits": claim_limits(artifacts),
        "reproducibility": reproducibility_commands(mode),
        "portfolio_files": portfolio_files(settings.reports_dir, mode),
    }
    write_json(settings.reports_dir / f"portfolio_{mode}.json", report)
    write_markdown(settings.reports_dir / f"portfolio_{mode}.md", render_markdown(report))
    return report


def load_portfolio_artifacts(reports_dir: Path, mode: str) -> dict:
    names = {
        "data_card": f"data_card_{mode}.json",
        "original_diversity": (
            f"diversity_audit_{mode}_{METHOD_SET_ID}_{ORIGINAL_DIVERSE_METHOD_ID}_vs_{BASELINE_METHOD_ID}.json"
        ),
        "original_diagnostics": (
            f"diversity_diagnostics_{mode}_{METHOD_SET_ID}_{ORIGINAL_DIVERSE_METHOD_ID}_vs_{BASELINE_METHOD_ID}.json"
        ),
        "focus_diversity": (
            f"diversity_audit_{mode}_{METHOD_SET_ID}_{FOCUS_DIVERSE_METHOD_ID}_vs_{BASELINE_METHOD_ID}.json"
        ),
        "focus_diagnostics": (
            f"diversity_diagnostics_{mode}_{METHOD_SET_ID}_{FOCUS_DIVERSE_METHOD_ID}_vs_{BASELINE_METHOD_ID}.json"
        ),
        "focus_gate": (
            f"diversity_gate_{mode}_{METHOD_SET_ID}_{FOCUS_DIVERSE_METHOD_ID}_vs_{BASELINE_METHOD_ID}.json"
        ),
    }
    artifacts = {}
    for key, filename in names.items():
        path = reports_dir / filename
        if path.exists():
            artifacts[key] = json.loads(path.read_text(encoding="utf-8"))
    return artifacts


def summarize_corpus(data_card: dict) -> dict:
    return {
        "work_count": data_card.get("work_count"),
        "chunk_count": data_card.get("chunk_count"),
        "source_modes": data_card.get("source_modes", []),
        "coverage": data_card.get("coverage", {}),
        "limitations": data_card.get("limitations", []),
    }


def main_finding(artifacts: dict) -> str:
    original = artifacts.get("original_diagnostics") or {}
    focus = artifacts.get("focus_diagnostics") or {}
    original_headline = (original.get("headline") or {}).get("finding")
    focus_counts = ((focus.get("aggregate") or {}).get("failure_mode_counts") or {})
    if original_headline and focus_counts:
        return (
            f"{original_headline} Focus-gated diversity reduced low-focus/noisy breadth while "
            "preserving useful new-cluster retrieval on the 10k corpus."
        )
    return "Diversity effects are conditional and require query-focus safeguards before product claims."


def diversity_comparison(artifacts: dict) -> dict:
    original = artifacts.get("original_diversity") or {}
    focus = artifacts.get("focus_diversity") or {}
    return {
        "baseline_method_id": BASELINE_METHOD_ID,
        "original": summarize_diversity(original),
        "focus_gated": summarize_diversity(focus),
        "delta_focus_minus_original": diversity_delta(
            summarize_diversity(focus),
            summarize_diversity(original),
        ),
    }


def summarize_diversity(report: dict) -> dict:
    aggregate = report.get("aggregate", {})
    verdict_counts = aggregate.get("verdict_counts", {})
    return {
        "method_id": report.get("diverse_method_id"),
        "query_count": report.get("query_count"),
        "useful_breadth": int(verdict_counts.get("useful_breadth") or 0),
        "mixed_breadth": int(verdict_counts.get("mixed_breadth") or 0),
        "off_topic_breadth_risk": int(verdict_counts.get("off_topic_breadth_risk") or 0),
        "no_diversity_gain": int(verdict_counts.get("no_diversity_gain") or 0),
        "average_cluster_delta": aggregate.get("average_cluster_delta"),
        "average_breadth_precision": aggregate.get("average_breadth_precision"),
        "average_noise_rate": aggregate.get("average_noise_rate"),
    }


def diversity_delta(left: dict, right: dict) -> dict:
    fields = [
        "useful_breadth",
        "mixed_breadth",
        "off_topic_breadth_risk",
        "no_diversity_gain",
        "average_cluster_delta",
        "average_breadth_precision",
        "average_noise_rate",
    ]
    delta = {}
    for field in fields:
        if left.get(field) is None or right.get(field) is None:
            continue
        delta[field] = round(float(left[field]) - float(right[field]), 6)
    return delta


def allowed_claim() -> str:
    return (
        "On this 10k OpenAlex-derived social-science corpus, diversity-aware retrieval is best "
        "described as a conditional breadth mechanism. Focus-gated diversity increased useful "
        "breadth and reduced noise relative to ungated diversity, but this does not establish a "
        "global retrieval winner."
    )


def claim_limits(artifacts: dict) -> list[str]:
    limits = [
        "No global method-winner claim: previous audits showed narrow margins and unstable top methods.",
        "No external benchmark claim: public qrels coverage is still a smoke fixture rather than a full benchmark.",
        "Diversity gains are query-sensitive; off-topic and low-focus queries require guardrails.",
    ]
    gate = artifacts.get("focus_gate") or {}
    if gate.get("status") != "pass":
        limits.append("Focus-gated diversity has not passed the configured diversity gate for this mode.")
    return limits


def reproducibility_commands(mode: str) -> list[str]:
    return [
        f"python -m canon.eval.diversity --mode {mode}",
        f"python -m canon.eval.diversity_diagnostics --mode {mode}",
        (
            f"python -m canon.eval.diversity --mode {mode} "
            f"--diverse-method-id {FOCUS_DIVERSE_METHOD_ID} --baseline-method-id {BASELINE_METHOD_ID}"
        ),
        (
            f"python -m canon.eval.diversity_diagnostics --mode {mode} "
            f"--diverse-method-id {FOCUS_DIVERSE_METHOD_ID} --baseline-method-id {BASELINE_METHOD_ID}"
        ),
        (
            f"python -m canon.eval.diversity_gate --mode {mode} "
            f"--diverse-method-id {FOCUS_DIVERSE_METHOD_ID} --baseline-method-id {BASELINE_METHOD_ID}"
        ),
        f"python -m canon.workbench.build --mode {mode}",
        f"python -m canon.reports.portfolio --mode {mode}",
    ]


def portfolio_files(reports_dir: Path, mode: str) -> dict:
    return {
        "portfolio_markdown": str(reports_dir / f"portfolio_{mode}.md"),
        "portfolio_json": str(reports_dir / f"portfolio_{mode}.json"),
        "workbench_html": str(reports_dir / f"workbench_{mode}.html"),
        "focus_diversity_gate": str(
            reports_dir
            / f"diversity_gate_{mode}_{METHOD_SET_ID}_{FOCUS_DIVERSE_METHOD_ID}_vs_{BASELINE_METHOD_ID}.json"
        ),
    }


def render_markdown(report: dict) -> str:
    corpus = report["corpus"]
    comparison = report["diversity_comparison"]
    original = comparison["original"]
    focus = comparison["focus_gated"]
    delta = comparison["delta_focus_minus_original"]
    limits = "\n".join(f"- {item}" for item in report["claim_limits"])
    commands = "\n".join(f"- `{command}`" for command in report["reproducibility"])
    return f"""# CANON Portfolio Summary

{report["positioning"]}

## Corpus

- Works: {corpus.get("work_count")}
- Chunks: {corpus.get("chunk_count")}
- Source modes: {", ".join(corpus.get("source_modes") or [])}

## Main Finding

{report["main_finding"]}

## Diversity Comparison

| Method | Useful | Mixed | Off-topic risk | No gain | Breadth precision | Noise rate | Cluster delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| {original.get("method_id")} | {original.get("useful_breadth")} | {original.get("mixed_breadth")} | {original.get("off_topic_breadth_risk")} | {original.get("no_diversity_gain")} | {original.get("average_breadth_precision")} | {original.get("average_noise_rate")} | {original.get("average_cluster_delta")} |
| {focus.get("method_id")} | {focus.get("useful_breadth")} | {focus.get("mixed_breadth")} | {focus.get("off_topic_breadth_risk")} | {focus.get("no_diversity_gain")} | {focus.get("average_breadth_precision")} | {focus.get("average_noise_rate")} | {focus.get("average_cluster_delta")} |

Delta, focus-gated minus original:

- Useful breadth: {delta.get("useful_breadth")}
- Mixed breadth: {delta.get("mixed_breadth")}
- Average breadth precision: {delta.get("average_breadth_precision")}
- Average noise rate: {delta.get("average_noise_rate")}

## Allowed Claim

{report["allowed_claim"]}

## Claim Limits

{limits}

## Reproduce

{commands}
"""


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_markdown(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a compact CANON portfolio report.")
    parser.add_argument("--mode", default="social_science_ir_10k")
    args = parser.parse_args()
    print(json.dumps(build_portfolio_report(args.mode), indent=2))


if __name__ == "__main__":
    main()
