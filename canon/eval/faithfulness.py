from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from canon.config import load_settings
from canon.eval.harness import load_gold
from canon.eval.methods import load_method_set, method_specs
from canon.retrieval.tokenize import tokenize
from canon.synthesis.answer import synthesize


CITATION_PATTERN = re.compile(r"\bC\d+\b")


def evaluate_faithfulness(
    mode: str,
    method_set_path: Path | None = None,
    query_limit: int = 5,
) -> dict:
    settings = load_settings()
    method_set = load_method_set(method_set_path)
    specs = method_specs(method_set)
    queries = load_gold(settings.root / "gold" / "seed_queries.json")[:query_limit]
    query_reports = []
    for query in queries:
        runs = []
        for spec in specs:
            synthesis = synthesize_for_faithfulness(query["query"], spec, mode)
            runs.append(score_synthesis(query, spec.id, spec.label, synthesis))
        query_reports.append({"id": query["id"], "query": query["query"], "method_runs": runs})
    report = {
        "mode": mode,
        "method_set_id": method_set["method_set_id"],
        "query_count": len(query_reports),
        "queries": query_reports,
        "method_summary": method_summary(query_reports),
        "failure_summary": failure_summary(query_reports),
    }
    report["leaderboard"] = leaderboard(report["method_summary"])
    write_json(settings.reports_dir / f"faithfulness_{mode}_{method_set['method_set_id']}.json", report)
    return report


def synthesize_for_faithfulness(query: str, spec, mode: str) -> dict:
    return synthesize(
        query=query,
        policy=spec.policy,
        mode=mode,
        top_k=spec.top_k,
        generator_name=spec.generator,
        generator_model=spec.generator_model,
    )


def score_synthesis(query: dict, method_id: str, label: str, synthesis: dict) -> dict:
    answer = synthesis.get("answer", "")
    citations = {citation["citation_id"]: citation for citation in synthesis.get("citations", [])}
    evidence = {item["citation_id"]: item for item in synthesis.get("evidence", [])}
    sentence_reports = []
    for sentence in split_sentences(answer):
        cited_ids = CITATION_PATTERN.findall(sentence)
        sentence_reports.append(score_sentence(sentence, cited_ids, citations, evidence))
    if not sentence_reports:
        sentence_reports = [score_sentence(answer, [], citations, evidence)]
    metrics = {
        "citation_validity": average(row["citation_validity"] for row in sentence_reports),
        "evidence_token_support": average(row["evidence_token_support"] for row in sentence_reports),
        "claim_backing_rate": average(row["claim_backed"] for row in sentence_reports),
        "unsupported_sentence_rate": average(1.0 - row["claim_backed"] for row in sentence_reports),
    }
    metrics["faithfulness_score"] = round(
        (metrics["citation_validity"] * 0.35)
        + (metrics["evidence_token_support"] * 0.35)
        + (metrics["claim_backing_rate"] * 0.3),
        6,
    )
    return {
        "method_id": method_id,
        "label": label,
        "query_id": query["id"],
        "metrics": metrics,
        "sentence_reports": sentence_reports,
    }


def score_sentence(sentence: str, cited_ids: list[str], citations: dict, evidence: dict) -> dict:
    valid_ids = [citation_id for citation_id in cited_ids if citation_id in citations and citation_id in evidence]
    sentence_tokens = content_tokens(sentence)
    evidence_tokens = set()
    attached_claim = False
    for citation_id in valid_ids:
        item = evidence[citation_id]
        evidence_tokens.update(content_tokens(item.get("preview") or ""))
        evidence_tokens.update(content_tokens(item.get("title") or ""))
        claim = item.get("claim") or {}
        if claim:
            attached_claim = True
            evidence_tokens.update(content_tokens(claim.get("text") or ""))
    support = len(sentence_tokens & evidence_tokens) / len(sentence_tokens) if sentence_tokens else 0.0
    if not cited_ids and not sentence_tokens:
        support = 1.0
    return {
        "sentence": sentence,
        "cited_ids": cited_ids,
        "valid_cited_ids": valid_ids,
        "citation_validity": round(len(valid_ids) / len(cited_ids), 6) if cited_ids else 0.0,
        "evidence_token_support": round(support, 6),
        "claim_backed": 1.0 if valid_ids and (support >= 0.12 or attached_claim) else 0.0,
    }


def split_sentences(text: str) -> list[str]:
    return [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", text) if sentence.strip()]


def content_tokens(text: str) -> set[str]:
    stop = {"across", "answer", "claim", "evidence", "grounded", "literature", "reports", "retrieved", "strongest", "that", "this", "through", "with"}
    return {token for token in tokenize(text) if len(token) >= 4 and token not in stop}


def method_summary(query_reports: list[dict]) -> dict:
    method_ids = sorted({run["method_id"] for query in query_reports for run in query["method_runs"]})
    summary = {}
    for method_id in method_ids:
        runs = [run for query in query_reports for run in query["method_runs"] if run["method_id"] == method_id]
        summary[method_id] = {
            "label": runs[0]["label"] if runs else method_id,
            "query_count": len(runs),
            "average_metrics": {
                metric: average(run["metrics"][metric] for run in runs)
                for metric in ["citation_validity", "evidence_token_support", "claim_backing_rate", "unsupported_sentence_rate", "faithfulness_score"]
            },
        }
    return summary


def leaderboard(summary: dict) -> list[dict]:
    return [
        {"rank": index, "method_id": method_id, **row["average_metrics"]}
        for index, (method_id, row) in enumerate(
            sorted(summary.items(), key=lambda item: item[1]["average_metrics"]["faithfulness_score"], reverse=True),
            start=1,
        )
    ]


def failure_summary(query_reports: list[dict]) -> dict:
    failures = []
    for query in query_reports:
        for run in query["method_runs"]:
            if run["metrics"]["unsupported_sentence_rate"] > 0.5:
                failures.append({"query_id": query["id"], "method_id": run["method_id"], "unsupported_sentence_rate": run["metrics"]["unsupported_sentence_rate"]})
    return {"failure_count": len(failures), "failures": failures}


def average(values) -> float:
    values = [float(value) for value in values]
    if not values:
        return 0.0
    return round(sum(values) / len(values), 6)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate generated answer faithfulness against cited evidence.")
    parser.add_argument("--mode", default="social_science_ir_v1_harvest10")
    parser.add_argument("--method-set", default=None)
    parser.add_argument("--query-limit", type=int, default=5)
    args = parser.parse_args()
    print(json.dumps(evaluate_faithfulness(args.mode, Path(args.method_set) if args.method_set else None, args.query_limit), indent=2))


if __name__ == "__main__":
    main()
