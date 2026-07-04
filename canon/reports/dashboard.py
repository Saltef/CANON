from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

from canon.config import load_settings


def build_dashboard(mode: str, method_set_id: str = "baseline_methods_v1") -> dict:
    settings = load_settings()
    artifacts = load_artifacts(settings.reports_dir, mode, method_set_id)
    output_path = settings.reports_dir / f"dashboard_{mode}_{method_set_id}.html"
    output_path.write_text(render_dashboard(mode, method_set_id, artifacts), encoding="utf-8")
    report = {
        "mode": mode,
        "method_set_id": method_set_id,
        "output_path": str(output_path),
        "artifact_keys": sorted(artifacts),
    }
    write_json(settings.reports_dir / f"dashboard_{mode}_{method_set_id}.json", report)
    return report


def load_artifacts(reports_dir: Path, mode: str, method_set_id: str) -> dict:
    names = {
        "method_eval": f"method_eval_{mode}_{method_set_id}.json",
        "evaluation_suite": f"evaluation_suite_{mode}_{method_set_id}.json",
        "batch_eval": f"batch_eval_{mode}_{method_set_id}.json",
        "stress_slices": f"stress_slices_{mode}_{method_set_id}.json",
        "probe_eval": f"probe_eval_{mode}_{method_set_id}.json",
        "weight_tuning": f"weight_tuning_{mode}_{method_set_id}.json",
    }
    artifacts = {}
    for key, name in names.items():
        path = reports_dir / name
        if path.exists():
            artifacts[key] = json.loads(path.read_text(encoding="utf-8"))
    return artifacts


def render_dashboard(mode: str, method_set_id: str, artifacts: dict) -> str:
    sections = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>CANON Evaluation Dashboard</title>",
        "<style>body{font-family:Arial,sans-serif;margin:32px;line-height:1.4}table{border-collapse:collapse;margin:12px 0;width:100%}td,th{border:1px solid #ccc;padding:6px;text-align:left}code{background:#eee;padding:2px 4px}</style>",
        "</head><body>",
        f"<h1>CANON Dashboard: {html.escape(mode)}</h1>",
        f"<p>Method set: <code>{html.escape(method_set_id)}</code></p>",
    ]
    probe = artifacts.get("probe_eval")
    if probe:
        sections.append("<h2>Probe Leaderboard</h2>")
        sections.append(table(["rank", "method_id", "average_probe_score", "expectation_pass_rate"], probe["probe_summary"]["leaderboard"]))
        sections.append("<h2>Probe Failures</h2>")
        sections.append(table(["probe_id", "category", "method_id", "failed_expectations"], probe["failure_summary"]["failures"]))
    suite = artifacts.get("evaluation_suite")
    if suite:
        sections.append("<h2>Method Leaderboard</h2>")
        sections.append(table(["rank", "method_id", "composite_score", "context_relevance", "conflict_awareness"], suite["leaderboard"]))
    batches = artifacts.get("batch_eval")
    if batches:
        sections.append("<h2>Batch Stability</h2>")
        sections.append(f"<pre>{html.escape(json.dumps(batches['stability'], indent=2))}</pre>")
    sections.append("</body></html>")
    return "\n".join(sections)


def table(columns: list[str], rows: list[dict]) -> str:
    header = "".join(f"<th>{html.escape(column)}</th>" for column in columns)
    body = []
    for row in rows:
        cells = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, list):
                value = ", ".join(str(item) for item in value)
            cells.append(f"<td>{html.escape(str(value))}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return "<table><thead><tr>" + header + "</tr></thead><tbody>" + "".join(body) + "</tbody></table>"


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a static CANON evaluation dashboard.")
    parser.add_argument("--mode", default="social_science_ir_v1_harvest10")
    parser.add_argument("--method-set-id", default="baseline_methods_v1")
    args = parser.parse_args()
    print(json.dumps(build_dashboard(args.mode, args.method_set_id), indent=2))


if __name__ == "__main__":
    main()
