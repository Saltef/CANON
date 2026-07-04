from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

from canon.config import load_settings


def build_workbench(mode: str = "live") -> dict:
    settings = load_settings()
    report_files = {
        "rag_eval": settings.reports_dir / f"rag_eval_{mode}.json",
        "claims": settings.reports_dir / f"claims_{mode}.json",
        "conflicts": settings.reports_dir / f"conflicts_{mode}.json",
        "quality": settings.reports_dir / f"quality_diagnostics_{mode}.json",
        "diversity": settings.reports_dir
        / f"diversity_audit_{mode}_baseline_methods_v1_diverse_k5_template_vs_lexical_k5_template.json",
    }
    loaded = {name: load_json(path) for name, path in report_files.items()}
    html_text = render_workbench(mode, loaded)
    output_path = settings.reports_dir / f"workbench_{mode}.html"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_text, encoding="utf-8")
    report = {
        "mode": mode,
        "output_path": str(output_path),
        "available_reports": {
            name: payload is not None for name, payload in loaded.items()
        },
    }
    write_json(settings.reports_dir / f"workbench_{mode}.json", report)
    return report


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def render_workbench(mode: str, reports: dict[str, dict | None]) -> str:
    rag = reports.get("rag_eval") or {}
    claims = reports.get("claims") or {}
    conflicts = reports.get("conflicts") or {}
    quality = reports.get("quality") or {}
    diversity = reports.get("diversity") or {}
    policy_summary = rag.get("policy_summary", {})
    rows = "\n".join(
        f"<tr><td>{html.escape(policy)}</td><td>{summary.get('average_citation_count', '')}</td>"
        f"<td>{summary.get('average_claim_citation_count', '')}</td>"
        f"<td>{summary.get('average_distinct_clusters', '')}</td>"
        f"<td>{summary.get('average_conflict_note_count', '')}</td></tr>"
        for policy, summary in policy_summary.items()
    )
    query_sections = "\n".join(render_query(query) for query in rag.get("queries", []))
    diversity_section = render_diversity_section(diversity)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CANON Workbench - {html.escape(mode)}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #172026; background: #f7f8fa; }}
    main {{ max-width: 1180px; margin: 0 auto; }}
    h1, h2, h3 {{ margin: 0 0 12px; }}
    section {{ background: #fff; border: 1px solid #d9dee5; border-radius: 8px; padding: 16px; margin: 14px 0; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
    th, td {{ border-bottom: 1px solid #e6e9ee; padding: 8px; text-align: left; vertical-align: top; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; }}
    .metric {{ background: #eef3f8; border-radius: 6px; padding: 12px; }}
    .verdict {{ display: inline-block; background: #eef3f8; border-radius: 999px; padding: 3px 8px; font-size: 12px; }}
    .risk {{ background: #fff1db; }}
    .useful {{ background: #e8f5ed; }}
    .answer {{ margin: 8px 0 14px; line-height: 1.45; }}
    code {{ background: #eef3f8; padding: 2px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
<main>
  <h1>CANON Workbench</h1>
  <section class="grid">
    <div class="metric"><strong>Mode</strong><br>{html.escape(mode)}</div>
    <div class="metric"><strong>RAG queries</strong><br>{rag.get('query_count', 0)}</div>
    <div class="metric"><strong>Claims</strong><br>{claims.get('claim_count', 0)}</div>
    <div class="metric"><strong>Conflicts</strong><br>{conflicts.get('conflict_count', 0)}</div>
    <div class="metric"><strong>Works</strong><br>{quality.get('work_count', '')}</div>
    <div class="metric"><strong>Diversity queries</strong><br>{diversity.get('query_count', 0)}</div>
  </section>
  {diversity_section}
  <section>
    <h2>Policy Summary</h2>
    <table>
      <thead><tr><th>Policy</th><th>Citations</th><th>Claim Citations</th><th>Clusters</th><th>Conflict Notes</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>
  </section>
  {query_sections}
</main>
</body>
</html>
"""


def render_query(query: dict) -> str:
    runs = "\n".join(
        f"<h3>{html.escape(run['policy'])}</h3><p class='answer'>{html.escape(run['answer'])}</p>"
        f"<p><code>citations={run['citation_count']}</code> "
        f"<code>claim_citations={run['claim_citation_count']}</code> "
        f"<code>clusters={run['distinct_clusters']}</code></p>"
        for run in query.get("policy_runs", [])
    )
    return f"""<section>
  <h2>{html.escape(query.get('id', 'query'))}</h2>
  <p>{html.escape(query.get('query', ''))}</p>
  <p><em>{html.escape(query.get('importance_expectation', ''))}</em></p>
  {runs}
</section>"""


def render_diversity_section(report: dict) -> str:
    if not report:
        return """<section>
  <h2>Diversity-First Audit</h2>
  <p>No diversity audit report is available for this mode.</p>
</section>"""
    aggregate = report.get("aggregate", {})
    verdict_counts = aggregate.get("verdict_counts", {})
    verdict_rows = "\n".join(
        f"<tr><td>{html.escape(verdict)}</td><td>{count}</td></tr>"
        for verdict, count in sorted(verdict_counts.items())
    )
    query_cards = "\n".join(
        render_diversity_query(query)
        for query in report.get("queries", [])
        if query.get("verdict") in {"useful_breadth", "mixed_breadth", "off_topic_breadth_risk", "noisy_breadth"}
    )
    return f"""<section>
  <h2>Diversity-First Audit</h2>
  <div class="grid">
    <div class="metric"><strong>Baseline</strong><br>{html.escape(report.get('baseline_method_id', ''))}</div>
    <div class="metric"><strong>Diverse method</strong><br>{html.escape(report.get('diverse_method_id', ''))}</div>
    <div class="metric"><strong>Cluster delta</strong><br>{aggregate.get('average_cluster_delta', '')}</div>
    <div class="metric"><strong>Breadth precision</strong><br>{aggregate.get('average_breadth_precision', '')}</div>
    <div class="metric"><strong>Noise rate</strong><br>{aggregate.get('average_noise_rate', '')}</div>
  </div>
  <h3>Verdicts</h3>
  <table>
    <thead><tr><th>Verdict</th><th>Queries</th></tr></thead>
    <tbody>{verdict_rows}</tbody>
  </table>
  <h3>Query Examples</h3>
  {query_cards}
</section>"""


def render_diversity_query(query: dict) -> str:
    verdict = query.get("verdict", "")
    css_class = "risk" if "risk" in verdict or "noisy" in verdict else "useful" if verdict == "useful_breadth" else ""
    added = ", ".join(
        html.escape(item.get("title") or "")
        for item in query.get("added_titles", [])[:3]
    )
    return f"""<section>
  <h3>{html.escape(query.get('query_id', 'query'))} <span class="verdict {css_class}">{html.escape(verdict)}</span></h3>
  <p>{html.escape(query.get('query', ''))}</p>
  <p><code>cluster_delta={query.get('cluster_delta', '')}</code>
  <code>breadth_precision={query.get('breadth_precision', '')}</code>
  <code>noise_rate={query.get('noise_rate', '')}</code></p>
  <p>{html.escape(query.get('interpretation', ''))}</p>
  <p><strong>Added evidence:</strong> {added}</p>
</section>"""


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a static CANON workbench.")
    parser.add_argument("--mode", default="live")
    args = parser.parse_args()
    print(json.dumps(build_workbench(args.mode), indent=2))


if __name__ == "__main__":
    main()
