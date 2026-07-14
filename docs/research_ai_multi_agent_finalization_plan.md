# Research AI + Multi-Agent Researcher Finalization Plan

## Executive Decision

Keep the two projects connected, but do not merge their responsibilities.

The strongest architecture is:

```text
Research AI / CANON
  = evidence discovery, research framing, retrieval calibration, source diversity,
    qrels, citations, evidence packets, human review artifacts

Multi-Agent Researcher
  = domain intelligence orchestration, specialist agents, risk interpretation,
    disagreement mapping, alerts, multilingual/public-opinion synthesis,
    final intelligence reports
```

This boundary is the difference between a serious system and a tangled RAG demo.
Research AI should produce trustworthy, inspectable evidence packets. Multi-Agent
Researcher should consume those packets and produce intelligence artifacts
without inventing evidence.

## Product Thesis

The combined system should help users answer:

- What changed?
- Why does it matter?
- Which evidence supports this?
- Who sees it differently?
- What is uncertain?
- What should be watched next?
- What report, alert, or decision artifact should be produced?

The flagship domain should remain:

> AI infrastructure and geopolitical risk.

Do not make the first version domain-neutral. Generality is a later product
reward, not a first milestone.

## Critical Positioning

Do not frame the system as "AI agents doing research."

Frame it as:

> A human-governed intelligence workflow where Research AI maps and validates
> evidence, and Multi-Agent Researcher turns reviewed evidence packets into
> cited domain intelligence reports.

This keeps the product credible for analysts, policy researchers, public
servants, journalists, and domain experts who may not want an "AI assistant" but
do need better evidence workflows.

## System Boundary

### Research AI Owns

- source profiling and ingestion
- chunking and document normalization
- retrieval and reranking
- semantic model comparison
- query diagnostics and terminology coaching
- research skill/frame calibration
- source diversity and evidence neighborhood mapping
- qrels review packets
- LLM-assisted judge suggestions for triage
- citation-backed evidence packets
- coverage gaps and retrieval uncertainty
- human review artifacts
- retrieval/product evaluation gates

### Multi-Agent Researcher Owns

- domain ontology
- task planning
- specialist agents
- evidence requests to Research AI
- interpretation over evidence packets
- disagreement and scenario mapping
- geopolitical and cultural framing
- public opinion synthesis
- alert generation
- report generation
- report evaluation
- analyst feedback loop

### Multi-Agent Researcher Must Not Own

- low-level vector search
- document chunking
- primary citation storage
- source parsing
- embedding model selection
- unsupported factual claims
- citation rewriting without traceability

## Combined Architecture

```text
User question / monitor trigger
  -> Research skill calibration
  -> Research AI evidence request
  -> Evidence packets + source gaps + query diagnostics
  -> Multi-Agent planner
  -> Specialist agents request scoped evidence
  -> Agent outputs with evidence IDs
  -> Disagreement collector
  -> Red-team grounding pass
  -> Synthesis editor
  -> Report evaluator
  -> Human review
  -> Feedback back to Research AI and agent policies
```

## Required Integration Contract

Research AI should expose an Evidence Packet API.

Minimum request:

```json
{
  "request_id": "req_001",
  "project_id": "ai_infra_geo_risk",
  "question": "What are the emerging geopolitical risks around AI data center expansion in Latin America?",
  "research_frame": {
    "topic": "AI data center expansion",
    "subdomains": ["energy", "water", "cloud dependency", "local opposition", "sovereign AI"],
    "regions": ["Latin America"],
    "languages": ["English", "Spanish", "Portuguese"],
    "question_type": "regional risk report",
    "representation_goals": ["regional diversity", "public opinion", "policy diversity"]
  },
  "evidence_requirements": {
    "top_k": 10,
    "include_conflicts": true,
    "include_source_diversity": true,
    "include_query_diagnostics": true,
    "include_public_opinion": true,
    "minimum_source_types": ["official", "local_media", "company", "academic_or_policy"]
  }
}
```

Minimum response:

```json
{
  "request_id": "req_001",
  "status": "complete",
  "query": "...",
  "research_frame": {},
  "evidence_packets": [
    {
      "packet_id": "packet_001",
      "claim": "Data center expansion may increase local grid stress in selected regions.",
      "support_level": "mixed",
      "confidence": "medium",
      "evidence_role": "direct_support",
      "issue_categories": ["energy", "data_center_buildout"],
      "regions": ["Chile", "Brazil"],
      "languages": ["English", "Spanish"],
      "source_types": ["company", "local_media", "official"],
      "supporting_evidence": [
        {
          "evidence_id": "ev_001",
          "chunk_id": "chunk:...",
          "document_id": "doc:...",
          "title": "...",
          "source_name": "...",
          "url": "...",
          "published_at": "...",
          "text": "...",
          "citation": "..."
        }
      ],
      "conflicting_evidence": [],
      "limitations": [],
      "source_diversity": {
        "distinct_sources": 4,
        "distinct_clusters": 3,
        "dominant_source_share": 0.25
      }
    }
  ],
  "query_diagnostics": {
    "matched_terms": [],
    "weak_terms": [],
    "field_phrases": [],
    "query_variants": [],
    "drift_risk": "needs_caution"
  },
  "coverage_gaps": [
    {
      "gap": "Limited Portuguese-language local-source coverage.",
      "severity": "medium",
      "suggested_next_query": "Portuguese local opposition data center water use Brazil"
    }
  ],
  "retrieval_metrics": {
    "estimated_confidence": "medium",
    "source_diversity_status": "pass",
    "human_review_status": "not_reviewed"
  }
}
```

## Domain Ontology For MVP

Create an AI infrastructure geopolitical risk ontology with these entities and
issue categories.

### Entities

- companies: Nvidia, AMD, TSMC, Samsung, Intel, Microsoft, Google, Amazon,
  Meta, Oracle, CoreWeave, major regional cloud providers
- infrastructure: GPUs, accelerators, HBM, advanced packaging, fabs, data
  centers, cloud regions, substations, grid interconnects, cooling systems
- jurisdictions: US, EU, China, Taiwan, Gulf, India, Latin America, Africa
- institutions: regulators, energy agencies, export-control bodies, ministries,
  municipal permitting bodies
- communities: local residents, labor groups, environmental groups, indigenous
  or local-rights groups where relevant

### Issue Categories

- compute supply
- semiconductor controls
- advanced packaging
- cloud dependency
- sovereign AI
- data localization
- data center buildout
- energy demand
- grid bottlenecks
- water and cooling
- environmental impact
- permitting and local opposition
- labor and construction capacity
- public legitimacy
- national security
- cross-border investment
- sanctions and export controls
- regional industrial policy

## Research Skill Calibration

The Research AI skill/template should become the starting point for every
combined run.

For the flagship question:

```text
What are the emerging geopolitical risks around AI data center expansion in Latin America?
```

The frame should include:

- subdomains: energy, water, data centers, cloud dependency, industrial policy,
  local politics, public opinion, environmental governance
- evidence standards: official energy/grid data, company announcements, local
  reporting, regulatory filings, credible policy reports, social/public opinion
  signals with caveats
- source caution: corporate PR, activist claims without corroboration, social
  posts treated as representative opinion, English-only source bias
- representation goals: regional diversity, source-type diversity, language
  diversity, stakeholder diversity
- lenses: broad discovery, regional risk, public opinion, policy exposure,
  overclaim check

## Product Workflow

### Step 1: Start Intelligence Project

Input:

- project name
- domain
- monitored regions
- monitored languages
- issue categories
- desired report types
- review cadence

Output:

- `project_config.json`
- domain ontology loaded
- source/corpus plan

Pass criteria:

- project has explicit regions, languages, issue categories, and output types
- no monitor can run without a declared source/corpus boundary

### Step 2: Build Or Connect Evidence Corpus

Research AI profiles and ingests sources.

Initial corpora:

- company announcements and filings
- official policy/regulatory sources
- reputable news/local media
- public-opinion/social samples
- policy/think-tank reports
- academic or technical reports where available

Pass criteria:

- data card exists
- source types are visible
- language coverage is visible
- corpus limitations are visible
- unsafe or prompt-injection-like text is quarantined

### Step 3: Build Evaluation Dataset

Create a domain benchmark before claiming model quality.

Minimum MVP dataset:

- 60 benchmark questions
- 300-600 qrels candidate labels
- 10 reviewed reports
- at least 3 regions
- at least 3 source types per full report where evidence exists
- at least 2 languages if multilingual evidence is in scope

Question slices:

- compute supply
- export controls
- data center energy
- water/cooling
- cloud dependency
- local opposition
- sovereign AI
- company exposure
- public opinion
- regional risk
- off-topic/no-evidence
- false-balance/overclaim checks

Pass criteria:

- qrels packet generated
- LLM judge suggestions generated
- human audit completed on all high-priority rows and at least 20% random sample
- reviewed qrels exported

Existing Research AI commands:

```powershell
python -m canon.eval.qrels_review prepare --mode <corpus_id> --top-k 10
python -m canon.eval.llm_judge qrels --csv reports/qrels_review_tasks_<corpus_id>.csv --output reports/qrels_review_tasks_<corpus_id>.judged.csv --provider heuristic
python -m canon.eval.qrels_review import-csv --csv reports/qrels_review_tasks_<corpus_id>.csv --benchmark-id <benchmark_id> --output gold/<benchmark_id>.json
```

### Step 4: Calibrate Retrieval Models

Evaluate:

- local hashed semantic baseline
- OpenAI embeddings
- Cohere embeddings
- hybrid retrieval policies
- Cohere rerank
- heuristic rerank baseline

Pass criteria:

- no model is declared winner without human-reviewed qrels
- model report includes slice metrics
- reranker report shows whether reranking helps or hurts
- exact-match/lexical robustness is not sacrificed
- source diversity and false-balance warnings remain visible

Commands:

```powershell
python -m canon.eval.model_evaluation --mode <corpus_id> --qrels gold/<benchmark_id>.json --providers local,openai,cohere --k 10
python -m canon.eval.rerank_evaluation --mode <corpus_id> --qrels gold/<benchmark_id>.json --rerankers heuristic,cohere --base-policy rag --candidate-k 25 --k 10
```

### Step 5: Produce Evidence Packets

Research AI returns evidence packets for each scoped agent request.

Pass criteria:

- every evidence packet has citations
- every factual claim has support or says insufficient evidence
- packet includes conflicts and coverage gaps where present
- query diagnostics and source diversity are attached

### Step 6: Run Multi-Agent Orchestration

Planner decomposes a question into tasks for:

- Technical Infrastructure Agent
- Semiconductor and Supply Chain Agent
- Energy and Water Agent
- Policy and Export Controls Agent
- Geopolitical Agent
- Cultural and Multilingual Context Agent
- Market and Company Strategy Agent
- Public Opinion and Social Signal Agent
- Red-Team Agent
- Synthesis Editor Agent

Pass criteria:

- at least five agents contribute to a full brief
- each agent output has evidence IDs or insufficient-evidence status
- duplicate-agent-output rate is <= 20%
- Red-Team can block or downgrade unsupported claims

### Step 7: Generate Reports

Required MVP reports:

- Weekly Intelligence Brief
- Regional Risk Report
- Company Exposure Report
- Public Opinion Report
- Alert Digest

Each report must include:

- executive summary
- what changed
- evidence-backed developments
- regional or entity exposure
- public opinion or explicit public-evidence gap
- contradictions and uncertainty
- next signals to watch
- citation appendix
- source gaps
- analyst review status

Pass criteria:

- citation coverage >= 95% for factual claims
- unsupported factual claims <= 5%
- uncertainty section present
- contradictions section present when conflicts exist
- source gaps section present
- report export available in Markdown and JSON

### Step 8: Human Review And Feedback

Human reviewer labels:

- report usefulness
- unsupported claims
- missing perspective
- source quality concern
- alert usefulness
- actionability
- overclaim risk

Pass criteria:

- feedback exported
- repeated errors become regression tests
- judge labels stay provisional
- final acceptance remains human-owned

### Step 9: Alerts And Monitoring

Monitoring should detect:

- new export-control item
- data center project announcement
- permitting/legal challenge
- water/energy constraint
- public controversy spike
- company exposure change
- new sovereign AI partnership

Pass criteria:

- alert includes evidence trigger
- alert includes affected region/entity/issue
- alert includes confidence and uncertainty
- duplicate alert rate <= 10%
- human can mark useful/not useful

## Evaluation Benchmarks

### Benchmark 1: Evidence Retrieval Benchmark

Owner: Research AI

Dataset:

- 60 domain questions
- qrels with graded relevance 0-3
- candidate pool from multiple retrieval policies

Metrics:

- nDCG@10
- Recall@10
- MRR@10
- Precision@10
- coverage@1/3/10
- source diversity
- graph cluster diversity
- dominant source share
- false semantic neighbor rate

MVP targets:

- nDCG@10 improves over local baseline or no winner is declared
- Recall@10 does not drop by more than 5% when reranking
- source diversity passes on at least 80% of benchmark questions
- all off-topic/no-evidence questions trigger weak-support or insufficiency

### Benchmark 2: Research Frame Coverage Benchmark

Owner: Research AI

Question:

> Did retrieval cover the research frame the user calibrated?

Metrics:

- intended subdomain coverage
- missing subdomain count
- source-type coverage
- language coverage
- representation-goal coverage
- terminology match rate
- suggested follow-up query usefulness

MVP targets:

- >= 80% of reports identify at least one useful missing/weak area
- >= 70% of suggested follow-up queries rated useful or partially useful
- every frame-vs-retrieval report labels inferred coverage as diagnostic, not
  proof

### Benchmark 3: Agent Grounding Benchmark

Owner: Multi-Agent Researcher

Test:

- provide agents with evidence packets
- require structured outputs
- verify every factual claim maps to evidence IDs

Metrics:

- grounded claim ratio
- unsupported claim count
- citation coverage
- insufficient-evidence compliance
- red-team block rate

MVP targets:

- grounded claim ratio >= 0.95
- citation coverage >= 0.95
- unsupported factual claims <= 0.05 per report claim set
- 100% of unsupported Red-Team objections resolved or marked unresolved

### Benchmark 4: Multi-Agent Contribution Benchmark

Owner: Multi-Agent Researcher

Metrics:

- agent completion rate
- unique perspective contribution
- duplicate-agent-output rate
- evidence request success rate
- agent latency
- agent disagreement contribution

MVP targets:

- at least five agents contribute to each full brief
- duplicate-agent-output rate <= 0.20
- every agent has at least one distinct issue category or interpretation role
- failed agents are visible in report metadata

### Benchmark 5: Report Quality Benchmark

Owner: Multi-Agent Researcher

Human labels:

- usefulness 1-5
- actionability 1-5
- evidence trust 1-5
- uncertainty clarity 1-5
- missing perspective yes/no
- unsupported claim yes/no

Automated checks:

- required sections present
- citations present
- uncertainty present
- source gaps present
- next-watch signals present

MVP targets:

- analyst usefulness average >= 4/5 over 10 reviewed reports
- actionability average >= 4/5
- uncertainty clarity average >= 4/5
- 100% reports include citations/source gaps/uncertainty
- no report passes with unresolved unsupported claims

### Benchmark 6: Public Opinion Representation Benchmark

Owner: Multi-Agent Researcher, with Research AI evidence support

Metrics:

- platform diversity
- language coverage
- region coverage
- narrative cluster count
- stance distribution
- volatility score
- caveat inclusion rate
- minority narrative surfaced count

MVP targets:

- public opinion outputs never treat social posts as representative polling
- every public opinion report includes source/platform caveats
- at least three narrative clusters when enough evidence exists
- missing-language warning appears when non-English evidence is absent for a
  local topic

### Benchmark 7: Alert Benchmark

Owner: Multi-Agent Researcher

Dataset:

- seeded timeline of evidence items
- known duplicate items
- known high/medium/low alert triggers

Metrics:

- alert precision
- duplicate alert rate
- severity calibration
- time to detect
- time to human review
- alert usefulness

MVP targets:

- duplicate alert rate <= 0.10
- high-severity alert precision >= 0.80 on reviewed alert set
- every alert has evidence trigger and citation IDs
- every alert has recommended follow-up

### Benchmark 8: End-To-End Acceptance Scenario

Question:

```text
What are the emerging geopolitical risks around AI data center expansion in Latin America?
```

Expected behavior:

1. Research AI calibrates frame.
2. Research AI retrieves multilingual evidence.
3. Research AI expands to water, grid, permitting, cloud regions, local
   opposition, sovereign AI, and investment without drifting into generic AI
   news.
4. Research AI returns cited evidence packets, query diagnostics, source gaps,
   and source diversity.
5. Multi-Agent Researcher runs at least five agents.
6. Red-Team checks unsupported claims.
7. Synthesis Editor produces cited report.
8. Alert Digest identifies new high-priority signals.
9. Human reviewer can audit evidence and rate usefulness.

Pass criteria:

- all factual claims cite Research AI evidence IDs
- report includes at least three issue categories
- report includes at least two regions if evidence exists
- report includes public opinion or explicit public evidence gap
- report includes uncertainty and contradictions
- report includes next-watch signals
- final report usefulness >= 4/5 from reviewer

## Implementation Roadmap

### Phase 0: Freeze Product Boundary

Deliverables:

- this plan
- final Research AI product journey
- final Multi-Agent boundary doc

Exit criteria:

- two-service boundary accepted
- no shared vector/index ownership
- evidence packet contract accepted

### Phase 1: Research AI Evidence Packet API

Deliverables:

- evidence packet schema
- CLI/API function to emit packets from `synthesize` output
- source diversity and query diagnostics attached
- coverage gaps included

Tests:

- packet schema validation
- citation presence
- no evidence -> insufficient evidence
- conflicts included when available

Exit criteria:

- 25 seed queries can produce evidence packets
- packets include citation IDs and retrieval diagnostics

### Phase 2: Frame-Vs-Retrieval Coverage

Deliverables:

- research frame JSON schema
- coverage evaluator
- missing subdomain/source/language report
- follow-up query generator

Tests:

- known frame with missing subdomain produces warning
- matched frame elements appear in report
- report labels inferred coverage as diagnostic

Exit criteria:

- frame coverage report exists for at least 10 acceptance questions

### Phase 3: Domain Fixture For AI Infrastructure

Deliverables:

- small fixture corpus for AI infrastructure geopolitical risk
- English + at least one Spanish/Portuguese sample
- public opinion/social sample
- company/policy/energy-water samples
- seed qrels/questions

Tests:

- ingest passes
- data card generated
- qrels review packet generated
- source diversity generated

Exit criteria:

- local fixture supports end-to-end demo without external crawling

### Phase 4: Multi-Agent Mock Evidence Runner

Deliverables:

- agent schemas
- planner
- mock evidence packet client
- five initial agents
- red-team pass
- synthesis editor

Tests:

- agents cannot emit factual claims without evidence IDs
- red-team blocks unsupported claims
- duplicate-agent-output is measured

Exit criteria:

- mock weekly brief generated from fixture evidence packets

### Phase 5: Live Integration Contract Test

Deliverables:

- Research AI evidence API endpoint or callable adapter
- Multi-Agent evidence client
- shared integration fixture

Test query:

```text
AI data center water risk in Latin America
```

Exit criteria:

- Multi-Agent report cites Research AI evidence IDs
- no invented citations
- source gaps included

### Phase 6: Evaluation Harness

Deliverables:

- retrieval benchmark report
- frame coverage report
- agent grounding report
- report quality checklist
- alert benchmark
- human feedback CSV/JSON

Exit criteria:

- all MVP target metrics are computed
- failing metrics block release

### Phase 7: First Production Handoff

Deliverables:

- weekly intelligence brief
- regional risk report
- company exposure report
- public opinion report
- alert digest
- evaluation dashboard/report
- human review packet

Exit criteria:

- 10 reviewed reports
- average usefulness >= 4/5
- grounded claim ratio >= 0.95
- unsupported factual claims <= 5%
- citation coverage >= 0.95
- duplicate alert rate <= 0.10

## Build Order Recommendation

As lead data scientist/engineer, I would build in this order:

1. Research AI evidence packet exporter.
2. Research frame schema and frame-vs-retrieval coverage.
3. AI infrastructure fixture corpus and seed qrels.
4. Multi-Agent mock evidence runner.
5. Agent grounding and red-team enforcement.
6. Report generation from mock packets.
7. Contract test with real Research AI packets.
8. Rerank/model evaluation on reviewed qrels.
9. Human feedback and report benchmark.
10. Monitoring and alert deduplication.

Do not start with autonomous agents. Start with the evidence contract and
evaluation harness. Agents are only valuable if their outputs are grounded and
measurable.

## Final Definition Of Done

The combined project is ready for final handoff when:

- Research AI can produce reviewed, cited evidence packets from a calibrated
  research frame.
- Multi-Agent Researcher can consume those packets and produce cited domain
  intelligence reports.
- Every factual report claim maps to evidence IDs.
- The system exposes uncertainty, disagreement, source gaps, and public opinion
  caveats.
- Retrieval models and rerankers are evaluated on human-reviewed qrels.
- LLM judge suggestions are available but marked provisional.
- Human review can approve/revise/reject outputs.
- Reports and alerts meet MVP target metrics.
- The flagship scenario works end to end:

```text
What are the emerging geopolitical risks around AI data center expansion in Latin America?
```

## Research Grounding

This plan is aligned with:

- ReAct: reasoning interleaved with tool/evidence calls.
- AutoGen: configurable multi-agent workflows.
- CAMEL: role-based agent collaboration.
- Self-RAG and CRAG: corrective behavior when evidence is weak.
- ARES: context relevance, answer faithfulness, answer relevance.
- GraphRAG: graph/community evidence mapping for broad sensemaking.
- BEIR-style evaluation: heterogeneous retrieval benchmarks and qrels.

The important move is not copying those systems. The contribution is combining
research-frame calibration, evidence-packet retrieval, source diversity,
multi-agent interpretation, red-team grounding, and human-reviewed intelligence
report evaluation into one auditable workflow.

