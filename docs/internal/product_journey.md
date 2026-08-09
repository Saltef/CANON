# CANON Product Journey

CANON should be built as an evidence discovery workbench for people who do
research work, not as a generic AI chat assistant. AI is part of the
infrastructure, but the product journey is organized around research practice:
framing, searching, inspecting, revising, and synthesizing with human judgment.

## Product Position

CANON helps a researcher move from a vague or contested question to a reviewed
evidence map and cautious synthesis.

The product does not promise final truth. It promises a better research path:

- clearer research framing
- better terminology
- broader evidence discovery
- visible source concentration
- explicit disagreement and weak support
- auditable human decisions

## Primary V1 User

The first user should be an interdisciplinary evidence reviewer:

- policy analyst
- academic research assistant
- think-tank researcher
- strategy or product researcher working from public literature

Journalists and public servants are adjacent users, but they should not define
the first workflow. The shared engine can support them later with workflow
variants for provenance, public records, primary-source tracing, jurisdictional
comparison, and stakeholder evidence.

## Core User Journey

```text
Start a project
  -> Define research skill/template
  -> Build or choose corpus
  -> Calibrate research frame
  -> Run initial retrieval
  -> Inspect evidence neighborhoods
  -> Learn terminology and revise query
  -> Compare research lenses
  -> Review evidence and source diversity
  -> Use LLM judge for triage only
  -> Human accepts/revises/rejects
  -> Export evidence map, brief, and audit trail
```

## Stage 1: Project Setup

The user creates a research project:

- project name
- topic
- corpus or data source
- intended audience
- sensitivity level
- desired output

Product behavior:

- profile source shapes
- ingest flexible data
- create data card
- show corpus limitations before answering

Existing modules:

- `canon.ingest.flexible`
- `canon.corpus.build`
- `canon.reports.data_card`
- `docs/test_own_documents.md`

## Stage 2: Research Skill Calibration

The research skill template is the front door. It lets a user calibrate the
research direction before retrieval.

The user defines:

- topic
- subdomain constellation
- question type
- evidence standards by subdomain
- source inclusion and caution rules
- time/geographic/cultural scope
- representation goals
- known vocabulary
- vocabulary to discover
- known disagreements
- research lens
- desired output

Product behavior:

- treat the frame as a hypothesis, not as evidence
- show which parts of the frame are user-supplied versus system-suggested
- carry the frame into retrieval, diagnostics, and review
- let the user edit the frame after seeing evidence

Existing module/doc:

- `research_skill_template.md`

Needed product object:

```json
{
  "research_frame_id": "frame_001",
  "topic": "climate migration",
  "subdomains": ["sociology", "anthropology", "economics", "geography"],
  "question_type": "evidence_map",
  "evidence_standards": {
    "sociology": ["surveys", "demographic datasets", "peer reviewed studies"],
    "anthropology": ["ethnography", "fieldwork", "local accounts"]
  },
  "representation_goals": ["disciplinary_diversity", "methodological_diversity"],
  "known_vocabulary": ["climate migration", "displacement"],
  "vocabulary_to_discover": ["immobility", "managed retreat"],
  "status": "draft"
}
```

## Stage 3: Initial Retrieval

The user runs a question against the selected corpus.

Product behavior:

- retrieve evidence with a default policy
- show source previews and score explanations
- do not jump straight to a final answer
- surface corpus and retrieval limitations immediately

Existing modules:

- `canon.retrieval.experiment`
- `canon.retrieval.trace`
- `canon.synthesis.answer`
- `canon.product.service.answer`

## Stage 4: Evidence Neighborhood Map

CANON maps the retrieved evidence:

- sources
- works
- graph clusters
- subdomain coverage
- dominant source share
- disagreement/conflict notes
- weak or off-topic candidates

Product behavior:

- show breadth and concentration before synthesis
- warn when one cluster or source dominates
- distinguish direct evidence from background evidence

Existing modules:

- `canon.eval.source_diversity`
- `canon.eval.research_lens`
- `canon.claims.conflict`
- `canon.evidence.corroboration`

Needed next module:

- frame-vs-retrieval coverage report:
  - intended subdomains
  - retrieved subdomain signals
  - missing or weakly covered frame elements
  - suggested follow-up queries

## Stage 5: Query Lingo Coach

CANON teaches the user how wording changes retrieval.

Product behavior:

- show matched query terms
- show weak or missing query terms
- extract recurring field phrases from retrieved semantic neighborhoods
- generate optional query variants
- label drift risk
- record accepted/rejected phrases

Existing module:

- `canon.retrieval.query_diagnostics`

Journey rule:

The system may suggest terms, but the user chooses whether to search them. It
must not silently rewrite the research question.

## Stage 6: Research Lens Comparison

The user compares research directions:

- broad discovery
- canonical depth
- recent scan
- controversy map
- mechanism map
- overclaim check

Product behavior:

- run the same question through multiple lenses
- show how evidence changes
- show whether depth and breadth disagree
- recommend human inspection points, not final conclusions

Existing module:

- `canon.eval.research_lens`

Current command:

```powershell
python -m canon.eval.research_lens "Do sanctions work?" --mode social_science_ir_v1_harvest10 --top-k 10
```

## Stage 7: Review Queue

The user reviews evidence and answer quality.

Product behavior:

- export answer-review CSV
- export qrels/relevance-review CSV
- optionally add LLM/heuristic judge suggestions
- keep judge labels separate from human labels
- prioritize high-risk rows

Existing modules:

- `canon.product.industry_pilot`
- `canon.eval.qrels_review`
- `canon.eval.llm_judge`

Current commands:

```powershell
python -m canon.product.industry_pilot --mode social_science_ir_v1_harvest10 --prepare-review
python -m canon.eval.qrels_review prepare --mode social_science_ir_v1_harvest10 --top-k 10
python -m canon.eval.llm_judge qrels --csv reports/qrels_review_tasks_social_science_ir_v1_harvest10.csv --output reports/qrels_review_tasks_social_science_ir_v1_harvest10.judged.csv --provider heuristic
```

## Stage 8: Model And Rerank Calibration

The user tests retrieval models after qrels labels exist.

Product behavior:

- compare embedding providers
- compare rerankers
- report slice-level performance
- avoid declaring a global winner
- require human-reviewed qrels before production model claims

Existing modules:

- `canon.eval.model_evaluation`
- `canon.eval.rerank_evaluation`

Current commands:

```powershell
python -m canon.eval.model_evaluation --mode social_science_ir_v1_harvest10 --qrels gold/ir_qrels_social_science_ir_v1_harvest10.json --providers local,openrouter,cohere --k 10
python -m canon.eval.rerank_evaluation --mode social_science_ir_v1_harvest10 --qrels gold/ir_qrels_social_science_ir_v1_harvest10.json --rerankers heuristic,cohere --base-policy rag --candidate-k 25 --k 10
```

## Stage 9: Synthesis And Export

Only after framing, retrieval diagnostics, source diversity, and review should
CANON produce an outward-facing synthesis.

Product behavior:

- produce a cautious cited brief
- include limitations
- include conflict notes
- include research frame and accepted/rejected decisions
- export audit trail

Existing modules:

- `canon.synthesis.answer`
- `canon.product.final_check`
- `canon.product.release_audit`

## What The User Sees

The product should feel like a guided workbench:

1. **Research Frame Panel**
   Shows the template, selected subdomains, evidence standards, and scope.

2. **Corpus Panel**
   Shows source count, document types, limitations, and ingestion status.

3. **Evidence Map**
   Shows clusters, source concentration, top evidence, and disagreement.

4. **Terminology Panel**
   Shows matched terms, weak terms, field phrases, query variants, and drift.

5. **Lens Comparison**
   Shows breadth/depth/recency differences.

6. **Review Queue**
   Shows rows needing human judgment, with LLM judge suggestions marked as
   provisional.

7. **Synthesis**
   Shows a cautious answer only after the evidence trail is visible.

## MVP Boundary

For v1, do not try to serve every researcher.

Build for:

> interdisciplinary evidence review over a controlled corpus.

Do not claim:

- full literature review automation
- comprehensive web research
- expert replacement
- universal model selection
- equal performance across journalism, academia, and public administration

## Product Success Criteria

CANON is working if a reviewer can:

- define a research frame
- see whether retrieval matches that frame
- learn better field terminology
- identify missing or overrepresented evidence neighborhoods
- compare breadth and depth lenses
- use judge suggestions to reduce review workload
- complete final labels with human authority
- export an audit trail another reviewer can inspect
