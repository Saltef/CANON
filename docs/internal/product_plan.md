# CANON Product Plan

CANON is a human-in-the-loop evidence briefing workbench for teams that need
cited, cautious answers from controlled literature and source corpora.

The product is intentionally narrow. It helps reviewers find candidate evidence,
understand why it was selected, draft a cited answer, and see where support is
weak or contested. It does not make final claims without human review.

## Product Shape

- Evidence answer API: cautious answers with citations, support level,
  limitations, and conflict notes.
- Evidence review API: compact source previews, retrieval explanations, and
  score contributors so a human can inspect the answer.
- Comparison API: side-by-side retrieval behavior for debugging and reviewer
  trust, not as an end-user leaderboard.
- Guardrail API: claim limits, corpus limitations, regression-gate status, and
  readiness checks.
- Query lingo layer: matched terms, missing terms, field phrases, suggested
  alternate phrasings, and answer-stability notes across query rewrites.
- Portfolio demo path: Dockerized local API and static workbench first, richer
  review UI later.

## First Use Case

**Evidence briefing copilot for literature-backed decisions.**

A policy analyst, research assistant, or strategy researcher asks a focused
question. CANON retrieves evidence, ranks and explains sources, drafts a cited
brief, highlights weak support or disagreement, and leaves the final judgment to
the reviewer.

The best early workflow is:

1. Ask a focused question.
2. Inspect top evidence and explanations.
3. Review query-language suggestions and try alternate phrasing when useful.
4. Review conflict notes and limitations.
5. Revise or reject the generated answer.
6. Export the cited answer and audit trail.

## Tangible Product Goals

- Evidence brief in 15 minutes: a reviewer can move from one focused question to
  a cautious cited first draft with visible support, limitations, and conflicts.
- Reviewer triage queue: retrieved evidence is grouped and explained so the
  human can review high-risk, weak, or conflicting items first.
- Overclaim check for drafts: a user can paste a draft claim or paragraph and
  CANON flags wording that is stronger than the available evidence supports.

## Industry Pilot Finish Line

The first industry-quality finish line is a human-in-the-loop evidence briefing
pilot. CANON is ready for that pilot when an industry reviewer can ask a focused
question, inspect evidence and query-lingo diagnostics, generate a cautious
cited brief, revise or reject it, and export an audit trail.

Pass criteria are defined in `industry_release_goal.md`. The short version:

- 30-question acceptance set
- at least 80% of questions return three or more reviewer-rated relevant
  evidence items in the top 10
- at least 90% of briefs include citations, support assessment, limitations,
  and conflict notes when applicable
- fewer than 5% of answer claims are reviewer-rated unsupported
- every answer includes query diagnostics with matched terms, weak terms,
  suggested field phrases, query variants, drift risk, and exploration level
- at least 70% of first-pass briefs can be reviewed in 15 minutes or less
- exploratory query variants are visible, optional, and recorded in the audit
  trail
- all automated tests and product smoke tests pass

## Query Lingo Design

CANON should teach users how their wording affects retrieval. For each question,
the product should expose the field language behind the results:

- matched terms that helped retrieve evidence
- user terms that were weak or missing in the corpus
- recurring phrases from high-ranking evidence
- suggested alternate phrasings
- semantic neighbors that lead to different evidence
- stability notes showing whether the answer changes under rewrites

This layer must be wired into semantic retrieval in two places:

1. **From user query to corpus:** use the same semantic scoring path that
   retrieves candidates from the user's query. Expose semantic similarity,
   lexical match, focus coverage, syntax/form match, missing query terms, and
   high-impact query terms.
2. **From semantic results back to query language:** analyze the top semantic
   result neighborhood and extract recurring terms, bigrams, field phrases,
   acronyms, method terms, and outcome terms that were absent from the user's
   original query.

The key output is not just "try these prompts." It is:

- what the user's wording retrieved
- what the result set appears to call the same topic
- which suggested wording changes materially change the retrieved evidence

## Semantic Exploration Freedom

The lingo layer should support a controlled level of freedom. Some useful
phrases will not be the highest-probability match to the original query, but
they may still be thematically important to the topic. CANON should expose these
as lower-confidence exploratory suggestions, not silently merge them into the
main answer.

The setting should behave like this:

- `strict`: only terms strongly supported by the original query and top results
- `balanced`: include field synonyms and recurring result-neighborhood phrases
- `exploratory`: include lower-probability adjacent concepts when they remain
  semantically close to the original query and the accepted evidence set

Every suggested term should carry:

- source: original query, retrieved result phrase, accepted reviewer pattern, or
  generated hypothesis
- semantic distance from the original query
- evidence support count from the result neighborhood
- reviewer history: accepted, rejected, unseen, or mixed
- drift risk: low, medium, or high

Learned patterns should come from human-in-the-loop behavior:

- accepted citations strengthen related phrases and query variants
- rejected evidence weakens its terms as future suggestions
- reviewer-edited wording becomes a candidate safer phrase
- repeated topic sessions can form a local vocabulary profile

Guardrail:

Exploratory terms can be shown to the user and run side by side, but they should
not be used to strengthen a final claim unless retrieved evidence supports them
and the human reviewer accepts the connection.

This is a human-in-the-loop prompt-learning aid. CANON may suggest alternate
phrasing, but it should not silently replace the user's query.

First API shape:

- `POST /v1/query-diagnostics`: return term coverage, suggested field phrases,
  query variants, and retrieval-stability summary.
- `POST /v1/compare`: include query variant comparison as a product-facing mode,
  not only policy comparison.
- `POST /v1/answer`: include a compact `query_diagnostics` object in the answer
  response.

Suggested `query_diagnostics` shape:

```json
{
  "original_query": "Do sanctions work?",
  "query_to_corpus": {
    "matched_terms": ["sanctions"],
    "weak_terms": ["work"],
    "semantic_similarity_summary": {
      "top_mean": 0.42,
      "top_max": 0.71
    }
  },
  "result_neighborhood": {
    "field_phrases": [
      "economic sanctions effectiveness",
      "coercive diplomacy",
      "targeted sanctions",
      "sanctions evasion"
    ],
    "missing_from_query": [
      "compliance",
      "enforcement",
      "humanitarian effects",
      "regime change outcomes"
    ]
  },
  "query_variants": [
    {
      "query": "economic sanctions effectiveness",
      "reason": "High-frequency phrase in semantically similar results.",
      "result_overlap_with_original": 0.54,
      "freedom_level": "balanced",
      "drift_risk": "low"
    },
    {
      "query": "coercive diplomacy and regime change outcomes",
      "reason": "Adjacent concept found in the semantic result neighborhood.",
      "result_overlap_with_original": 0.31,
      "freedom_level": "exploratory",
      "drift_risk": "medium"
    }
  ],
  "stability": {
    "status": "sensitive",
    "note": "Alternate field terminology changes the evidence set materially."
  }
}
```

Implementation sketch:

- Reuse `canon.retrieval.semantic.semantic_scores` for query-to-document
  similarity.
- Reuse candidate `signals.query_text_relevance.semantic_similarity` and
  `focus_coverage` from retrieval traces.
- Extract n-grams from the top semantic candidates and rank them by frequency,
  distinctiveness, and absence from the original query.
- Add a learned pattern store for reviewer-accepted phrases, rejected phrases,
  edited safer wording, and topic-local vocabulary profiles.
- Score candidate terms with an exploration budget that balances semantic
  distance, result-neighborhood support, reviewer history, and drift risk.
- Run selected query variants through the same retrieval pipeline and report
  rank overlap, added evidence, removed evidence, and answer-stability status.

## Things CANON Should Do Great

- Evidence triage over a known corpus.
- Cited first-draft answers.
- Weak-support, disagreement, and abstention surfacing.
- Query wording diagnostics and field-terminology coaching.
- Source-quality and retrieval-explanation visibility.
- Reproducible audit artifacts for reviewers.

## Things CANON Should Not Claim

- It does not determine truth.
- It does not replace expert review.
- It does not guarantee corpus completeness.
- It does not validate high-stakes decisions without human approval.
- It does not prove one retrieval policy is globally best.

## Non-Negotiable Guardrails

- Every externally usable answer must expose citations and limitations.
- Weak support must trigger caution or abstention instead of a confident answer.
- Conflict notes must be visible when retrieved evidence disagrees.
- Query rewrites must be visible to the user and recorded in the audit trail.
- Corpus limitations stay visible in product responses.
- Diagnostic baselines stay in the product only when labeled as diagnostic.
- The product must not claim a stable global winner while the claim-decision
  report blocks that claim.
- Internal qrels can support technical diagnostics, not public benchmark claims.
- The product corpus must be built from explicit source profiles. OpenAlex is
  the implemented primary scholarly index and author index; Crossref, Semantic
  Scholar, Unpaywall, and DOAJ/CORE remain planned verification/backfill sources
  until connectors are implemented and audited.

## First Product Endpoints

- `GET /health`
- `GET /v1/summary`
- `GET /v1/reports/audit`
- `GET /v1/reports/claim-decision`
- `GET /v1/reports/data-card`
- `GET /v1/reports/regression-gate`
- `GET /v1/reports/diversity`
- `GET /v1/diversity/queries`
- `GET /v1/diversity/queries/{query_id}`
- `POST /v1/answer`
- `POST /v1/compare`
- `POST /v1/query-diagnostics`
- `POST /v1/diversity-audit`
