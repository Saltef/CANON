# CANON Human Review Rubric

Use this rubric to complete industry-pilot review files such as
`reports/human_review_tasks_v1.json` or `reports/human_review_tasks_v1.review.csv`,
and intelligence-brief review files such as
`reports/intelligence_brief_review_tasks_ai_infra_geo_risk_demo.review.csv`.

CANON should be reviewed as a first-pass evidence briefing tool. Do not score it
as an autonomous expert or final decision system.

## Intelligence Brief Required Fields

For intelligence brief review packets, complete:

- `usefulness_1_5`
- `actionability_1_5`
- `evidence_trust_1_5`
- `uncertainty_clarity_1_5`
- `missing_perspective`
- `unsupported_claim`
- `overclaim_risk`
- `final_review_status`
- `reviewer_notes`

Create the review handoff CSV and commands:

```powershell
python -m canon.product.intelligence_review --review-handoff --records reports/intelligence_brief_review_tasks_ai_infra_geo_risk_demo.json
```

The intelligence final check treats the fixture as reviewed only after every
intelligence brief record has valid required labels. The current automated
handoff intentionally stops at `blocked_human_review` until that happens.

## Industry Pilot Required Fields

For every question, complete:

- `relevant_evidence_count`
- `substantive_claim_count`
- `unsupported_claim_count`
- `brief_review_minutes`
- `evidence_relevance`
- `citation_support`
- `answer_usefulness`
- `query_lingo_ratings`
- `drift_risk_review`
- `final_review_status`

The industry gate treats a question as unreviewed until all required fields are
present and valid.

After importing labels, generate the feedback summary:

```powershell
python -m canon.product.intelligence_review --feedback-report --records reports/intelligence_brief_review_tasks_ai_infra_geo_risk_demo.completed.json
```

The feedback report aggregates reviewer scores, counts rejected or
needs-more-evidence outcomes, and proposes regression candidates for unsupported
claims, missing perspectives, and overclaim risk. It does not automatically
change qrels or release status.

## Evidence Relevance

- `irrelevant`: the retrieved evidence does not help answer the query.
- `partial`: at least one item is related, but important evidence is missing or
  off-topic results dominate.
- `relevant`: several retrieved items directly help answer the query.
- `highly_relevant`: the top evidence is focused, specific, and sufficient for a
  cautious first brief.

For `relevant_evidence_count`, count only top-10 evidence items that directly
support answering the question.

## Citation Support

- `unsupported`: cited evidence does not support the answer's main claims.
- `weak`: some claims are supported, but important claims are too broad,
  uncited, or only indirectly supported.
- `adequate`: main claims are supported, with some caution needed.
- `strong`: main claims are clearly tied to cited evidence and phrased with
  appropriate limits.

For `substantive_claim_count`, count factual or interpretive claims that matter
to the answer. For `unsupported_claim_count`, count claims that are not
traceable to cited evidence.

## Answer Usefulness

- `unusable`: the answer would mislead or require a full rewrite.
- `major_revision`: the answer has useful pieces but needs substantial repair.
- `minor_revision`: the answer is directionally useful and needs light editing.
- `usable`: the answer can serve as a cautious first-pass brief.

## Query Lingo Usefulness

Use one rating per suggested phrase you inspect:

- `not_useful`: off-topic, misleading, or too generic.
- `somewhat_useful`: related but only mildly helpful.
- `useful`: likely to help a newcomer search the topic.
- `very_useful`: strong field vocabulary or an important adjacent concept.

Rate suggestions based on whether they teach the user useful terminology, not
whether they perfectly replace the original query.

## Drift Risk Review

- `acceptable`: variants and suggested phrases stay within the user's topic.
- `needs_caution`: variants may broaden the search and should be inspected by a
  human before use.
- `drifted`: variants move away from the user's intent or introduce misleading
  themes.

## Final Review Status

- `accepted`: the brief can be used as-is for a first pass.
- `revised`: the reviewer made edits but the output was useful.
- `rejected`: the output should not be used.
- `needs_more_evidence`: the answer may be plausible, but the corpus evidence is
  insufficient.

## Timing

Record `brief_review_minutes` as the actual first-pass review time. Include time
spent reading the answer, checking evidence, inspecting query lingo, and making
the final review decision.
