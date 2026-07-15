# CANON Industry Pilot Goal

## Finish Line

Ship CANON as a high-quality pilot for human-in-the-loop evidence briefing over
controlled source corpora.

The pilot is ready when an industry reviewer can use CANON to:

1. ask a focused question,
2. inspect retrieved evidence and query-lingo diagnostics,
3. generate a cautious cited first brief,
4. see weak support, disagreement, and corpus limitations,
5. revise or reject the output,
6. export an audit trail showing what evidence and query variants were used.

The release should be judged as an analyst-assist tool, not as an autonomous
decision system.

## Target User

The target user is an analyst, strategy researcher, policy researcher, product
researcher, diligence associate, or research operations team member who needs a
defensible first pass over a bounded literature or document corpus.

## Pass Criteria

### 1. Evidence Brief Quality

CANON passes if, on a 30-question acceptance set:

- at least 80% of questions return three or more reviewer-rated relevant
  evidence items in the top 10
- at least 90% of generated briefs include citations, support assessment,
  limitations, and conflict notes when applicable
- at least 90% of substantive answer claims can be traced to cited evidence
- fewer than 5% of answer claims are rated unsupported by reviewers
- weak-support questions produce caution or abstention rather than confident
  conclusions

### 2. Query Lingo Diagnostics

CANON passes if, on the same acceptance set:

- every answer includes query diagnostics
- matched and weak user terms are visible
- at least 80% of suggested field phrases are traceable to retrieved evidence or
  accepted reviewer patterns
- every query variant reports result overlap, drift risk, and exploration level
- exploratory variants are shown as optional tests, never silent rewrites
- reviewers rate at least 70% of suggested field phrases as useful or
  reasonable for learning the topic vocabulary

### 3. Human Review Workflow

CANON passes if a reviewer can complete a first-pass brief workflow in 15
minutes or less for at least 70% of acceptance questions.

The workflow must include:

- reading the generated answer
- inspecting top evidence
- reviewing query-lingo suggestions
- checking limitations and conflict notes
- marking the answer as accepted, revised, rejected, or needs more evidence

### 4. Safety And Claim Boundaries

CANON passes if:

- prompt-injection or unsafe retrieved text is rejected or quarantined before
  ordinary generation context
- corpus limitations remain visible in answers and exported reports
- query rewrites and exploratory terms are recorded in the audit trail
- the product does not claim final truth, full corpus coverage, or autonomous
  domain judgment
- high-risk decisions remain explicitly human-approved

### 5. Operational Quality

CANON passes if:

- the full automated test suite passes
- a smoke test covers health, summary, answer, compare, and query diagnostics
- product responses use stable schemas with documented fields
- errors are structured and actionable
- a reproducible demo can be run from documented commands
- generated pilot reports are committed or reproducible for GitHub reviewers

## Required Evidence Artifacts

The pilot should produce these artifacts:

- `reports/industry_pilot_acceptance_v1.json`
- `reports/industry_pilot_acceptance_v1.md`
- `reports/query_diagnostics_acceptance_v1.json`
- `reports/evidence_briefing_acceptance_v1.md`
- `reports/human_review_tasks_v1.json`
- `reports/human_review_status_v1.json`
- `reports/human_review_status_v1.md`
- `reports/product_release_audit_<mode>.json`
- `reports/product_release_audit_<mode>.md`
- `reports/product_final_check_<mode>.json`
- `reports/product_final_check_<mode>.md`
- `reports/product_readiness_<mode>.json`
- `reports/regression_gate_<mode>_baseline_methods_v1.json`

The Markdown report should be readable by a hiring manager, product leader, or
technical reviewer without running the code.

Run the pilot gate with:

```powershell
python -m canon.product.industry_pilot --mode social_science_ir_v1_harvest10
```

Without a 30-question human-reviewed acceptance file, this gate should fail
closed. That is intentional: an industry-ready claim requires reviewer evidence,
not only automated smoke tests.

For report-only local inspection before labels exist, use:

```powershell
python -m canon.product.industry_pilot --mode social_science_ir_v1_harvest10 --no-fail
```

To prepare the 30-question human review packet and companion reviewer reports,
run:

```powershell
python -m canon.product.industry_pilot --mode social_science_ir_v1_harvest10 --prepare-review
```

This writes:

- `reports/human_review_tasks_v1.json`
- `reports/query_diagnostics_acceptance_v1.json`
- `reports/evidence_briefing_acceptance_v1.md`

Re-running `--prepare-review` preserves existing review labels by question id.
Use `--reset-review-labels` only when intentionally starting the review over.

Before running the final gate, check review completion:

```powershell
python -m canon.product.industry_pilot --mode social_science_ir_v1_harvest10 --records reports/human_review_tasks_v1.json --export-review-csv --output reports/human_review_tasks_v1.review.csv
python -m canon.product.industry_pilot --mode social_science_ir_v1_harvest10 --records reports/human_review_tasks_v1.json --import-review-csv reports/human_review_tasks_v1.review.csv
python -m canon.product.industry_pilot --mode social_science_ir_v1_harvest10 --records reports/human_review_tasks_v1.json --review-status
```

The CSV export/import path is for human convenience only. The JSON review packet
remains the authoritative input to the pilot gate.
CSV import validates labels before writing; duplicate ids, unknown ids, invalid
labels, or malformed numeric counts leave the JSON packet unchanged.

Reviewers should use `docs/human_review_rubric.md` when assigning labels and
numeric counts.

The JSON task packet is intentionally compatible with the pilot gate. After a
reviewer fills in the `review` fields, run:

```powershell
python -m canon.product.industry_pilot --mode social_science_ir_v1_harvest10 --records reports/human_review_tasks_v1.json
```

Run the product smoke gate before product readiness:

```powershell
python -m canon.product.smoke --mode social_science_ir_v1_harvest10
python -m canon.product.readiness --mode social_science_ir_v1_harvest10
python -m canon.product.release_audit --mode social_science_ir_v1_harvest10
python -m canon.product.final_check --mode social_science_ir_v1_harvest10 --records reports/human_review_tasks_v1.json --no-fail
```

The release audit checks source-report integrity before interpreting pass/fail
statuses. Missing source reports, wrong report ids, or reports generated for a
different mode block release even if a source report contains `status: pass`.
The final-check command runs these product gates sequentially, writes
`reports/product_final_check_<mode>.json/.md`, and treats incomplete human review
as a blocked release rather than an automated failure. For CI or release gating,
omit `--no-fail`; any status other than `pass`, including incomplete human
review, exits nonzero. The final-check report records path, size, and SHA-256
fingerprints for the human review task packet and the source artifacts used by
the release audit. A missing or malformed review task packet is treated as a
source-integrity failure; a valid packet with incomplete labels is treated as a
human-review block.

## Acceptance Question Set

The acceptance set should include at least 30 questions across:

- straightforward evidence requests
- vague beginner wording
- terminology-sensitive questions
- disagreement or mixed-evidence questions
- weak-support or no-evidence questions
- recency-sensitive questions
- source-quality-sensitive questions
- overclaim checks from draft paragraphs

At least 10 questions should intentionally use non-expert wording so the query
lingo layer can be tested.

## Human Label Schema

Each reviewed answer should include:

- `evidence_relevance`: irrelevant, partial, relevant, highly relevant
- `citation_support`: unsupported, weak, adequate, strong
- `answer_usefulness`: unusable, needs major revision, needs minor revision,
  usable
- `query_lingo_usefulness`: not useful, somewhat useful, useful, very useful
- `drift_risk_review`: acceptable, needs caution, drifted
- `final_review_status`: accepted, revised, rejected, needs more evidence

## Release Blockers

Do not present the pilot as industry-ready if any of these are true:

- unsupported answer claims exceed 5%
- query variants are applied silently
- exploratory suggestions are not labeled with drift risk
- citation support cannot be traced from answer to evidence
- unsafe or instruction-like retrieved text can enter ordinary generation
  context
- corpus limitations are hidden from product responses
- the acceptance report cannot be reproduced

## Success Statement

CANON v1 is successful when reviewers say:

> This does not replace my judgment, but it helps me find evidence faster,
> understand the field vocabulary, see where the answer is weak, and produce a
> better first draft with a clear audit trail.
