# CANON Use Case

CANON is a human-in-the-loop evidence review tool for teams that need to turn a
messy literature or source set into a cautious, cited briefing.

It is not a truth engine, autonomous researcher, or replacement for expert
review. It is designed to do a few narrow jobs well:

- find candidate evidence across a controlled corpus
- explain why each source was retrieved
- show whether support is strong, weak, mixed, or missing
- surface disagreement and source-quality concerns before drafting
- produce a cited answer that a human can accept, revise, or reject

## Primary User

The primary user is an analyst, policy researcher, strategy associate, graduate
researcher, or product researcher who needs to answer questions such as:

- What does the literature say about this policy claim?
- Which sources support, qualify, or conflict with this conclusion?
- Are we relying on narrow, stale, low-quality, or one-sided evidence?
- Where should a human reviewer spend attention first?

## Product Promise

CANON helps a reviewer move from a broad question to an evidence-backed first
brief faster, while keeping uncertainty visible.

The product does not promise final truth. It promises a better review queue:
ranked evidence, cited drafts, conflict notes, source diagnostics, and explicit
limitations.

## Core Workflow

1. A user asks a focused question.
2. CANON retrieves candidate evidence from a named corpus.
3. The system ranks evidence using relevance, source quality, citation signals,
   recency, diversity, conflict awareness, and safety checks.
4. CANON returns a cited answer with support level, limitations, and conflict
   notes.
5. A human reviewer inspects the evidence, edits the answer, marks weak or
   misleading citations, and decides what can be used.

## Query Lingo Coach

Search quality is shaped by word choice. A new user may describe a topic in
plain language while the corpus uses field-specific terms, methods, acronyms,
or policy phrases. CANON should make that gap visible and help the user learn
the language of the field.

The goal is not to hide retrieval sensitivity. The goal is to teach it.

This feature must be connected directly to semantic retrieval, not bolted on as
a generic prompt-tip layer. It should use two linked signals:

1. **User-query semantic retrieval:** compare the user's exact query against the
   corpus and expose which query terms, bigrams, and semantic matches drove or
   failed to drive retrieval.
2. **Result-neighborhood semantic retrieval:** inspect the high-ranking semantic
   result set and extract the field phrases, recurring concepts, and semantic
   neighbors that the corpus itself uses for the same topic.

The product question is:

> What did the user's wording retrieve, and what language do the retrieved
> results suggest the field uses instead?

CANON should also learn from prior review patterns. If users repeatedly accept
evidence, phrases, or query variants around a topic, those patterns should
become stronger suggestions in future sessions. If users reject a term as
tangential, promotional, or misleading, that term should become less likely to
appear as a suggested expansion.

This creates a controlled exploration loop:

- original query terms keep the anchor
- accepted result-neighborhood terms expand the user's vocabulary
- rejected terms reduce future drift
- lower-probability thematic terms can appear when they remain close to the
  original query and the accepted evidence neighborhood

When a user asks a question, CANON should show:

- matched query terms that helped retrieval
- important query terms that found little or no evidence
- field phrases that appear often in high-ranking evidence
- suggested alternate phrasings to try
- semantic neighbors that retrieve different evidence
- terms and phrases from the semantic result set that were not present in the
  original user query
- whether results changed materially under query rewrites

Example:

User query:

> Do sanctions work?

CANON may suggest:

- economic sanctions effectiveness
- sanctions compliance
- sanctions enforcement
- coercive diplomacy
- targeted sanctions
- sanctions evasion
- regime change outcomes
- humanitarian effects of sanctions

The user learns that the field may not only ask "do sanctions work?" It may ask
which sanctions, under what mechanism, against which outcome, and with which
tradeoffs.

This should be presented as a learning aid, not as automatic query rewriting
that silently changes the user's intent.

Definition of done:

- each answer shows a compact query-language panel
- low-match user terms are visible
- suggested field phrases are tied to retrieved evidence, not invented
- suggestions include evidence-result phrases discovered from the semantic
  neighborhood, not only terms from the original query
- exploratory suggestions are labeled by distance from the original query:
  close match, field synonym, adjacent concept, or exploratory tangent
- alternate phrasings can be run side by side
- the system reports whether the answer is stable or sensitive across rewrites
- user-selected query variants are kept in the review audit trail

Why this matters:

People learning a field also learn its vocabulary. If CANON exposes the terms
that move retrieval, it becomes both an evidence tool and a topic-learning tool.

## Three Tangible Goals

### Goal 1: Evidence Brief In 15 Minutes

Help a reviewer produce a first-pass cited brief from a focused question in 15
minutes or less.

Target workflow:

1. Ask one question.
2. Review the top evidence list.
3. Inspect support level, limitations, and conflict notes.
4. Export or revise a short cited brief.

Definition of done:

- at least 80% of test questions return three or more relevant evidence items
- every answer includes citations, limitations, and support assessment
- weak-support answers are cautious or abstain
- a reviewer can trace each major sentence back to retrieved evidence

Why this matters:

This is the simplest product promise. CANON should make the first brief faster
without hiding the review work that remains.

### Goal 2: Reviewer Triage Queue

Turn retrieval results into a review queue that tells the human where to spend
attention first.

Target workflow:

1. Ask a question or upload a draft claim.
2. CANON groups evidence into support, qualification, conflict, and weak match.
3. The reviewer opens the highest-risk items first.
4. The reviewer marks evidence as accepted, rejected, needs review, or missing.

Definition of done:

- every retrieved item has an explanation trace
- conflict and weak-support items are visible above routine supporting evidence
- review decisions can be exported with the answer
- the system records why evidence was accepted or rejected

Why this matters:

The product is not just "answer generation." Its real value is helping a person
review faster and with fewer hidden assumptions.

### Goal 3: Overclaim Check For Drafts

Use CANON as a claim-boundary checker for memos, grant drafts, literature
reviews, policy briefs, or product strategy documents.

This is the unique use case: instead of starting with a question, the user starts
with a draft paragraph or claim. CANON checks whether the available corpus
actually supports the wording.

Target workflow:

1. Paste a draft claim or paragraph.
2. CANON extracts the major claims.
3. For each claim, CANON retrieves supporting, qualifying, and conflicting
   evidence.
4. The system flags statements that look too strong for the available evidence.
5. The human rewrites, narrows, cites, or removes the claim.

Definition of done:

- extracted claims are shown separately from the draft text
- each claim receives a support status: supported, qualified, contested, weak,
  or not found
- at least one suggested safer wording is generated for weak or contested claims
- the final output preserves a review trail linking claims to evidence

Why this matters:

Most teams do not only need answers. They need to know when their own draft is
outrunning the evidence. CANON can become useful at the moment before a memo,
brief, or proposal leaves the team.

## What CANON Should Do Great

### 1. Evidence Triage

CANON should make the first pass over a source set more efficient. Its output
should help the user see which sources deserve review, why they were selected,
and what risk each source carries.

Success looks like:

- the top evidence contains genuinely relevant sources
- each source has an explanation trace
- weak, noisy, or off-topic evidence is visible
- reviewers can quickly identify what to read next

### 2. Cited Brief Drafting

CANON should produce a short, cautious answer grounded in retrieved evidence.
The draft should cite the evidence it depends on and avoid strong conclusions
when support is thin.

Success looks like:

- every substantive sentence is tied to cited evidence
- the answer distinguishes support from qualification or disagreement
- unsupported claims are omitted or marked as limitations
- the draft is useful enough for a human to revise rather than start from zero

### 3. Disagreement And Weak-Support Surfacing

CANON should be especially good at saying, "this is not settled." For real
knowledge work, that may be more valuable than a smooth answer.

Success looks like:

- conflicting evidence is surfaced instead of averaged away
- low-support answers trigger abstention or caution
- source limitations are part of the response
- users can inspect why the system did not make a stronger claim

### 4. Review Audit Trail

CANON should leave behind enough trace information for another reviewer to
understand the answer.

Success looks like:

- retrieval policy and score contributors are visible
- cited chunks can be traced back to source records
- generated reports capture corpus limitations
- portfolio/demo outputs are reproducible from documented commands

## Human-In-The-Loop Boundary

A human reviewer remains responsible for:

- final conclusions
- citation validity
- domain interpretation
- deciding whether evidence is representative
- handling sensitive, legal, medical, financial, or policy implications
- approving any answer used externally

CANON should assist the reviewer, not bypass them.

## Limitations To Keep Visible

- Corpus coverage may be incomplete or biased.
- OpenAlex-derived metadata does not guarantee full-text coverage.
- Retrieval can miss relevant evidence.
- Source-quality signals are proxies, not expert judgments.
- Citation counts and author prominence can reinforce existing field bias.
- Generated answers can still misread or overstate source material.
- Current faithfulness checks are technical aids, not a substitute for human
  verification.
- Public benchmark validation and expert labels are still needed before broad
  claims.

## Best First Market Framing

The strongest initial use case is not "AI research assistant for everything."
It is:

> Evidence briefing copilot for literature-backed decisions, with built-in
> citation review, disagreement surfacing, and claim limits.

Good first users:

- policy and think-tank analysts
- academic research assistants
- market and strategy teams reviewing public literature
- product teams evaluating technical or regulatory claims
- grant, diligence, or memo writers who need cited evidence fast

Poor first users:

- anyone expecting fully autonomous conclusions
- teams without a reviewer who can inspect citations
- high-stakes domains without expert oversight
- broad web search use cases where corpus boundaries are unknown

## Portfolio Story

For GitHub, present CANON as a product-minded evidence workbench:

- focused problem: cited evidence briefing from controlled corpora
- visible workflow: retrieve, compare, draft, inspect, decide
- honest limits: human approval required
- technical depth: evaluation gates, traceable ranking, safety checks, data cards
- product discipline: the system knows when not to overclaim
