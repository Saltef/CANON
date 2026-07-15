# RAG Importance, Trust, And Safety Roadmap

Date: 2026-07-04

This note reviews CANON as a research system for assigning importance to text in
RAG. It is intentionally critical. The project is promising because it already
treats importance as multidimensional and auditable, but it is not yet a
trustworthy-source system, a quality classifier, or a prompt-injection defense.

## Executive Take

The central idea is right: RAG should not retrieve by semantic similarity alone.
For research and decision support, retrieval should separate at least four
questions:

1. Is this text relevant to the query?
2. Is this text important evidence for the question being asked?
3. Is the source trustworthy enough to use?
4. Is the text safe to expose to the generator?

CANON currently handles the first two partly, handles source trust only through
rough proxies, and has not yet built the fourth as a first-class layer. The next
scientific jump should be to split the existing single ranking score into a
typed evidence-selection pipeline: candidate retrieval, evidence utility
estimation, source trust estimation, safety screening, diversification, and
generation-time citation control.

My strongest recommendation: stop thinking of "importance" as one score. Model
it as a vector with explicit decision points. Some dimensions should boost
ranking, some should only annotate, and some should gate or quarantine content.

## What CANON Already Does Well

CANON is not a toy semantic-search wrapper. The repo already has several
research-grade instincts:

- It has explicit retrieval policies in `conf/settings.toml`: lexical,
  semantic, balanced, recency, diversity, focus-diverse, source-quality-heavy,
  and conflict-aware.
- It records score components and trace explanations through
  `canon/retrieval/policies.py` and `canon/retrieval/trace.py`.
- It separates source-level signals from chunk-level signals in the docs:
  citation impact, retraction, open access, reference coverage, author signal,
  section role, claim density, recency, and diversity.
- It has a meaningful evaluation scaffold: method comparisons, perturbation
  robustness, citation faithfulness, regression gates, data cards, qrels
  validation, bootstrap uncertainty, paired significance checks, and a
  scientific audit.
- It has already learned a real lesson: diversity-first retrieval is not
  globally better. The 10k portfolio report shows focus-gated diversity
  improved useful breadth and reduced measured noise relative to ungated
  diversity, but the claim is properly conditional.

That last point matters. A system that discovers "my clever ranking idea is only
sometimes good" is more scientifically serious than one that forces every
metric to say yes.

## What Is Scientifically Fragile Right Now

### 1. Source Quality Is Too Shallow

The current `source_quality` score is mostly a metadata availability score:
non-retracted, open access, PDF URL, source name present, and references present.
That is useful for diagnostics, but it is not source trust.

Open access is not quality. A PDF URL is not quality. Having a source name is
not quality. Reference-list coverage helps, but only weakly. Citation counts and
author prominence are influence signals, not reliability signals. In some
domains, they can amplify prestige, field age, language, institution, and
network effects.

Current risk: a highly cited, on-topic, available source can dominate even if it
is methodologically weak, outdated for the query, contradicted by later work, or
from a venue with poor editorial controls.

### 2. Text Importance Is Mostly Section And Keyword Heuristics

Chunk importance currently uses section labels and claim-density markers such as
"find", "show", "evidence", "significant", and "effect". This is a good seed,
but it overvalues assertion-heavy prose. A confident claim is not necessarily
good evidence.

In scholarly RAG, the most important text is often not the most claim-dense
text. It may be a limitation, sample definition, identification assumption,
measurement detail, robustness check, preregistered hypothesis, null result, or
contradictory finding.

Current risk: CANON can over-rank polished conclusions and under-rank the
methods/results details needed to judge whether the conclusion deserves weight.

### 3. Weighted Linear Ranking Is Too Blunt

The weighted average in `canon/retrieval/policies.py` is understandable and
auditable, which is a virtue. But linear scalarization hides incompatible
decision types.

For example:

- Retraction should be a hard gate, not just a low score.
- Prompt-injection suspicion should quarantine or sanitize a chunk, not merely
  subtract a little ranking weight.
- Low relevance should usually block source-quality boosting.
- High source trust should not rescue irrelevant text.
- Conflicting evidence should often be selected deliberately, not treated as
  noise.

Current risk: a single score can mix "good evidence", "popular source",
"semantically close", and "unsafe but high similarity" into one opaque outcome,
even if the trace shows the components.

### 4. The Evaluation Stack Is Stronger Than The Ground Truth

The repo has a good evaluation harness, but much of the judgment is still based
on internal weak qrels, technical probes, lexical faithfulness, and deterministic
claim scaffolds. That is fine for development, but not enough for strong
scientific claims.

The existing scientific audit is honest about this: public benchmark coverage is
currently a smoke fixture, not a real external benchmark. The claim model is not
validated. The corpus is OpenAlex-derived and social-science-specific, so the
results do not yet generalize.

Current risk: the system can become excellent at passing its own tests while
remaining brittle under external corpora, expert labels, adversarial documents,
or real mixed-quality web content.

### 5. Prompt Injection Is Not Yet In The Architecture

The user goal mentions eventually flagging low-quality or harmful content to
avoid prompt injection. That goal should not be bolted onto generation later.
The literature increasingly points to retrieval-stage and pre-generation
controls because asking the generator to ignore malicious retrieved text is
not enough.

Current risk: if CANON later ingests web pages, PDFs, comments, reviews,
emails, or semi-trusted institutional documents, the same retrieval machinery
that finds relevant text can deliver adversarial instructions directly into the
model context.

## Literature And Project Signals

### Reliability-Aware RAG

[Retrieval-Augmented Generation with Estimation of Source Reliability](https://arxiv.org/abs/2410.22954)
proposes estimating heterogeneous source reliability and using it during both
retrieval and aggregation. The important idea for CANON is not the exact method;
it is the separation between source reliability and document relevance. CANON's
source-quality-heavy policy points in this direction, but it does not yet
estimate reliability from agreement patterns or calibration over sources.

### Corrective And Diagnostic RAG

[Corrective Retrieval Augmented Generation](https://arxiv.org/abs/2401.15884)
uses a retrieval evaluator to decide whether retrieved documents are good enough
and whether corrective actions are needed. CANON should adopt this pattern: do
not just rank top-k; classify the retrieved context as sufficient, weak,
conflicted, unsafe, stale, or needing expansion.

[RAGChecker](https://arxiv.org/abs/2408.08067) is useful because it evaluates
RAG at finer granularity across retrieval and generation rather than collapsing
system quality into one answer score. CANON already has pieces of this
philosophy. It should go further by reporting which evidence units are useful,
which answer claims they support, and where source trust or safety gates changed
the context.

[TREC 2024 RAG / AutoNuggetizer](https://arxiv.org/abs/2411.09607) is relevant
because "nuggets" are closer to the unit CANON needs than whole-document
relevance. Importance for RAG is often answer-nugget utility, not document
similarity.

### Trustworthy RAG And Citation Control

[TrustRAG](https://arxiv.org/abs/2502.13719) emphasizes hierarchical/contextual
chunking, utility-based filtering, and fine-grained citation enhancement.
CANON should borrow the structure: chunk context should travel with every
retrieved unit, and citation links should be sentence-level, not answer-level.

[GraphRAG from Microsoft](https://arxiv.org/abs/2404.16130) and
[LightRAG](https://arxiv.org/abs/2410.05779) show why graph structure matters
for global or sensemaking queries. CANON already has graph diagnostics and
cluster-aware diversity. The next step is to make graph position an evidence
interpretation layer: central claims, bridge claims, minority clusters,
methodological schools, and contradiction neighborhoods.

### Prompt Injection And Retrieval-Stage Defense

[Not what you've signed up for](https://arxiv.org/abs/2302.12173) established
indirect prompt injection as a threat where retrieved external content can
manipulate an LLM-integrated application.

[OWASP LLM01: Prompt Injection](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
continues to treat prompt injection as a primary LLM-application risk. The
mitigations are architectural: least privilege, isolating untrusted content,
human oversight for sensitive actions, input/output validation, and adversarial
testing.

[Backdoored Retrievers for Prompt Injection Attacks on RAG](https://arxiv.org/abs/2410.14479)
shows that retrievers themselves can be part of the attack surface. The lesson:
do not assume a dense retriever is neutral infrastructure.

[DataFilter](https://arxiv.org/abs/2510.19207), a 2025 preprint, proposes
test-time filtering that removes malicious instructions from data before the
backend LLM sees it. Whether or not that exact model holds up, the design
principle is right for CANON: untrusted retrieved content should be transformed,
screened, or dropped before generation.

[SD-RAG](https://arxiv.org/abs/2601.11199), a 2026 preprint, argues for
retrieval-stage selective disclosure rather than prompt-level enforcement. This
is highly aligned with CANON's future: security and privacy policies should be
enforced before augmentation.

[RAGPart and RAGMask](https://arxiv.org/abs/2512.24268), a 2025 preprint,
focuses on retrieval-stage defenses against corpus poisoning. The useful idea is
to test whether suspicious terms or partitions cause unstable retrieval
behavior.

[Can It Reach the Generator?](https://arxiv.org/abs/2605.28017), a 2026
preprint, is especially relevant because it evaluates attacks across a realistic
retriever-reranker-generator pipeline. This is the right framing for CANON's
security evaluation: attack success should be measured by whether malicious
content is ingested, retrieved, reranked, exposed, and obeyed.

Security caveat: several 2025-2026 papers above are recent preprints. Treat them
as design inspiration and adversarial test ideas, not settled doctrine.

## A Better Mental Model Of Text Importance

CANON should define importance as task-conditional evidence utility:

```text
Importance(chunk, query, task) =
  relevance_to_question
  x evidence_role
  x source_trust
  x support_specificity
  x marginal_utility_given_already_selected_context
  x safety_eligibility
```

But these terms should not all be multiplied into one production score. The
pipeline should use them differently:

- Relevance: candidate retrieval and minimum floor.
- Syntax/form relevance: candidate recall, evidence routing, and domain
  annotation.
- Evidence role: reranking and context composition.
- Source trust: reranking, aggregation weighting, and warning language.
- Support specificity: citation selection and answer grounding.
- Marginal utility: diversity, novelty, and redundancy control.
- Safety eligibility: gate, quarantine, sanitize, or require review.

## Proposed Signal Taxonomy

### Query-Text Relevance

Purpose: find candidate evidence.

Signals:

- BM25 / lexical relevance.
- Dense embedding similarity.
- Syntax/form relevance: whether the text looks like the kind of evidence the
  query asks for, such as method detail, limitation, quantitative result, legal
  authority, public expression, transcript/dialogue, risk disclosure,
  definition, or narrative context.
- Cross-encoder or LLM reranker score.
- Query term coverage after removing generic terms.
- Negative query constraints and exclusion penalties.
- Query intent classification: factual, comparative, disagreement-seeking,
  recency-sensitive, method-focused, source-audit, or safety-critical.

Decision use: retrieval floor and reranking. Relevance should usually be a hard
minimum before any prestige or quality boost is allowed.

Current implementation note: CANON now exposes `syntax_relevance` separately
from `semantic_similarity`. The first version uses transparent category
detectors plus a corpus-level rarity profile. This is useful for experiments,
but it is not yet a supervised form classifier or a domain-validated discourse
model.

### Evidence Role

Purpose: decide what kind of text the chunk contributes.

Signals:

- Section role: abstract, introduction, theory, methods, data, results,
  robustness, limitations, conclusion.
- Rhetorical role: claim, evidence, method, assumption, limitation, caveat,
  definition, citation context, related-work summary.
- Claim type: causal, descriptive, predictive, normative, methodological,
  measurement, null finding, replication, critique.
- Specificity: whether the chunk names variables, population, time period,
  method, effect direction, uncertainty, and scope.
- Local citation role: whether the passage is citing others, reporting the
  authors' own result, or speculating.

Decision use: context composition. For a scientific answer, a methods chunk and
a limitation chunk may be more important than a conclusion chunk.

### Source Trust

Purpose: estimate whether the source deserves evidentiary weight.

Signals:

- Retraction, correction, expression of concern.
- Venue identity and venue type, with a transparent allowlist/registry rather
  than a hidden prestige score.
- DOI / Crossref / OpenAlex / Semantic Scholar consistency.
- Peer-review status where available.
- Publisher and journal integrity checks, including known predatory-source lists
  if licensing permits.
- Citation impact normalized by field and year, not raw count alone.
- Reference-list quality and citation-network neighborhood.
- Author signal, but separated from quality to avoid prestige double counting.
- Replication status, preregistration, data/code availability, open materials.
- Agreement with other independent sources after controlling for citation
  dependence.

Decision use: aggregation weighting, warnings, and source-audit reports. Some
signals should annotate rather than rank.

### Evidence Corroboration

Purpose: avoid treating a single retrieved chunk as settled truth.

Signals:

- Number of independent sources supporting the same claim.
- Contradictory claims and disagreement clusters.
- Citation-dependence graph: independent support versus repeated citation of
  one original source.
- Recency of confirming or disconfirming work.
- Methodological diversity among supporting sources.

Decision use: answer confidence, conflict notes, abstention, and synthesis
structure.

### Safety And Injection Risk

Purpose: prevent unsafe or adversarial retrieved text from controlling the
generator.

Signals:

- Instructional phrases inside retrieved content: "ignore previous",
  "system prompt", "developer message", "do not reveal", "execute", "call this
  tool", "send credentials", and similar patterns.
- Hidden or suspicious formatting: tiny text, white text, CSS hiding, unusual
  Unicode, base64 blobs, HTML comments, prompt-looking delimiters.
- Role confusion markers: text pretending to be system/developer/user messages.
- Exfiltration intent: asking for secrets, credentials, private files, tokens,
  chain-of-thought, or policy text.
- Tool-control intent: asking the assistant to browse, click, buy, email, run
  shell commands, or modify memory.
- Retrieval instability: the chunk ranks highly only because of suspicious
  trigger strings or repeated unnatural terms.
- Corpus provenance risk: web page, user-uploaded doc, unknown PDF, comment
  field, OCR artifact, or generated content.

Decision use: hard gate, sanitization, quarantine, or separate display to the
user. Safety should not be a normal ranking boost/penalty.

## Recommended Architecture

### Stage 1: Candidate Retrieval

Retrieve broadly with lexical plus dense methods. Keep recall high. Do not let
source trust dominate this stage, because that can hide minority, new, or
contradictory evidence.

Output: 50-200 candidate chunks with raw retrieval traces.

### Stage 2: Structured Chunk Enrichment

Attach typed metadata to every chunk:

- `evidence_role`
- `syntax_profile`
- `claim_type`
- `method_signal`
- `limitation_signal`
- `citation_context_role`
- `source_trust_vector`
- `safety_risk_vector`
- `provenance`
- `corroboration_group`

This can start deterministic and later become model-assisted with validation.

### Stage 3: Safety Gate Before Reranking

Run prompt-injection and content-risk screening before any chunk can reach the
generator.

Possible decisions:

- `allow`: usable as evidence.
- `sanitize`: strip instruction-like spans but keep factual content.
- `quarantine`: exclude from generation but report in diagnostics.
- `block`: exclude and raise a safety event.

This is the most important architectural change for the later security goal.

### Stage 4: Evidence Utility Reranking

Rerank with a utility function that knows the query type. A recency-heavy query,
a disagreement query, and a methods query should not use the same notion of
importance.

Use a non-linear policy:

```text
if relevance < floor:
  reject
if safety_decision in {quarantine, block}:
  reject from generator context
if source has retraction:
  allow only for "retraction/history/critique" queries
score = relevance + evidence_utility + trust_adjustment + novelty_bonus
```

### Stage 5: Context Assembly

Assemble a balanced evidence packet, not just top-k chunks:

- 1-2 high-relevance direct evidence chunks.
- 1 methods or measurement chunk when available.
- 1 limitation or caveat chunk when available.
- 1 corroborating independent source.
- 1 contradicting source for disagreement-sensitive queries.
- No quarantined content in the generator prompt.

### Stage 6: Generation With Citation Discipline

The generator should receive evidence as data, not instructions. Each evidence
item should have an ID, source metadata, safety status, and allowed use.

The answer should cite at sentence or claim level. It should state weak support,
conflicts, and source-quality limitations. It should abstain when evidence is
thin, unsafe, or not sufficiently relevant.

### Stage 7: Post-Generation Verification

Run answer checks after generation:

- Every factual sentence has a supporting evidence ID.
- Cited evidence actually contains the necessary claim terms or is verified by
  NLI/judge.
- No answer sentence follows instructions from retrieved text.
- No quarantined evidence was cited.
- Claims from low-trust sources are hedged or excluded.

## Concrete Repo Pathway

### Phase A: Refactor Importance Into Typed Decisions

Add a new module, probably `canon/importance/`, with typed records:

- `SourceTrustVector`
- `EvidenceRoleVector`
- `SafetyRiskVector`
- `EvidenceDecision`
- `ContextPacket`

Keep the current weighted policies, but make them consume typed fields rather
than ad hoc dictionaries. This preserves current tests while creating room for
gates and richer reports.

Acceptance criteria:

- Existing retrieval policies still run.
- Trace output includes separate `relevance`, `evidence_role`, `source_trust`,
  `corroboration`, and `safety` sections.
- Retraction and safety decisions can be hard gates.

### Phase B: Improve Text-Level Importance

Replace the current claim-density heuristic with a richer rhetorical-role
extractor.

Start deterministic:

- Section title classifier.
- Limitation/caveat markers.
- Method/data markers.
- Result/effect markers.
- Null/uncertainty markers.
- Citation-context markers.

Then create a small labeled set and compare deterministic labels against an LLM
or scientific sentence classifier.

Acceptance criteria:

- A chunk can be important because it is a method, limitation, null result, or
  contradiction, not only because it is claim-dense.
- Reports show evidence-role distribution by policy.
- Evaluation includes role coverage: direct claim, method, limitation,
  corroboration, contradiction.

### Phase C: Build A Real Source Trust Layer

Create a source registry and provenance resolver:

- OpenAlex work/source IDs.
- Crossref DOI metadata.
- Semantic Scholar citation fields if available.
- Retraction Watch or Crossmark where available/licensed.
- Venue/publisher registry.
- Source type: peer-reviewed article, preprint, book, policy report, web page,
  blog, news, unknown.

Important: keep `author_score` separate. It should never be silently folded into
source quality.

Acceptance criteria:

- `source_quality` becomes `source_trust_vector`.
- Reports show missingness for every trust dimension.
- Rankings can be compared with and without prestige-like signals.
- Retractions/corrections are treated as gates or severe warnings.

### Phase D: Add Corroboration And Conflict-Aware Importance

Use the existing graph and claim infrastructure to group evidence by claim.

Add:

- Claim normalization.
- Claim-support clusters.
- Contradiction candidates.
- Independence estimate based on citation graph and source overlap.
- Support count by independent source, not just chunk count.

Acceptance criteria:

- Answers can say "supported by multiple independent sources", "single-source
  support", or "conflicted".
- Retrieval evaluation rewards useful contradiction retrieval for disagreement
  queries.
- The system can downweight repeated near-duplicate support.

### Phase E: Add Safety Screening Before Generation

Create `canon/safety/` with deterministic detectors first:

- Prompt-injection phrase patterns.
- Role-label spoofing.
- Hidden-text and suspicious markup checks for HTML/PDF-derived text.
- Secrets/tool-control/exfiltration patterns.
- Unicode/control-character anomalies.
- Repetition and trigger-token diagnostics.

Add a `safety_decision` field to every retrieved chunk.

Acceptance criteria:

- Quarantined chunks never enter synthesis prompts.
- Safety events are visible in retrieval traces and reports.
- Tests include malicious chunks that are relevant but blocked.
- Retrieval metrics distinguish "retrieved" from "exposed to generator".

### Phase F: Build An Adversarial RAG Security Benchmark

Add a small local benchmark with clean and poisoned corpora:

- Benign scholarly abstracts.
- Malicious prompt-injection chunks.
- Biased-but-factual chunks.
- Hidden instruction variants.
- Role-confusion variants.
- Retrieval trigger attacks.
- Low-quality SEO-like content.

Metrics:

- Attack retrieval rate.
- Attack reaches reranker rate.
- Attack exposed to generator rate.
- Attack followed by generator rate.
- Benign utility loss.
- False positive quarantine rate.

Acceptance criteria:

- Security reports show the full path from ingestion to generation.
- A defense is not credited unless it reduces attack exposure without destroying
  benign retrieval quality.

### Phase G: Upgrade Evaluation With Human And External Anchors

Add at least three external anchors:

- A BEIR/TREC-style retrieval benchmark with real qrels.
- A RAG answer benchmark or TREC RAG-style nugget set.
- A small expert-labeled CANON set for source trust and evidence role.

Acceptance criteria:

- Public benchmark metrics are reported separately from internal metrics.
- Confidence intervals are shown for key comparisons.
- Expert labels are used to calibrate trust/importance classifiers.
- The claim-decision report blocks claims that lack external or human support.

## Suggested Data Model

```python
@dataclass(frozen=True)
class EvidenceSignals:
    relevance: dict[str, float]
    evidence_role: dict[str, float]
    source_trust: dict[str, float | str | bool | None]
    corroboration: dict[str, float | int | str]
    safety: dict[str, float | str | bool]
    provenance: dict[str, str | None]

@dataclass(frozen=True)
class EvidenceDecision:
    chunk_id: str
    retrieval_score: float
    utility_score: float
    trust_tier: str
    safety_decision: str
    allowed_uses: list[str]
    reasons: list[str]
```

This design lets CANON say:

- "Relevant but unsafe."
- "Trusted but not relevant."
- "Relevant and important, but single-source."
- "Highly cited but contradicted."
- "Useful for historical context, not current answer."
- "Blocked from generation due to role-confusion injection."

Those distinctions are the heart of a serious RAG importance system.

## Hard Product Rules I Would Adopt

1. No retrieved text is allowed to issue instructions to the model.
2. Safety screening happens before generation, not only after.
3. Retractions and prompt-injection flags are gates, not ranking weights.
4. Source trust cannot compensate for low query relevance.
5. High citation count cannot be treated as truth.
6. Author prominence must remain separate from source quality.
7. Every answer claim should be traceable to sentence-level evidence.
8. Contradictory evidence should be surfaced for contested questions.
9. Missing trust metadata should be reported as missing, not imputed as neutral.
10. Evaluation must track what reaches the generator, not just what retrieval
    returns.

## Near-Term Implementation Plan

### Week 1: Make Importance Typed

- Add `canon/importance/signals.py`.
- Convert current chunk/work dictionaries into typed signal vectors.
- Extend retrieval traces with typed sections.
- Add tests proving current policies produce the same rankings when gates are
  disabled.

### Week 2: Add Safety Decisions

- Add deterministic safety scanner.
- Add poisoned fixture chunks.
- Add tests that relevant malicious chunks are retrieved but not exposed to
  synthesis.
- Add a safety report under `reports/`.

### Week 3: Improve Evidence Roles

- Replace simple `claim_density` with richer role signals.
- Add role coverage metrics to RAG eval.
- Add a policy variant that requires method/limitation coverage when available.

### Week 4: Source Trust Registry

- Add source trust schema.
- Add registry-backed venue/source metadata.
- Add missingness diagnostics.
- Split `source_quality` into transparent subdimensions.

### Month 2: Corroboration And External Evaluation

- Group claims across chunks.
- Estimate independent support.
- Add contradiction-aware context packets.
- Run one public retrieval benchmark beyond smoke validation.
- Add a small human/expert labeling workflow for evidence role and trust.

### Month 3: Security Evaluation

- Add adversarial corpus benchmark.
- Track attack reach through retrieval, reranking, context assembly, and
  generation.
- Compare deterministic scanner, sanitization, and model-based filtering.
- Add regression gates for attack exposure and benign utility loss.

## What Not To Do

- Do not market the current system as trustworthy RAG. It is an experimental
  importance-aware retrieval workbench.
- Do not hide weak signals behind a confident "quality score".
- Do not let source prestige become a truth proxy.
- Do not assume semantic search plus citations equals grounded reasoning.
- Do not rely on prompt instructions to solve prompt injection.
- Do not optimize only average retrieval metrics; contested and adversarial
  slices are where the system's real value will show.

## Bottom Line

CANON's best path is to become an evidence selection laboratory: a system that
shows why text was selected, what role it plays, how trustworthy its source is,
whether it is corroborated, and whether it is safe to show to the generator.

The repo already has enough scaffolding to get there. The next move is not a
bigger embedding model. It is a clearer separation between relevance,
importance, trust, corroboration, and safety.
