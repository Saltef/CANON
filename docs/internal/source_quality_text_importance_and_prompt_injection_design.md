# Source Quality, Text Importance, And Prompt-Injection Design

Date: 2026-07-04

## Position

CANON should use one common evidence schema across domains, but it should not
use one universal quality score. The stable standard is architectural:
separate relevance, evidence role, source trust, corroboration, provenance, and
safety. The domain-specific layer should decide how to interpret those signals.

This matters because economics, psychology, anthropology, cultural studies, and
history do not share the same epistemic contract. A randomized experiment,
archival monograph, ethnography, legal filing, investor presentation, and scraped
web page can all be useful evidence, but the reasons they are useful are not the
same.

## Proposed Signal Taxonomy

The taxonomy should have one common shape and many domain profiles. In practice,
that means every retrieved chunk carries the same families of signals, but each
domain decides which signals are decision-critical, which are weak hints, and
which should only be shown as annotations.

The common schema is:

```text
Evidence unit
  query_text_relevance
  syntax_profile
  evidence_role
  source_trust
  corroboration
  temporal_validity
  domain_fit
  provenance
  safety_risk
  decision
```

This lets CANON say more precise things than "this chunk scored 0.82":

- relevant but unsafe
- relevant but low-trust
- high-trust but not relevant
- useful as a method detail, not as a conclusion
- useful as market narrative, not as verified fact
- strong for historical context, weak for current state
- admissible for inspection, not allowed into generator context

### 1. Query-Text Relevance

Purpose: decide whether a chunk is about the user's question.

Core signals:

- lexical match: BM25, exact phrase match, named entities, dates, jurisdiction,
  ticker/company names, author names, statute names
- semantic match: dense retrieval, cross-encoder reranking, query-document
  entailment
- syntax/form match: whether the retrieved text has the evidence form the query
  appears to ask for, such as method detail, limitation, quantitative result,
  legal authority, public expression, transcript/dialogue, risk disclosure,
  definition, or narrative context
- focus coverage: how many non-generic query terms appear in title, abstract,
  headings, or body
- negative constraints: "excluding", "not", "except", date ranges, geography,
  sector, population, court, market, field
- query intent: factual, comparative, causal, disagreement-seeking,
  source-audit, recency-sensitive, method-focused, legal, investment,
  historical, safety-critical

Universal decision rule:

- Relevance is usually a floor. A trustworthy source should not be allowed to
  rescue irrelevant text.
- Syntax relevance is not semantic relevance and not quality. It should help
  find the right kind of evidence, but it should not by itself certify that the
  evidence is true, reliable, or safe.

Domain differences:

- Economics and psychology: variable names, interventions, populations, methods,
  outcomes, and time windows matter.
- History and anthropology: place, period, archive, actors, language, and
  interpretive school matter.
- Legal review: jurisdiction, court, procedural posture, date, party, document
  type, and authority level matter.
- Investor/market review: company, market, segment, geography, reporting period,
  supplier/customer relation, and materiality matter.
- Web/grey literature: source identity and publication context matter because
  topical relevance alone is too easy to manipulate.

Current implementation:

- `canon/retrieval/syntax.py` builds a deterministic syntax vector for each
  retrieved chunk over `method_detail`, `limitation_caveat`,
  `quantitative_result`, `legal_authority`, `public_expression`,
  `dialogue_or_transcript`, `risk_disclosure`, `narrative_context`,
  `definition_or_taxonomy`, and `generic_claim`.
- The query gets its own syntax target vector. Candidate retrieval now mixes
  lexical, semantic, syntax, and focus signals, while final reranking exposes
  `syntax_relevance` as an independent score component.
- The corpus profile computes category counts and rarity weights over the
  current corpus. This is a corpus-learned prior over form, not a supervised
  syntax model.
- Retrieval traces include `decision.stage.signals.syntax_profile`, including
  the document syntax category, document syntax vector, query syntax targets,
  and corpus syntax profile.

How this should change by domain:

- Economics and psychology should often boost method, sample, result,
  limitation, and replication-related forms for evidence-audit queries.
- Anthropology, cultural studies, and history should often annotate narrative,
  archive, fieldwork, translation, positionality, and interpretive-frame forms
  rather than force them into a universal quality hierarchy.
- Legal review should treat legal authority, jurisdiction, procedural posture,
  holdings, and adverse authority as specialist syntax/evidence forms, with
  source authority and current-law gates doing the real trust work.
- Investor and market review should separate quantitative result, risk
  disclosure, management statement, customer evidence, competitor signal, and
  public-expression forms. Syntax can help route evidence, but filings,
  independence, recency, and materiality decide how much to trust it.
- Social/public-opinion analysis should usually keep public-expression syntax
  as aggregate-only unless sampling, collection legality/terms, privacy, bot
  risk, and representativeness checks are satisfied.

### 2. Evidence Role

Purpose: decide what kind of contribution the text makes.

Core roles:

- claim: conclusion, assertion, thesis, effect direction
- method: sample, design, model, identification, source base, measurement
- result: estimate, qualitative finding, observation, case outcome
- limitation: caveat, bias, missingness, scope condition, uncertainty
- definition: term, construct, doctrine, category, metric
- context: background, literature review, historical framing
- contradiction: explicit disagreement, failure to replicate, alternative
  interpretation
- provenance: where data or documents came from
- instruction-like text: text that looks like commands to the model or user

Universal decision rule:

- Importance is task-conditional. A methods passage can outrank a conclusion
  when the query asks whether evidence is credible. A limitation can outrank a
  finding when the query asks what can go wrong.

Domain differences:

- Economics: method, identification, robustness, data construction, and
  external validity are high-value roles.
- Psychology: measurement validity, sample, power, preregistration, replication,
  and exclusions are high-value roles.
- Anthropology: fieldwork context, positionality, informant/source relation,
  ethics, translation, and interpretive framing are high-value roles.
- Cultural studies: theoretical frame, corpus selection, interpretive move,
  citation context, and positional critique are high-value roles.
- History: primary-source provenance, archival citation, chronology,
  historiographic disagreement, and translation/edition notes are high-value
  roles.
- Legal review: holding, rule, dicta, procedural posture, standard of review,
  jurisdictional qualifier, and adverse authority are high-value roles.
- Investor review: revenue driver, risk factor, management claim, customer
  evidence, competitor signal, regulatory exposure, and forward-looking caveat
  are high-value roles.

### 3. Source Trust

Purpose: estimate how much evidentiary weight the source deserves.

Core signals:

- integrity: retraction, correction, expression of concern, fraud notice,
  supersession, withdrawal
- bibliographic provenance: DOI, stable URL, archive ID, court docket,
  regulator filing ID, publisher metadata, version history
- publication channel: peer-reviewed journal, book, working paper, preprint,
  policy report, corporate filing, court filing, website, comment, scraped page
- transparency: methods, data, code, materials, source corpus, appendix,
  funding/conflict statements
- study design or evidence design: experiment, quasi-experiment, observational
  study, ethnography, archival study, doctrinal analysis, case study, market
  survey
- influence: field/year-normalized citations, citation-network position,
  downstream use
- author/institution signal: separated from source trust because it can encode
  prestige and network effects
- accessibility: open access, PDF URL, readable full text, OCR quality

Universal decision rule:

- Integrity failures are gates. Access fields are provenance/access signals, not
  reliability signals. Citation and author prominence are influence signals, not
  truth signals.

Domain differences:

- Economics: working papers may be important early signals but should be marked
  as less settled than peer-reviewed or replicated evidence.
- Psychology: preregistration, replication, sample transparency, open materials,
  and measurement validity should matter more than raw citation counts.
- Anthropology and cultural studies: peer review and source transparency matter,
  but "replication" is not always the right criterion; interpretive rigor,
  field/context disclosure, and ethical transparency matter more.
- History: primary-source chain, archive quality, edition, translation,
  chronology, and historiographic reception matter more than recency.
- Legal review: authority hierarchy and current legal status dominate academic
  prestige. A recent binding case can outrank a famous old article.
- Investor review: official filings, audited financials, regulator actions, and
  first-party statements have different trust semantics than sell-side reports,
  blogs, forums, or scraped competitor pages.

### 4. Corroboration And Conflict

Purpose: avoid treating one retrieved chunk as settled truth.

Core signals:

- independent support count
- contradiction count
- source independence: whether sources cite the same original claim or share
  authors/institutions/data
- methodological diversity
- temporal pattern: older claim, newer update, reversal, replication,
  non-replication
- cluster position: mainstream, minority, bridge, isolated, contested

Universal decision rule:

- CANON should distinguish support volume from independent support. Ten sources
  repeating one original claim are not the same as ten independent confirmations.

Domain differences:

- Economics and psychology: replication and independent datasets matter heavily.
- Anthropology/history/cultural studies: corroboration can mean convergence
  across archives, witnesses, cases, or interpretations, not identical
  statistical replication.
- Legal: conflict means split authority, contrary precedent, jurisdictional
  mismatch, or overruled/superseded law.
- Investor review: conflict may be exactly what the user wants: management says
  one thing, customers, competitors, filings, and market data imply another.

### 5. Temporal Validity

Purpose: decide whether a source is still current enough for the task.

Core signals:

- publication date
- update date
- legal effective date
- filing period
- data collection period
- event date
- whether newer evidence supersedes older evidence

Universal decision rule:

- Recency is not always better. It depends on query intent.

Domain differences:

- Economics and market intelligence: recency can matter strongly for live market
  conditions.
- Psychology: recent replication or meta-analysis can supersede older famous
  findings.
- History: older primary documents can be more valuable than recent summaries.
- Legal: current status and jurisdiction matter more than publication age alone.
- Anthropology/cultural studies: recency may matter for current discourse, but
  older canonical texts may remain central.

### 6. Provenance And Corpus Risk

Purpose: decide how much caution to attach to the route by which the text
entered CANON.

Core signals:

- source mode: scholarly index, publisher, court docket, regulator, company
  filing, curated archive, web scrape, user upload, comment/social source
- extraction route: API metadata, PDF text, OCR, HTML scrape, transcript,
  manual upload
- versioning: stable version, mutable page, unknown update history
- extraction quality: OCR confidence, parse errors, boilerplate ratio,
  duplicated text, missing sections
- chain of custody: known origin, mirrored source, unknown origin

Universal decision rule:

- Provenance should affect trust and safety, but it should also be shown to the
  user. Unknown provenance should not be silently treated as neutral.

Domain differences:

- Academic corpora can rely on Crossref/OpenAlex/Semantic Scholar-style
  metadata, but should still track missingness.
- Legal and investor corpora need stronger document-chain provenance.
- Web and grey literature need higher default safety risk and stronger
  extraction diagnostics.

### 7. Safety And Prompt-Injection Risk

Purpose: prevent retrieved text from controlling the model or causing unsafe
actions.

Core signals:

- instruction override: "ignore previous instructions", "disregard above"
- role confusion: "system:", "developer:", XML/Markdown role tags
- exfiltration: secrets, credentials, tokens, hidden prompts, private files
- tool control: run commands, browse, click, email, buy, upload, download
- hidden text: comments, CSS hiding, tiny text, white text, unusual markup
- Unicode/control anomalies
- retrieval-trigger repetition or keyword stuffing
- suspicious provenance: scraped page, comment, user upload, unknown PDF

Universal decision rule:

- Safety is a gate, not a normal ranking signal. Unsafe text can be retrieved
  for audit, but it should not enter generator context unless explicitly
  sanitized and allowed.

Domain differences:

- Academic PDF corpora have lower baseline injection risk but still need checks
  for OCR artifacts and malicious text in uploaded PDFs.
- Web scraping and social sources have high baseline risk.
- Legal discovery and investor diligence can contain adversarial text by
  design, so the system must preserve evidence for inspection while preventing
  model obedience to embedded instructions.

### 8. Decision Layer

Purpose: convert signal vectors into explicit actions.

Recommended decisions:

- `retrieve`: eligible for candidate pool
- `rerank`: eligible for utility ranking
- `allow_context`: can enter generator context
- `sanitize_context`: factual content may enter after instruction-like spans are
  removed
- `quarantine`: excluded from generator context but shown in safety/audit report
- `block`: excluded and treated as a safety event
- `annotate`: usable, but with trust/quality warnings
- `abstain`: evidence is too weak, conflicted, stale, unsafe, or irrelevant

Universal decision rule:

- The final system should not only return top-k. It should return an evidence
  packet plus an explanation of what was allowed, downgraded, quarantined, or
  missing.

## Domain Profiles

The table below sketches how the same taxonomy should change by domain. These
are starting priors, not fixed laws.

| Domain | High-value evidence roles | Strong trust signals | Weak or dangerous shortcuts | Special gates |
| --- | --- | --- | --- | --- |
| Economics | identification, data, model, robustness, external validity | peer review, field/year-normalized influence, data/code, credible design, replication | raw citations, famous authors, working paper prestige | retraction, unavailable method/data for empirical claims, stale market data |
| Psychology | preregistration, measurement, sample, power, exclusions, replication | preregistration, open materials, validated measures, replication, risk-of-bias clarity | p-value language, sample size without design quality, citation count | retraction, failed integrity notices, missing method for causal claims |
| Anthropology | fieldwork context, positionality, informant/source relation, ethics, translation | peer review, field transparency, archive/interview provenance, ethical disclosure | forcing replication criteria, citation count, institution prestige | ethical concern, missing provenance for sensitive claims |
| Cultural studies | theoretical frame, corpus selection, interpretive move, citation context | peer review, transparent corpus, interpretive rigor, historiographic/theoretical placement | treating interpretive disagreement as factual failure, citation count | missing corpus description, unsafe scraped content |
| History | primary-source provenance, chronology, archival citation, historiographic dispute | archive quality, source chain, edition/translation clarity, peer review | recency as default, citations as truth, summaries without primary-source trail | forged/suspect source, untraceable quotation |
| Legal | holding, rule, dicta, procedural posture, jurisdiction, adverse authority | binding authority, current status, docket/source provenance, citator status | academic prestige, semantic similarity without jurisdiction, old overruled cases | overruled/superseded authority, wrong jurisdiction for legal conclusion |
| Investor/market | filings, risk factors, customer evidence, unit economics, regulatory exposure, management claims | audited filings, regulator data, primary documents, timestamped market data, source independence | SEO pages, promotional reports, anonymous claims, stale data | market-moving unsupported claims, prompt injection, unknown provenance |
| Web/grey literature | factual assertion, source identity, document provenance, narrative position | institutional source, stable URL, update history, citations/links, author identity | topical match, polished prose, virality, backlink count | injection risk, hidden text, content farm signals |

## How This Should Work In CANON

The implementation now has all seven stages of this first pipeline:

```text
query
  -> broad candidate retrieval        [implemented]
  -> structured chunk enrichment      [implemented]
  -> safety screening before rerank   [implemented]
  -> evidence utility reranking       [implemented]
  -> evidence packet assembly         [implemented]
  -> grounded generation payload      [implemented]
  -> post-generation verification     [implemented]
```

For each chunk, CANON should compute:

```text
relevance_vector
evidence_role_vector
source_trust_vector(domain)
corroboration_vector
temporal_validity_vector
provenance_vector
safety_risk_vector
```

Current Stage 1 candidate retrieval is implemented in
`canon/retrieval/stages.py::retrieve_candidates`. It builds a broad candidate
pool using lexical relevance, semantic similarity, and query-focus coverage
before any utility ranking or source-trust boosting happens.

Current Stage 2 structured enrichment is implemented in
`canon/retrieval/stages.py::enrich_candidates`. It emits structured sections for
`query_text_relevance`, `evidence_role`, `source_trust`, and `provenance` for
each candidate.

Current Stage 3 safety gating is implemented in
`canon/retrieval/stages.py::apply_safety_gate`. It runs before reranking and
rejects quarantined or blocked candidates from the generator-facing retrieval
set. Retrieval reports now include `stage_summary` and `rejected_candidates` so
blocked-but-relevant items are auditable rather than invisible.

Current Stage 4 evidence utility reranking is implemented through
`canon/retrieval/decisions.py::evidence_utility` and
`apply_evidence_decision`. It reports utility metadata including base score,
role bonuses, trust-relevance multiplier, and post-gate score.

Current Stage 5 context assembly is implemented in
`canon/evidence/context.py::assemble_context_packet`. It builds a balanced
packet of direct evidence, method detail, limitations/caveats, corroborating
sources, contradicting sources, and background while excluding unsafe evidence.

Current Stage 6 citation-disciplined generation is implemented in
`canon/synthesis/prompting.py`. It creates a structured generation payload where
retrieved evidence is represented as data with `citation_id`, `role`, and
`allowed_use`, plus citation rules that tell the model not to treat evidence
text as instructions.

Current Stage 7 post-generation verification is implemented in
`canon/synthesis/verification.py::verify_generation`. It checks citation
validity, disallowed citations, blocked/quarantined evidence leakage, and
unsupported factual sentences.

Then it should apply domain/use-case rules:

```text
if safety_decision in {quarantine, block}:
    exclude from generator context
if integrity_gate == false and query is not about integrity/retraction:
    exclude from normal evidence
if relevance < domain_profile.relevance_floor:
    do not let trust or prestige rescue the chunk
if evidence_role matches query intent:
    add utility bonus
if trust_tier is low:
    allow only with warning or as narrative/adversarial evidence
if independent_support is weak:
    hedge or abstain
```

The important design point: the same signal can have different decision use by
domain. For example, recency is often a boost in market analysis, sometimes a
gate in law, and often only an annotation in history. Replication is central in
psychology, useful in economics, and often the wrong vocabulary for cultural
studies or anthropology. Primary-source provenance is helpful everywhere, but it
is existential in history, law, and investor diligence.

## Literature-Backed Design Principles

### Quality Is Not Access

Open access, PDF availability, and a source name are useful for provenance and
retrievability, but they are not reliability signals by themselves. CANON now
keeps these as `accessibility` or bibliographic metadata rather than treating
them as direct source quality.

The stronger quality concepts come from evidence appraisal: risk of bias,
reporting transparency, publication integrity, study design, missingness,
selective reporting, and whether conclusions are supported by the methods and
data [1]-[6].

### Trust Requires Domain Profiles

There should be a shared schema, but multiple profiles:

- `economics`: empirical design, identification, data transparency, working
  paper versus journal status, recency, replication files.
- `psychology`: preregistration, power/sample transparency, measurement
  validity, replication status, open data/code, risk of selective reporting.
- `anthropology`: fieldwork transparency, positionality, archive/interview
  provenance, interpretive depth, ethics, peer review.
- `cultural_studies`: theoretical contribution, interpretive rigor, source
  corpus transparency, citation context, disciplinary venue.
- `history`: primary-source provenance, archival traceability, historiographic
  position, edition/translation issues, peer review.
- `legal_market`: jurisdiction, authority hierarchy, filing status, document
  provenance, date validity, source chain, adversarial posture.
- `web_grey_literature`: author/institution identity, publication incentives,
  update history, links/citations, content farm markers, safety risk.

The implementation added `SourceTrustProfile` in
`canon/quality/signals.py` so the same source can be interpreted differently by
domain.

### Importance Is Evidence Utility, Not Assertion Density

RAG often overvalues confident summaries. For academic and professional use
cases, important text may be a method, limitation, measurement note, null result,
definition, contradictory finding, caveat, jurisdictional qualifier, or data
provenance statement.

CANON now adds text-role signals in `canon/ingest/chunker.py`:

- `claim_density`
- `method_signal`
- `limitation_signal`
- `evidence_specificity`
- `uncertainty_signal`
- `section_role`

This is still heuristic, but it moves the system away from "conclusion-like text
is important" and toward "task-relevant evidence roles are important."

### Chunking Is A Trust And Safety Decision

Chunking should not be treated as a neutral preprocessing parameter. The chunk
is the unit that gets embedded, retrieved, scored, safety-screened, cited, and
possibly shown to the generator. A bad chunk can hide a caveat from a claim,
merge a useful source sentence with a prompt-injection sentence, or detach a
legal exception from the rule it qualifies.

CANON therefore treats chunking as a typed decision:

- atomic evidence unit: the smallest unit allowed to be retrieved, gated, and
  cited
- parent context: nearby text that can be added only when the source type and
  safety gate allow expansion
- aggregate unit: public-opinion/social text that may support aggregate analysis
  but should not be used as direct generator context
- audit-only unit: text preserved for inspection, prompt-injection review, or
  source-quality diagnosis

The current implementation is deterministic:

- Academic/preprint/policy/market reports use sentence-boundary evidence
  windows.
- News/open-web text preserves paragraph boundaries before sentence grouping.
- Legal and regulatory material uses smaller authority/filing windows with
  minimum overlap.
- Social/forum public-opinion text uses atomic sentence-level chunks with zero
  overlap so a useful opinion sentence can survive while an adjacent malicious
  instruction is quarantined.
- Archival/OCR and unknown unstructured text use conservative windows because
  extraction quality and boundaries are fragile.
- Web and unknown unstructured text use a safety override: when
  instruction-like sentences are detected, the chunker falls back to atomic
  sentence chunks before retrieval ranking. This keeps benign web claims,
  warranty statements, or market assertions from being fused with adjacent
  prompt-injection text.

The benchmark now measures two different chunking obligations:

- evidence-span containment: text that should stay together is preserved inside
  at least one chunk
- forbidden-merge avoidance: text that should remain separate, especially
  benign evidence and instruction-like attack text, is not fused into one
  generator-facing evidence unit

CANON also now has a chunking-variant benchmark. It compares fixed token
windows, the production boundary-aware policy, atomic sentence chunks, a lexical
semantic-cohesion surrogate, local hashed embedding-semantic grouping, and
paragraph-or-sentence chunking under the same fixture cases. The important
result is methodological: a large fixed window can score perfectly on span
containment because it contains everything, while still failing safety by
merging benign evidence with adjacent attack text. The variant report therefore
tracks balanced utility and safety-first objectives in addition to span
preservation.

The current local embedding-semantic variant is a real technical step beyond
raw lexical overlap: it embeds adjacent sentences with CANON's hashed semantic
encoder and breaks chunks when adjacent semantic similarity is low or
instruction-like text appears. In the current fixture report it avoids forbidden
merges and unsafe contamination like the safer atomic variants, but still trails
the production boundary-aware policy on balanced utility because it fragments
more. That is useful evidence, not a final verdict: local hashed embeddings
exercise the semantic-vector path, but they are not equivalent to neural
embedding models or late chunking.

The benchmark also feeds a human-review task family. Reviewers are asked to
label:

- `chunking_harmful_merge`: whether the candidate policy fused text that should
  remain separate
- `qualifier_preservation`: whether caveats, jurisdictional limits, dates,
  scope conditions, or methods stayed with the relevant claim
- `fragmentation`: whether the policy under-fragmented, over-fragmented, or
  produced an appropriate evidence unit
- `preferred_chunking_variant`: which variant should be used for that case

This is deliberately not one scalar "chunk quality" score. Harmful-merge
avoidance can gate or quarantine content, qualifier preservation can change
evidence usefulness, and fragmentation can be a utility/cost tradeoff. The
current deterministic labels are proxy diagnostics only; the evaluation anchor
`human_chunking_quality_labels_v1` now blocks any claim that the chunking policy
is scientifically validated before human labels exist.

This is not the final chunking model. Recent work points to stronger variants:
late chunking embeds longer context before pooling chunk representations [18],
contextual retrieval and reconstruction methods try to preserve document-level
meaning around local chunks [19], RAPTOR builds hierarchical summary trees for
multi-hop questions over long documents [20], GraphRAG uses graph/community
summaries for corpus-level sensemaking [17], and chunking-free/contextual
retrieval methods try to locate evidence spans without committing to fixed
pre-split chunks [21].

The scientific path is to compare these methods under the same metrics:

- evidence-span containment
- caveat/qualifier preservation
- retrieval precision and recall
- citation faithfulness
- prompt-injection reach rate
- benign usefulness retained after safety gates
- human-rated answer utility
- latency and indexing cost

The expected production design is multi-granular rather than one chunk size:
retrieve atomic chunks for precision and safety, optionally expand to parent
context for interpretation, and use graph/tree summaries for broad questions
about themes, conflicts, and market or legal narratives.

### Linear Scores Are Useful Diagnostics, Not Final Decisions

Linear scalarization remains useful for comparison and traceability, but it is
the wrong final abstraction for trust and safety. Some dimensions are boosts,
some are annotations, and some are gates.

CANON now adds a non-linear retrieval decision layer in
`canon/retrieval/decisions.py`:

- Retraction blocks ordinary retrieval unless the query is about retraction or
  correction.
- Safety quarantine/block decisions prevent malicious chunks from winning
  retrieval even when they are lexically relevant.
- Low relevance limits how much source trust can help.
- Method, limitation, and specificity can act as task-sensitive utility bonuses.

### Prompt Injection Must Be Tested As A Pipeline Failure

Indirect prompt injection is not just a generation problem. Retrieved text can
attack the system during ingestion, retrieval, reranking, context assembly, or
generation [9]-[14]. The right metric is not only "did the model obey the bad
instruction?" but "did the attack reach the generator?"

CANON now has a first deterministic scanner in `canon/safety/scanner.py` for:

- instruction override
- role confusion
- exfiltration intent
- tool-control intent
- hidden text markers
- Unicode/control-character anomalies
- retrieval-trigger repetition

This is not a complete defense. It is the first measurement layer. The next
step is an adversarial corpus where malicious chunks are relevant enough to be
retrieved, then the system measures whether they are blocked before generation.

## What Changed In Code

### Source Trust

New function:

```python
source_trust_vector(work, domain="general_academic")
```

The vector separates:

- `integrity_gate`
- `retraction_risk`
- `bibliographic_provenance`
- `publication_channel`
- `reporting_transparency`
- `open_science`
- `field_normalized_influence`
- `author_influence`
- `accessibility`
- `trust_tier`
- `source_trust`

`source_quality_vector` remains for legacy diagnostics, but new work should use
`source_trust_vector`.

### Text Importance

Chunk metadata now includes richer evidence-role signals. This lets retrieval
and reports distinguish a claim from a method detail or limitation. The current
implementation is deterministic and deliberately simple; it should become a
labeled evaluation problem before it is trusted.

Chunking quality is now part of that labeled evaluation problem. The
unstructured task pack includes chunking-variant review cases, and the
calibration report creates proxy labels for harmful merges, qualifier
preservation, fragmentation, and preferred variant while keeping status at
`needs_labels`.

### Non-Linear Decision Layer

`apply_evidence_decision()` now sits after the weighted base score. It can reject
or penalize chunks based on integrity, safety, relevance floors, and low trust.
This is a better shape than pretending all concerns are commensurable ranking
weights.

### Corroboration And Adversarial Echo Testing

`assess_corroboration()` in `canon/evidence/corroboration.py` now groups
evidence by normalized claim keys and reports:

- `support_count`
- `independent_support_count`
- `contradiction_count`
- `dependent_echo_count`
- `source_fingerprints`
- `echo_risk`
- group and overall verdicts

This directly tests the adversarial failure mode where several retrieved
documents repeat the same claim from one source chain. The first benchmark is
`canon.eval.corroboration`, which writes
`reports/adversarial_corroboration_v1.json`. It includes:

- a mixed case with independent support, company-source echoes, and a
  contradicting source
- a pure echo attack where four sources repeat a vendor-originated claim

The synthesis support model now uses corroboration to downgrade confidence when
support is single-source/dependent or when echo risk is high.

### Prompt-Injection Screening

`scan_text()` returns:

- `risk_score`
- `decision`: `allow`, `sanitize`, `quarantine`, or `block`
- `categories`
- `matched_patterns`

For now, quarantined and blocked chunks are rejected from retrieval ranking.
Later, they should also be reported separately as safety events.

## Recommended Next Scientific Steps

1. Add domain-specific corpus packs for economics, psychology, anthropology,
   cultural studies, and history.
2. Add a `domain` field to corpus manifests and propagate it into
   `source_trust_vector`.
3. Build a labeled evidence-role set across disciplines.
4. Add source-trust labels: strong, moderate, limited, low, blocked.
5. Add chunking-quality labels for harmful merges, qualifier preservation,
   fragmentation, and preferred policy across domains and document types.
6. Build an adversarial RAG security fixture with clean and poisoned versions of
   the same corpus.
7. Track attack reach rate: ingested, retrieved, reranked, exposed to generator,
   cited, and obeyed.
8. Add grey-literature profiles before ingesting web scraping at scale.
9. Separate use-case profiles for scholarly review, investor diligence, legal
   document review, and market intelligence.

## Open Research Questions

1. How much of source trust can be inferred from metadata versus full text?
2. Can one evidence-role classifier generalize across social sciences and
   humanities?
3. How should CANON score interpretive rigor in fields where replicability is
   not the dominant quality criterion?
4. When should low-trust evidence be excluded, and when should it be surfaced as
   a warning because it reflects market narratives or adversarial positions?
5. How can prompt-injection screening avoid suppressing legitimate texts about
   security, law, or AI safety?
6. Which chunking failures should be hard gates, which should be annotations,
   and which should become domain-specific utility tradeoffs?

## References

[1] M. J. Page et al., "The PRISMA 2020 statement: an updated guideline for
reporting systematic reviews," *BMJ*, vol. 372, 2021, Art. no. n71. Available:
https://www.prisma-statement.org/prisma-2020

[2] J. P. T. Higgins, J. Savović, M. J. Page, R. G. Elbers, and J. A. C.
Sterne, "Chapter 8: Assessing risk of bias in a randomized trial," in
*Cochrane Handbook for Systematic Reviews of Interventions*, version 6.5,
Cochrane, 2024. Available:
https://www.cochrane.org/authors/handbooks-and-manuals/handbook/current/chapter-08

[3] J. A. C. Sterne et al., "ROBINS-I: a tool for assessing risk of bias in
non-randomised studies of interventions," *BMJ*, vol. 355, 2016, Art. no.
i4919. Available: https://www.riskofbias.info/welcome/robins-i-v2

[4] G. H. Guyatt et al., "GRADE: an emerging consensus on rating quality of
evidence and strength of recommendations," *BMJ*, vol. 336, no. 7650,
pp. 924-926, 2008.

[5] J. P. A. Ioannidis, "Why most published research findings are false,"
*PLoS Medicine*, vol. 2, no. 8, 2005, Art. no. e124.

[6] B. A. Nosek et al., "Promoting an open research culture," *Science*,
vol. 348, no. 6242, pp. 1422-1425, 2015.

[7] N. Thakur et al., "BEIR: A heterogeneous benchmark for zero-shot evaluation
of information retrieval models," in *Proc. NeurIPS Datasets and Benchmarks*,
2021.

[8] S. Es et al., "RAGAS: Automated evaluation of retrieval augmented
generation," arXiv:2309.15217, 2023.

[9] K. Greshake et al., "Not what you've signed up for: Compromising real-world
LLM-integrated applications with indirect prompt injection," arXiv:2302.12173,
2023.

[10] OWASP, "OWASP Top 10 for Large Language Model Applications," 2025.
Available: https://owasp.org/www-project-top-10-for-large-language-model-applications/

[11] Q. Zhan, Z. Liang, Z. Ying, and D. Kang, "InjecAgent: Benchmarking
indirect prompt injections in tool-integrated large language model agents,"
arXiv:2403.02691, 2024.

[12] G. De Stefano, L. Schönherr, and G. Pellegrino, "Rag and roll: An
end-to-end evaluation of indirect prompt manipulations in LLM-based application
frameworks," arXiv:2408.05025, 2024.

[13] H. Guo and Z. Wei, "Hidden-in-Plain-Text: A benchmark for social-web
indirect prompt injection in RAG," arXiv:2601.10923, 2026.

[14] J. Lee, H. Jang, and K. S. Choi, "MPIB: A benchmark for medical prompt
injection attacks and clinical safety in LLMs," arXiv:2602.06268, 2026.

[15] S. Asai et al., "Self-RAG: Learning to retrieve, generate, and critique
through self-reflection," arXiv:2310.11511, 2023.

[16] Y. Yan et al., "Corrective retrieval augmented generation,"
arXiv:2401.15884, 2024.

[17] Microsoft Research, "GraphRAG: From local to global search," arXiv:2404.16130,
2024.

[18] M. Gunther, I. Mohr, D. J. Williams, B. Wang, and H. Xiao, "Late chunking:
Contextual chunk embeddings using long-context embedding models,"
arXiv:2409.04701, 2024.

[19] C. Merola and J. Singh, "Reconstructing context: Evaluating advanced
chunking strategies for retrieval-augmented generation," arXiv:2504.19754,
2025.

[20] P. Sarthi, S. Abdullah, A. Tuli, S. Khanna, A. Goldie, and C. D. Manning,
"RAPTOR: Recursive abstractive processing for tree-organized retrieval,"
arXiv:2401.18059, 2024.

[21] H. Qian, Z. Liu, K. Mao, Y. Zhou, and Z. Dou, "Grounding language model
with chunking-free in-context retrieval," arXiv:2402.09760, 2024.
