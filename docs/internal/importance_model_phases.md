# Importance Model Phases

Date: 2026-07-04

This document tracks the implementation path for CANON's importance, trust,
corroboration, safety, and evaluation model. The current implementation is
deterministic by design. That is useful for auditability and regression testing,
but it is not the final scientific model.

The guiding rule is:

> Use deterministic methods to make the pipeline explicit, measurable, and
> testable; then compare them against stronger learned, human-labeled, and
> external-benchmark methods.

## Global Deterministic Limitations

The current methods are intentionally conservative and transparent, but they
have major limits:

- Marker-based text-role detection misses subtle methods, limitations,
  theoretical moves, and domain-specific evidence roles.
- Source trust is metadata-driven and cannot yet fully judge methodological
  validity, interpretive rigor, legal authority, or market reliability.
- Prompt-injection screening is pattern-based and will miss novel attacks,
  multilingual attacks, and context-dependent instructions.
- Corroboration groups use lexical/metadata fingerprints and can over-split
  paraphrases or under-detect hidden source dependence.
- Post-generation verification is lexical/citation-structural, not a full NLI
  or expert fact-checking system.
- Domain profiles are starting priors, not validated disciplinary standards.

Every deterministic score should therefore be treated as a diagnostic feature,
not as truth.

## Rule: Retrieved Text Cannot Instruct The Model

Current rule:

> Retrieved text is evidence data. It cannot issue instructions to the model.

This is correct for the current academic/web/grey-literature RAG pipeline.
However, future exceptions may be needed for controlled tasks where the retrieved
text is explicitly the object of instruction following, such as:

- reading a contract clause that instructs a party to perform an action
- analyzing a prompt-injection example in a security dataset
- following a user-selected procedural checklist from a trusted internal manual
- executing a legal or compliance workflow where authoritative text prescribes
  steps

Any future exception must be explicit and gated:

- source must be trusted or user-authorized
- instruction scope must be constrained
- no secrets, tools, browsing, shell, email, purchase, or external action may be
  triggered directly by retrieved text
- the generator must know whether it is analyzing the instruction, summarizing
  it, or following a bounded workflow
- exception use must be logged in the evidence decision

Until such an exception mechanism exists, retrieved instruction-like text should
be sanitized, quarantined, or blocked.

## Phase A: Refactor Importance Into Typed Decisions

Status: implemented as a deterministic first pass.

Implemented:

- `canon/retrieval/stages.py` exposes candidate, enrichment, and safety-gate
  objects.
- `canon/retrieval/decisions.py` exposes utility and retrieval decision
  metadata.
- `canon/evidence/signals.py` defines typed `EvidenceSignals`,
  `EvidenceDecision`, and conversion helpers for current retrieval results.
- `canon/evidence/domain.py` defines domain decision profiles for general
  academic, economics, psychology, anthropology, history, cultural studies, and
  legal/market use cases.
- `canon/eval/contracts.py` defines `canon_rag_contract_v1`, a validation layer
  for retrieval, synthesis, adversarial-security, corroboration, and anchor
  reports.
- `canon/eval/phase_gate.py` runs an end-to-end deterministic Phase A-G gate.
- Retrieval traces now expose `decision.stage.signals`, utility metadata, safety
  metadata, and candidate-stage metadata.
- `canon/eval/committee.py` adds a deterministic evidence committee over
  retrieval candidates, with separate relevance, source-trust, domain-fit,
  safety, corroboration, and public-opinion aggregation votes.
- Retrieval traces now include `syntax_profile`, and score components include
  `syntax_relevance`, so evidence form can be weighted separately from lexical
  relevance and semantic similarity.
- `committee_rag` adds an experimental hard-gated retrieval policy that applies
  the committee before final scoring; only candidates with `generator_context`
  in their allowed uses can enter selected final context, while excluded rows
  remain auditable through rejected-candidate traces and stage summaries.
- `canon/eval/committee_gate_comparison.py` adds a paired comparison between
  audit-only `rag` and hard-gated `committee_rag`, reporting selected-context
  policy-violation reduction, final-context retention, and abstention.
- `canon/eval/committee_gate_usefulness.py` converts committee-gate losses and
  abstentions into review tasks with explicit human-label fields for evidence
  usefulness, exclusion correctness, and abstention correctness.

Limitations:

- Type wrappers currently normalize existing deterministic dictionaries; they do
  not yet enforce a full schema at storage boundaries.
- The v1 contract validates required fields and unsafe state transitions, but it
  is not a full JSON Schema or database migration system.
- Decision confidence is rule-derived, not calibrated.
- Committee decisions are still deterministic and uncalibrated; the hard-gated
  `committee_rag` policy should be compared against audit-only `rag` for recall
  loss, answer usefulness, safety exposure, and aggregate-only abstention.

Future tests:

- Compare deterministic decisions against human labels.
- Add schema validation for every generated report.
- Add persisted schema migration tests when report storage becomes durable.
- Stress-test decision stability under query perturbations and adversarial
  chunk injection.
- Compare audit-only committee reports against hard-gated and learned-arbiter
  retrieval variants.
- Treat `committee_rag` retention/abstention as proxy metrics until answer
  usefulness and missed-evidence costs are measured with human or external
  anchors.
- Use the usefulness audit to prioritize human review of lost context before
  treating committee-gated retrieval as a default production policy.

## Phase B: Improve Text-Level Importance

Status: implemented as deterministic heuristics.

Implemented:

- `method_signal`
- `limitation_signal`
- `evidence_specificity`
- `uncertainty_signal`
- `claim_density`
- `section_role`
- corpus-profiled syntax relevance for method detail, limitation/caveat,
  quantitative result, legal authority, public expression, transcript/dialogue,
  risk disclosure, narrative context, definition/taxonomy, and generic claim
  forms
- boundary-aware chunking with deterministic token-window fallback
- chunk diagnostics for token count, relative source position, and
  boundary-aligned share
- document-type-aware chunking policies for academic/preprint, policy/market
  reports, news/web pages, legal authorities, regulatory filings,
  social/forum public opinion, transcripts, archival/OCR sources, and unknown
  unstructured text
- strategy-specific boundary enforcement: public social/forum text now uses
  atomic sentence-level chunks with zero overlap, and news/open-web text
  preserves paragraph boundaries before sentence grouping
- high-risk web/unknown text switches to atomic sentence chunks when
  instruction-like sentences are detected, so prompt-injection spans are
  isolated before retrieval ranking and safety gating
- chunk-policy metadata persisted on each chunk so retrieval and data-card
  reports can audit whether chunks came from fixed windows, clause/thread/OCR
  policies, or sentence-boundary policies
- deterministic parent-child chunk metadata through `atomic_chunk_id` and
  `parent_context_id`
- safety-gated parent context expansion for synthesis, where sibling context is
  reconstructed only when the chunk policy allows expansion and the assembled
  parent text passes safety screening
- deterministic chunking-strategy benchmark comparing fixed windows and
  boundary-aware chunks on academic, psychology, anthropology, history/OCR,
  legal, filing, market, news, web, social/public-opinion, and transcript cases
- forbidden-merge probes that test whether chunking accidentally joins text
  that should remain separate, such as benign web/social claims and adjacent
  prompt-injection instructions
- large-window chunking safety regression that verifies boundary-aware
  public-expression chunks still isolate prompt-injection text when a naive
  fixed-window configuration would merge benign and unsafe sentences
- deterministic chunking-variant benchmark comparing fixed token windows, the
  current boundary-aware policy, atomic sentence chunks, a lexical
  semantic-cohesion surrogate, local hashed embedding-semantic grouping, and
  paragraph-or-sentence chunking on span preservation, harmful merge avoidance,
  unsafe-span isolation, fragmentation, balanced utility, and safety-first
  objectives
- chunking-quality human-review tasks derived from the variant benchmark, with
  labels for harmful merge, qualifier preservation, fragmentation, and preferred
  variant
- query-intent notes for method, limitation, conflict, current-state, and
  authority queries
- candidate retrieval now blends lexical relevance, hashed semantic similarity,
  corpus-profiled syntax relevance, and query-focus coverage; final policies
  can independently weight `syntax_relevance`
- the evidence committee treats strong syntax alignment as a partial relevance
  signal, but syntax cannot override source-trust or safety votes
- `reports/syntax_retrieval_ablation_v1.json/.md` compares no-syntax,
  candidate-only syntax, and candidate-plus-rerank syntax variants. The current
  fixture shows neutral expected-evidence MRR deltas because the hand-authored
  anchors are already solved by lexical/semantic matching; this is recorded as
  a saturation limitation, not as validation.

Limitations:

- Heuristics are marker-based and English-centric.
- Syntax relevance is evidence-form detection, not reliability detection. It can
  find method-looking or legal-looking text without knowing whether that text is
  valid, binding, current, representative, or true.
- The current corpus profile learns category rarity with an IDF-style
  deterministic weight, not from human labels, external qrels, or neural syntax
  embeddings.
- Domain-specific syntax categories are still coarse. Legal, investor,
  ethnographic, historical, and public-opinion workflows need more specialized
  form detectors before syntax relevance should carry high decision weight.
- Boundary-aware chunking is sentence-based, not discourse-aware, semantic,
  table-aware, legal-clause-aware, or thread-aware.
- The semantic-cohesion chunker is only a lexical technical surrogate, and the
  embedding-semantic chunker uses local hashed embeddings; neither should be
  treated as equivalent to production neural embedding chunking, late chunking,
  or learned layout/discourse segmentation.
- Atomic public-expression chunking can split useful stance context from
  surrounding explanation; this is intentional for safety but must be measured
  as a possible usefulness loss.
- Atomic safety overrides for web/unknown text can also split benign explanatory
  context; this should be compared against later span-level sanitizers,
  model-based safety classifiers, and human utility labels.
- Paragraph-first news/web chunking depends on extraction quality; bad HTML/PDF
  parsing can erase paragraph boundaries and force weaker sentence or token
  fallback.
- Parent-context expansion is deterministic and sibling-based; it does not yet
  use learned semantic parents, layout-aware PDF structure, table boundaries, or
  discourse trees.
- Document-type-aware policy is metadata-driven; wrong or missing document-type
  labels can choose the wrong chunking strategy.
- Token-window fallback can still split claims from qualifiers when sentence
  segmentation fails.
- They may overvalue formulaic terms and undervalue subtle reasoning.
- They are not yet domain-profile-specific enough for history, anthropology,
  cultural studies, legal evidence, or investor due diligence.

Future tests:

- Build a labeled sentence/chunk set across economics, psychology,
  anthropology, cultural studies, history, legal, market, and web/grey
  literature.
- Compare deterministic labels against expert labels and model-assisted
  classifiers.
- Compare deterministic syntax relevance against learned form classifiers,
  cross-encoders trained on query-evidence-role labels, discourse parsers,
  legal-clause classifiers, transcript role models, and market-document section
  classifiers.
- Ablate `syntax_relevance` across domains to measure when it improves recall,
  when it merely annotates usefully, and when it harms precision by rewarding
  formulaic text.
- Add harder syntax ablations with topical near-duplicate distractors, such as
  a semantically relevant news paragraph versus a transcript, a market blog
  versus a filing risk factor, and a public comment versus a factual source.
- Compare fixed windows, sentence chunks, semantic chunks, discourse chunks,
  legal-clause chunks, table-aware chunks, and social-thread chunks on retrieval
  precision, citation faithfulness, safety exposure, and answer usefulness.
- Compare the local hashed embedding-semantic baseline with external neural
  embedding chunking and late chunking, then run the same contract-backed report
  to see whether stronger methods improve utility without reintroducing harmful
  merges.
- Add chunk-level human labels for evidence sufficiency, missing qualifier,
  harmful merge, excessive fragmentation, and safe parent-context expansion.
- Add contextual-retrieval and late-chunking variants, then test whether
  hierarchical parent-child retrieval improves legal, investor, historical, and
  public-opinion workflows.
- Track false positives where words like "bias" or "method" are used casually.

## Phase C: Build A Real Source Trust Layer

Status: implemented as a deterministic metadata/provenance layer.

Implemented:

- `source_trust_vector`
- domain profiles for economics, psychology, anthropology, history, cultural
  studies, and legal/market contexts
- domain decision profiles that separate high-value roles, gates, annotation
  signals, recency policy, and corroboration policy
- unstructured document-type profiles for academic articles, preprints, policy
  reports, regulatory filings, legal authority, market reports, news, web pages,
  social-media posts, forum comments, transcripts, archival primary sources,
  and unknown text
- a `public_opinion` domain profile for public social/forum expression evidence
  with explicit aggregation and safety-gate requirements
- separation of source trust from accessibility
- retraction/integrity gate
- author influence kept separate from source trust

Limitations:

- Trust is not yet connected to Crossref, Retraction Watch/Crossmark, court
  citators, SEC filings, publisher registries, or validated predatory-source
  data.
- It cannot yet judge internal validity from full methods text.
- It cannot yet model legal authority hierarchy or market document incentives
  deeply.
- Public-opinion evidence is not yet representative; current social/forum
  signals only support fixture-level routing, aggregation requirements, and
  safety screening.
- Document-type detection is deterministic and metadata/text-marker based; it
  should be treated as a routing/profile signal, not a validated classifier.

Future tests:

- Add external metadata checks.
- Compare trust tiers against expert judgments.
- Add domain-specific trust labels and missingness reports.
- Add compliant social/public web collectors, bot/coordination detection,
  stance/theme labels, temporal burst features, and platform-sampling metadata.

## Phase D: Add Corroboration And Conflict-Aware Importance

Status: implemented as deterministic corroboration and echo-risk logic.

Implemented:

- `assess_corroboration`
- support count versus independent support count
- contradiction count
- dependent echo count
- source fingerprints
- echo-risk verdicts
- adversarial corroboration fixture

Limitations:

- Claim grouping is lexical and can split paraphrases.
- Source independence is inferred from metadata/fingerprints and can miss hidden
  dependence.
- Contradiction depends on coarse stance labels.

Future tests:

- Compare deterministic claim grouping with embedding/NLI grouping.
- Add source-chain metadata from citation graphs, filings, syndication, and
  press-release tracking.
- Build adversarial cases with paraphrases, multilingual echo, and synthetic
  source diversification.

## Phase E: Add Safety Screening Before Generation

Status: implemented as deterministic pre-rerank screening.

Implemented:

- instruction override detection
- role confusion detection
- exfiltration detection
- tool-control detection
- hidden text markers
- Unicode/control anomaly checks
- repetition/trigger-shape checks
- safety gate before reranking
- deterministic sanitization for `sanitize` decisions before context assembly
- explicit retrieved-instruction exception policy placeholder for future
  trusted, bounded workflows

Limitations:

- Pattern-based screening is incomplete.
- It may false-positive on legitimate security, legal, or AI-safety documents.
- It is not yet OCR/layout-aware for PDFs.
- Sanitization is regex/span based and can remove too much or too little; it is
  useful as an auditable baseline, not a complete defense.

Future tests:

- Compare against public prompt-injection benchmarks.
- Add multilingual and obfuscated attack variants.
- Add span-level sanitization and evaluate benign utility loss.

## Phase F: Build An Adversarial RAG Security Benchmark

Status: implemented as a deterministic first benchmark.

Implemented:

- `canon/eval/security.py`
- prompt-injection reach metrics
- benign retention metrics
- attack retrieval/candidate/exposure accounting
- hidden-text injection fixture
- public social-media/forum injection fixture
- report artifact target

Limitations:

- Current fixture is small and synthetic.
- It tests deterministic pattern attacks more than adaptive attacks.
- It does not yet evaluate a live LLM's attack-following behavior.
- It does not yet test HTML/PDF hidden text extraction failures.
- Social/public-opinion fixtures test ingestion and pre-rerank blocking, but do
  not yet test live platform collectors, bot networks, coordinated campaigns, or
  subtle instruction attacks.

Future tests:

- Add a larger poisoned corpus.
- Track attack path across ingestion, retrieval, reranking, context assembly,
  generation, citation, and action.
- Compare deterministic scanner with model-based and sanitizer-based defenses.

## Phase G: Upgrade Evaluation With Human And External Anchors

Status: implemented as a deterministic anchor registry; human and full external
anchors are still planned or limited.

Implemented:

- `canon/eval/anchors.py`
- anchor registry for human labels and external benchmarks
- active deterministic anchors for adversarial RAG security and adversarial
  corroboration
- hard-negative anchor preference set for prompt-injection gating,
  echo-count inflation, source-quality/accessibility confusion,
  conflict-aware synthesis, and recency/authority handling
- expanded human-label task schema for evidence role, source trust, safety, and
  corroboration handling, plus chunking-quality fields for harmful merge,
  qualifier preservation, fragmentation, and preferred variant
- stratified unstructured human-label task pack via
  `canon/labeling/unstructured_tasks.py`, covering document type, evidence
  role, source trust, source independence, safety disposition, answer
  usefulness, and chunking quality across domain/document-type strata plus
  committee-gate loss, abstention, and chunking-variant review cases
- human-vs-technical calibration report via `canon/labeling/calibration.py`
- pure technical calibration report via `canon/eval/technical_calibration.py`
- transparent ridge calibration model via
  `canon/modeling/calibration_model.py`
- transparent pairwise preference model via
  `canon/modeling/preference_model.py`
- anchor-source pairwise training over
  `canon/eval/hard_negatives.py`
- claim-limit rules that prevent overstating source-trust, evidence-role,
  chunking-quality, public benchmark, or prompt-injection validation
- self-validation under `canon_rag_contract_v1`
- report artifact target
- explicit limitation reporting for missing anchors

Limitations:

- No human-labeled evidence-role/source-trust/chunking-quality dataset exists
  yet.
- The unstructured task scaffold is ready for annotation, but the labels are
  intentionally empty and must not be treated as validation evidence.
- Technical calibration labels are proxy labels and must not be treated as
  expert ground truth.
- Hard-negative anchors are minimum expectation tests, not broad validation.
- Public benchmark use is still limited.
- Expert review workflow is not yet active.

Future tests:

- Build a human-labeled CANON evidence-role/trust set.
- Build a chunk-level human-label set for harmful merges, missing qualifiers,
  fragmentation, and preferred chunking policy by domain and document type.
- Add full public retrieval benchmark evaluation beyond smoke tests.
- Add legal/investor domain-specific validation sets if those use cases become
  first-class.
- Compare scalar calibration, pairwise preference modeling, and domain-specific
  models against the same human and external anchors.

## Unified Phase Gate

`canon/eval/phase_gate.py` now produces `reports/importance_phase_gate_v1.json`.
The report checks that:

- Phase A exposes versioned contracts and typed decisions.
- Phase B has text-importance and domain-role schema hooks.
- Phase C has source-trust anchors and domain profiles.
- Phase D passes the corroboration/echo-risk benchmark.
- Phase E keeps attack text out of context.
- Phase F includes retrieved-then-rejected adversarial attacks, including hidden
  text.
- Phase G exposes anchor/claim-limit contracts.
- Phase H exercises mixed unstructured document routing across academic, legal,
  market, policy, transcript, archival, news, and web-text fixtures.

This is a coverage gate, not a truth gate. Passing it means the implementation
surface exists and current deterministic fixtures pass. It does not mean the
system has validated source trust, validated evidence roles, or broad
prompt-injection robustness. It also does not mean heterogeneous document-type
classification is validated on real corpora.

`canon/reports/unstructured_portfolio.py` now adds a broader heterogeneous
unstructured portfolio artifact. It connects the phase gate with
heterogeneous-readiness checks, public-opinion aggregation boundaries,
committee decisions, chunking evaluation, adversarial security, public-opinion
synthesis context exclusion, and evaluation-anchor claim limits. This portfolio
is the current top-level artifact for the unstructured-data experiment, while
the older 10k portfolio remains focused on OpenAlex/social-science retrieval
diversity.

The portfolio contract now requires the large-window chunking probe as a
regression check. This prevents a report from passing on ordinary small-window
span containment while silently missing the safety failure where a broad token
window merges useful public-expression text with adjacent prompt-injection
instructions.

The portfolio also now treats chunking quality as a label and anchor family.
The deterministic variant benchmark can show that fixed windows fail on harmful
merges and that boundary-aware chunking wins the current safety-first objective,
but this remains a proxy result until chunk-level human labels confirm whether
the chosen chunks preserve enough context for real tasks.

`canon/eval/unstructured_matrix.py` adds the coverage matrix underneath that
portfolio. It is intentionally stricter than a pass/fail gate: the current
matrix status is expected to block broad validity claims while source-form
coverage remains fixture-only, preferred document types are still thin inside
some domain profiles, external corpora are missing, and human/domain labels
remain incomplete.
