# Quality And Importance Model

CANON separates signals that are often collapsed in ordinary RAG systems.

## Source-Level Signals

- venue signal
- retraction signal
- citation impact
- graph centrality
- author prominence
- open-science indicators
- publisher/source legitimacy

## Document-Type Signals

CANON now profiles unstructured text before treating it as evidence. The
document type is not a quality score. It is an interpretation context that
changes which risks and checks matter.

Current document types:

- academic article
- preprint
- policy report
- regulatory filing
- legal authority
- market report
- news article
- social media post
- forum comment
- web page
- transcript
- archival primary source
- unknown unstructured text

The profile emits:

- evidence family
- document-type confidence
- authority signal
- timeliness need
- source-intent risk
- adversarial surface
- extraction risk
- grey-literature flag
- recommended decision role

Corpus data cards now summarize these profiles with document-type counts,
evidence-family counts, recommended decision-role counts, grey-literature
share, unknown-type share, average authority signal, average timeliness need,
average source-intent risk, average adversarial surface, and average extraction
risk. These fields are used to prevent overclaiming when a corpus is really
single-type, mostly unknown, or high-risk open-web material.

Document-type slice diagnostics now measure how retrieval policies expose,
rank, and reject each document type. The diagnostic is deliberately descriptive:
it reports candidate counts, final exposure, average rank, and safety rejections
by type instead of converting type into a universal quality scalar.

## Domain-Fit Signals

CANON now adds a deterministic domain-fit layer on top of source trust and
document type. The common schema is shared, but each domain profile defines
different preferred document types, evidence families, critical quality
signals, gates, and annotation-only signals.

Current domain profiles:

- general academic
- economics
- psychology
- anthropology
- history
- cultural studies
- legal/market
- public opinion

The domain-fit vector emits:

- domain fit score
- domain fit tier
- document-type fit
- evidence-family fit
- authority fit
- transparency fit
- provenance fit
- publication-channel fit
- open-science fit
- safety fit
- gate flags
- preferred document types and evidence families
- critical quality signals
- required gates
- annotation-only signals
- recency and corroboration policies

These signals are not ranking truth. They are an audit layer that shows whether
retrieval is leaning on evidence that fits the declared domain. For example,
legal/market retrieval emphasizes authority, filing provenance, jurisdiction,
and timeliness; psychology emphasizes measurement, sample, preregistration,
replication, and open-science signals; history emphasizes primary-source
provenance, source criticism, and historiographic context.

Domain-slice diagnostics summarize exposure, average rank, domain-fit score,
and safety rejection by domain. This helps distinguish a system that works on
one academic corpus from one that can be stress-tested across heterogeneous
unstructured data.

Examples:

- A legal opinion should be gated by jurisdiction, date, and authority chain.
- A market filing can be authoritative for disclosed facts but still reflects
  strategic incentives.
- A public social post or forum comment should be treated as expression
  evidence. It can help identify themes, concerns, claims in circulation, and
  narrative shifts, but it should not be used as factual authority or
  representative polling without aggregation, sampling context, and safety
  screening.
- A web page needs stronger safety screening and corroboration.
- A preprint should expose review status rather than being treated as equivalent
  to a peer-reviewed article.

## Public-Opinion Evidence

CANON treats public social-media and forum text as a distinct evidence family:
`public_opinion`.

The current deterministic profile intentionally gives these records:

- low authority signal
- high timeliness need
- high source-intent risk
- high adversarial surface
- a required `aggregation_required` domain gate
- the recommended role `aggregate_public_opinion_and_screen_safety`

This means the records can be useful for investor, legal, policy, or cultural
analysis when the question is about public expression, themes, concerns,
support, opposition, trust, or controversy. They are not treated as reliable
claims about the world by default.

The current gate distinguishes channel-level risk from text-level safety. A
social source has a high adversarial surface, but that alone does not quarantine
every public-opinion record. Prompt-injection, exfiltration, role-confusion, and
tool-control patterns are handled by the safety gate before reranking and
generation.

Current limitations:

- The system does not yet model representativeness, bots, coordination, or
  platform sampling bias.
- Engagement counts are annotation-only signals and are not quality evidence.
- The deterministic profile cannot infer demographics, geography, or exposure
  without explicit metadata.
- Live scraping must respect platform terms, robots rules, consent, privacy,
  and legal requirements.
- Future versions should add stance/theme models, de-duplication, bot and
  coordination detection, temporal burst detection, and cross-platform
  corroboration.

`canon/eval/public_opinion.py` provides the deterministic baseline for that
future work. It reports lexical theme/stance probes, safety-allowed versus
all-public counts, duplicate text, author-hash concentration, temporal coverage,
and aggregation readiness. These are audit features, not validated opinion
measurements.

## Evidence Committee

`canon/eval/committee.py` adds a deterministic multi-judge audit over retrieval
candidates. It does not collapse importance into one scalar. Instead, it asks
separate judges to vote on:

- query relevance
- source trust
- domain fit
- safety
- corroboration
- public-opinion aggregation constraints

The committee emits:

- per-judge votes and reasons
- committee decisions such as `use_as_context`, `use_with_caution`,
  `aggregate_only`, `inspect_only`, and `reject`
- conflict tags such as `relevant_but_unsafe`,
  `relevant_but_low_trust`, `relevant_but_domain_review`, and
  `public_opinion_requires_aggregation`
- selected-context policy violations, where normal retrieval selected an item
  the committee does not consider generator-context eligible

Run it with:

```powershell
python -m canon.eval.committee --mode social_public_opinion_demo_corpus --policies balanced,rag --top-k 3
```

The report is written to `reports/evidence_committee_<mode>.json` and `.md`.
The committee report remains an audit layer, but context assembly now uses the
same policy logic as a hard generation gate. Items marked for public-opinion
aggregation, inspection, rejection, or quarantine can still be retained for
audit and aggregate analysis, but they are excluded from ordinary generator
context unless an explicit future exception grants a bounded use.

## Text-Level Signals

- section role
- claim density
- citation density
- methods/results/theory status
- recency relative to query need

## Chunking Signals

Chunking is treated as part of the evidence model, not just preprocessing.
CANON now uses deterministic document-type-aware boundary chunking:

- sentence boundaries are preserved when the text has usable sentence structure
- deterministic token-window chunking remains the fallback for overlong
  sentences, OCR fragments, tables, abstracts without punctuation, and other
  boundary-poor text
- overlap is preserved at sentence boundaries when possible
- chunking policy is selected from document-type metadata when available
- every chunk records `chunk_token_count`, `chunk_relative_start`,
  `chunk_relative_end`, `boundary_aligned`, `chunk_strategy`,
  `chunk_policy_document_type`, `chunk_policy_domain`,
  `chunk_policy_target_tokens`, and `chunk_policy_overlap_tokens`
- every chunk also records typed context decisions: `chunk_resolution`,
  `parent_context_mode`, `parent_context_recommended`,
  `generation_context_role`, `context_expansion_policy`,
  `evidence_containment_score`, `chunk_safety_contamination_risk`, and
  `chunk_quality_tier`
- chunks now carry `atomic_chunk_id` and deterministic `parent_context_id`
  metadata so atomic retrieval units can be linked back to section, clause,
  filing, OCR, speaker-turn, or thread/post context groups
- chunking policy canonicalizes raw source-form aliases through the
  document-type profiler, so labels such as `blog` and `court_opinion` route to
  web and legal policies instead of creating one-off chunking categories

Current deterministic policies:

- academic/preprint: section sentence boundaries
- policy/market reports: section sentence boundaries with moderate windows
- news/web pages: smaller paragraph/section-oriented windows because
  attribution and extraction risk matter
- legal authorities: clause-oriented boundaries with overlap to preserve
  exceptions and qualifiers
- regulatory filings: section-oriented boundaries with overlap for facts,
  risk disclosures, dates, and scoped assertions
- social/forum public opinion: post/thread boundaries with no overlap so
  prompt-injection or toxic text is not spread into neighboring chunks
- transcripts: turn-boundary candidate windows with overlap
- archival/OCR primary sources: conservative shorter windows with overlap
- unknown unstructured text: conservative boundary-aware windows until the
  document type is better known

Data cards summarize chunking behavior with:

- average/min/max chunk tokens
- sentence-boundary aligned share
- token-window fallback share
- chunk-strategy counts
- chunk-policy document-type counts
- chunk-resolution counts
- parent-context mode counts
- generation-context role counts
- parent-context recommended share
- parent-context count
- multi-chunk parent-context count
- parent-context expansion eligible share
- context-expansion policy counts
- average safety-contamination risk
- average evidence-containment score

This matters because bad chunking can split a claim from its qualifier, a method
from its result, a legal clause from its exception, or a social post from its
thread context. The current typed decisions make that risk inspectable: academic
and legal boundary chunks can be marked safe for parent-context expansion, while
web chunks require inspection and corroboration and public-opinion chunks remain
atomic/aggregate-only by default. Recent chunking work also argues that chunk quality must be
evaluated directly rather than assumed from chunk size alone: HOPE evaluates
passage properties and document coherence [1], late chunking keeps long-document
context in embeddings before pooling chunk representations [2], and hierarchical
chunking/auto-merge methods test whether multi-level document structure improves
retrieval [3]. Those ideas support CANON's direction: chunking should be a typed
decision layer with diagnostics, not a hidden preprocessing constant.

These chunking diagnostics are still deterministic. Future work should compare
sentence-aware chunking against semantic, discourse-aware, citation-aware,
legal-clause-aware, table-aware, thread-aware, contextual-retrieval, and
late-chunking variants. The comparison should be domain-sliced because the best
unit is unlikely to be universal: a legal exception, an ethnographic fieldnote,
a market-risk disclosure, and a social-media thread have different failure
modes.

`canon/evidence/chunk_context.py` adds the first multi-resolution context
expansion utility. It starts from retrieved atomic chunks, groups sibling chunks
by `parent_context_id`, reconstructs bounded parent text, and then runs the
safety scanner on the assembled parent context before exposing it to synthesis.
Expansion is blocked when the atomic chunk is unsafe, when the chunk policy is
not `safe_parent_expansion_allowed`, or when the expanded parent text itself
contains prompt-injection or quarantine-worthy material. This gives CANON a
deterministic baseline for parent-child retrieval without letting larger context
silently reintroduce unsafe text.

`canon/eval/chunking.py` is the first deterministic chunking benchmark. It
compares fixed token windows with boundary-aware chunking on academic, legal,
social/public-opinion, and OCR-like cases. The report measures:

- expected evidence-span containment
- boundary-aligned share
- chunk count
- unsafe prompt-injection span isolation
- document-type coverage across chunking probes

Run it with:

```powershell
python -m canon.eval.chunking --chunk-tokens 14 --overlap-tokens 0 --write-report
```

The output is written to `reports/chunking_strategy_eval_ct14_ov0.json` and
`.md`. This is a technical diagnostic. A strategy that preserves known spans in
the fixture is more plausible, but still needs downstream tests for retrieval,
faithfulness, citation support, and human usefulness.

Current deterministic benchmark result for `--chunk-tokens 14 --overlap-tokens
0`: boundary-aware chunking preserved all expected evidence spans in the fixture
set, while fixed token windows preserved two thirds. Boundary-aware chunking
also isolated the prompt-injection span in the social/public-opinion probe,
where fixed windows split it across context.

## Required Diagnostics

Every corpus report should include missingness and correlation checks where the
data exists. The system must not silently impute missing quality dimensions.

Author prominence is separated from source quality because it can amplify
existing prestige hierarchies and double-count citation advantage.

## References

[1] H. Bradland, M. Goodwin, P.-A. Andersen, A. S. Nossum, and A. Gupta, "A New
HOPE: Domain-agnostic Automatic Evaluation of Text Chunking," arXiv:2505.02171,
2025.

[2] M. Gunther, I. Mohr, D. J. Williams, B. Wang, and H. Xiao, "Late Chunking:
Contextual Chunk Embeddings Using Long-Context Embedding Models,"
arXiv:2409.04701, 2024.

[3] W. Lu, K. Chen, R. Qiao, and X. Sun, "HiChunk: Evaluating and Enhancing
Retrieval-Augmented Generation with Hierarchical Chunking," arXiv:2509.11552,
2025.

[4] C. Merola and J. Singh, "Reconstructing Context: Evaluating Advanced
Chunking Strategies for Retrieval-Augmented Generation," arXiv:2504.19754,
2025.
