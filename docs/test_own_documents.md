# Testing CANON on Your Own Documents

Use this workflow when you want to test CANON on private documents, a new
portfolio topic, or an industry-specific corpus.

The easiest path is the flexible ingest CLI. It profiles a file or folder,
infers the source shape, proposes a mapping, normalizes records, chunks text,
and writes a report before you build a corpus.

Supported first-pass inputs:

- JSONL
- JSON lists or objects with `records`, `documents`, `items`, or `data`
- CSV
- TXT
- Markdown
- folders containing a mix of those files

PDF and Word files still need to be converted to text, Markdown, CSV, JSON, or
JSONL before ingest.

## 1. Create a Test Corpus

Create a folder such as `data/my_docs/battery_storage/`.

For quick tests, you can drop `.txt` or `.md` files into the folder. For
structured records, use CSV, JSON, or JSONL.

Each line should be one JSON object. Minimal format:

```json
{"id":"doc-001","title":"Battery Storage Market Memo","year":2026,"source_name":"Internal research","document_type":"memo","provenance":"internal","domain":"energy_storage","text":"Battery storage projects face interconnection delays, financing constraints, and regional permitting variation."}
```

Sectioned format is better when the source has headings:

```json
{"id":"doc-002","title":"Grid Interconnection Report","year":2025,"source_name":"Public utility filing","document_type":"regulatory_filing","provenance":"official","domain":"energy_storage","url":"https://example.test/report","sections":[{"section":"queue delays","text":"The report describes interconnection queue backlogs for renewable and storage projects."},{"section":"cost exposure","text":"Developers report higher carrying costs when projects wait longer for approval."}]}
```

Recommended fields:

- `id`: stable unique id.
- `title`: document title.
- `year`: publication or document year.
- `source_name`: publisher, organization, author, or internal source.
- `document_type`: report, memo, paper, filing, transcript, article, policy,
  note, or another consistent label.
- `provenance`: official, peer_reviewed, internal, public_web, vendor,
  uncertain, or another consistent label.
- `domain`: your topic area.
- `url`: optional source link.
- `text` or `sections`: source content.

## 2. Ingest the Documents

Choose a short mode name for the raw document set. First inspect the source:

```powershell
python -m canon.ingest.flexible --input data/my_docs/battery_storage --mode battery_storage_v1 --profile-only
```

Then ingest it:

```powershell
python -m canon.ingest.flexible --input data/my_docs/battery_storage --mode battery_storage_v1
```

This writes:

- `data/raw/flexible_battery_storage_v1.json`
- `data/raw/unstructured_battery_storage_v1.json`
- `data/processed/works_battery_storage_v1.json`
- `data/processed/chunks_battery_storage_v1.json`
- `reports/flexible_ingest_battery_storage_v1.json`

Open the ingest report and check:

- detected source shape
- proposed field mapping
- skipped records and reasons
- inferred document types
- chunk count
- corpus limitations

If you already have clean JSONL in CANON's expected shape, the stricter importer
still works:

```powershell
python -m canon.ingest.unstructured --input data/my_docs/battery_storage.jsonl --mode battery_storage_v1
```

## 3. Build a Named Corpus

Build a corpus from that mode:

```powershell
python -m canon.corpus.build --corpus-id battery_storage_v1_corpus --from-modes battery_storage_v1 --corpus-only
```

Use the corpus id for retrieval, diagnostics, answering, and evaluation.

## 4. Ask Smoke-Test Questions

Start with real questions you would ask during research:

```powershell
python -m canon.synthesis.answer "What are the main risks for battery storage project delivery?" --mode battery_storage_v1_corpus --policy rag --top-k 5
```

Run a researcher-lens calibration before treating the answer as useful. This
compares the same query under broad discovery, canonical-depth, and recent-scan
lenses:

```powershell
python -m canon.eval.research_lens "What are the main risks for battery storage project delivery?" --mode battery_storage_v1_corpus --top-k 10
```

This writes:

- `reports/research_lens_<mode>_<query>.json`
- `reports/research_lens_<mode>_<query>.md`

Use this report to inspect:

- whether breadth and depth retrieve different evidence
- whether one source or graph cluster dominates
- which field phrases the corpus suggests
- which lens should drive human review for the question
- whether the answer should be treated as discovery, depth review, or a
  recency check

Run query diagnostics to see how wording changes retrieval:

```powershell
python -m canon.retrieval.query_diagnostics "battery storage project delays" --mode battery_storage_v1_corpus --top-k 5 --candidate-k 20 --freedom-level balanced --write-report
```

Use `--freedom-level strict` when you want conservative terminology suggestions.
Use `--freedom-level exploratory` when you want lower-probability adjacent
phrases that still need human inspection.

## 5. Build a Topic Test Set

Create 10 to 30 questions for each topic. Include:

- obvious expert wording
- beginner wording
- vague wording
- synonym-heavy wording
- queries with field-specific lingo
- questions where the right answer should be "not enough evidence"
- questions that could retrieve misleading but related evidence
- questions where source type matters

Example:

```text
1. What delays battery storage projects?
2. What does interconnection backlog mean for storage developers?
3. Are financing costs affecting project delivery?
4. Which documents support permitting as a delivery risk?
5. Is there enough evidence to compare utility-scale and residential storage?
```

## 6. Evaluate the Corpus

Run a data card:

```powershell
python -m canon.reports.data_card --mode battery_storage_v1_corpus
```

Run RAG evaluation across retrieval policies:

```powershell
python -m canon.eval.rag --mode battery_storage_v1_corpus --top-k 5 --policies lexical,balanced,semantic,rag
```

If you have relevance labels, compare semantic models on your own questions:

```powershell
python -m canon.eval.model_evaluation --mode battery_storage_v1_corpus --qrels gold/battery_storage_qrels.json --providers local,openai,cohere --k 10
```

Remote providers are marked `unavailable` when API keys are not configured. The
report still runs with the local model and records which providers were excluded
from the model leaderboard.

For each important question, save the answer output and query-diagnostics
report. Review them with `docs/human_review_rubric.md`.

Run source-diversity checks so the pilot does not overuse the same repeated
sources:

```powershell
python -m canon.eval.source_diversity --mode battery_storage_v1_corpus --policy rag --top-k 10
```

## 7. Pass Criteria for a Personal Corpus Pilot

Use these criteria before treating the result as production-quality for that
topic:

- At least 20 reviewed questions for the topic.
- At least 80% of questions retrieve three or more directly relevant top-10
  evidence items.
- At least 80% of cited answers are rated `minor_revision` or `usable`.
- Unsupported substantive claims stay below 5% of all reviewed claims.
- Every `needs_more_evidence` case is clearly identified instead of forced into
  a confident answer.
- At least 70% of inspected query-lingo suggestions are rated `useful` or
  `very_useful`.
- At least 70% of lens-calibration reports are rated by the reviewer as
  matching the intended research lens: broad discovery, canonical depth, or
  recent scan.
- Broad-discovery reports should usually add source or cluster breadth without
  burying the top evidence in off-topic matches.
- Drift risk is `acceptable` for strict and balanced query diagnostics.
- Exploratory query diagnostics may be broader, but any `needs_caution` or
  `drifted` cases are visible to the reviewer.
- Reviewers can inspect evidence, terminology suggestions, and limitations in
  under 15 minutes for at least 70% of questions.

## 8. Archive Or Encyclopedia Test

For an archive or encyclopedia, start with discovery questions rather than
final-answer questions. The system should help you map the collection before it
tries to synthesize conclusions.

Good first questions:

- What major topics does this archive contain about X?
- Which sources explain X directly, and which only mention it in passing?
- What terms does the archive use for X across decades or sections?
- Which entries are central, repeated, or cross-referenced?
- Where does the archive have thin or missing evidence?

Run the lens report for each question:

```powershell
python -m canon.eval.research_lens "What terms does this archive use for X?" --mode my_archive_v1_corpus --top-k 10
```

For encyclopedia-style corpora, pay special attention to:

- source concentration, because many entries may share the same publisher
- section titles and cross-reference terms
- historical wording that differs from modern terminology
- short entries that retrieve well lexically but lack depth
- broad survey entries that are useful for orientation but weak as evidence

The best result is not a perfect answer. The best result is a calibrated review
map: broad neighborhoods, central entries, recurring terminology, and a shortlist
of sources for human inspection.

## 9. What Good Looks Like

A strong personal-corpus test does not prove CANON is always right. It shows:

- the corpus can be ingested reproducibly
- retrieval works on the user's actual vocabulary
- terminology suggestions teach useful domain language
- researcher-calibrated lenses reveal breadth/depth tradeoffs
- cited answers stay grounded in retrieved evidence
- weak evidence is visible instead of hidden
- a human reviewer can accept, revise, reject, or request more evidence

That is the right finish line for CANON as a human-in-the-loop evidence briefing
tool.
