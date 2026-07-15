# Research Skill Template

Use this template before running CANON, an LLM, or a search workflow on a broad
or controversial topic. The goal is to make the research frame explicit: topic,
subdomains, evidence standards, vocabulary, representation goals, and review
risks.

This is not a prompt for producing a final answer. It is a calibration template
for producing better questions, better retrieval, and a clearer human review
plan.

## 1. Topic

What is the broad topic?

Examples:

- nuclear energy
- vaccine confidence
- climate migration
- food insecurity
- microplastics
- urban policing

## 2. Subdomain Constellation

List multiple subdomains that may shape the evidence. A topic rarely belongs to
one field. Different disciplines ask different questions, use different terms,
and treat different source types as strong evidence.

### Social Research Example

Topic: climate migration

Possible connected subdomains:

- **Sociology:** inequality, stratification, institutions, social networks,
  community resilience, migration decision-making.
- **Anthropology:** lived experience, local knowledge, household adaptation,
  displacement narratives, kinship, place attachment.
- **History:** colonial land use, past displacement, environmental change over
  time, state formation, archival patterns.
- **Cultural studies:** media framing, identity, representation, discourse,
  memory, belonging.
- **Economics:** labor markets, remittances, household income, risk, insurance,
  cost-benefit models, development outcomes.
- **Political science:** state capacity, border policy, conflict, governance,
  international institutions.
- **Geography:** spatial exposure, vulnerability, place, mobility corridors,
  regional variation.

The researcher should decide which subdomains are central and which are
contextual. CANON should then retrieve across neighborhoods without pretending
that all subdomains use the same evidence standard.

### Natural Science Example

Topic: microplastics and health

Possible connected subdomains:

- **Biology:** organism exposure, cellular effects, inflammation, endocrine
  pathways, toxicology, bioaccumulation.
- **Chemistry:** polymer composition, additives, degradation, adsorption of
  pollutants, analytical detection methods.
- **Environmental science:** water systems, soil transport, marine exposure,
  ecological pathways.
- **Medicine/public health:** human exposure, epidemiology, clinical relevance,
  dose-response uncertainty.
- **Materials science:** plastic properties, particle size, manufacturing
  sources, degradation behavior.
- **Statistics/methods:** measurement error, confounding, sampling bias,
  comparability across assays.
- **Policy/regulation:** thresholds, precautionary principles, monitoring
  standards, waste policy.

For a science topic, CANON should separate mechanistic plausibility from direct
human-outcome evidence. Biology and chemistry may show a plausible pathway, but
that does not automatically prove population-level health effects.

## 3. Question Type

What kind of question is being asked?

- evidence map
- controversy map
- causal claim
- mechanism question
- intervention effectiveness
- risk assessment
- historical trend
- policy comparison
- terminology discovery
- source-quality audit
- overclaim check

## 4. Claim Being Tested

Optional, but useful when the user wants to test a strong statement.

Example:

> Climate migration is primarily caused by environmental change rather than
> political or economic conditions.

## 5. Evidence Standards By Subdomain

Different subdomains should have different standards.

Example for climate migration:

- Sociology: peer-reviewed empirical studies, survey data, interviews,
  demographic datasets.
- Anthropology: ethnography, fieldwork, local accounts, interpretive analysis.
- History: archives, historical records, longitudinal accounts.
- Cultural studies: discourse analysis, media studies, textual evidence.
- Economics: econometric studies, labor data, household surveys, model
  assumptions.
- Policy: official reports, legal documents, agency data, implementation
  records.

Example for microplastics:

- Biology: experimental studies, exposure models, toxicology assays.
- Chemistry: validated detection methods, polymer analysis, concentration
  measures.
- Public health: epidemiology, exposure assessment, dose-response evidence.
- Environmental science: sampling studies, pathway analysis, ecosystem data.
- Policy: regulatory thresholds, monitoring standards, risk assessments.

## 6. Source Inclusion

Which source types should be included?

- peer-reviewed articles
- books or book chapters
- government reports
- intergovernmental reports
- legal or regulatory documents
- datasets
- archival records
- interviews
- field notes
- news
- NGO or advocacy reports
- industry reports
- social media or public comments

## 7. Source Caution

Which sources should be flagged or downweighted?

- advocacy materials
- industry-funded claims
- anonymous commentary
- outdated sources
- non-peer-reviewed claims
- sources with unclear methods
- sources outside the geographic or historical scope
- sources that are useful for opinion representation but weak as evidence

## 8. Time Frame

What period matters?

- recent only
- post-2020
- post-2015
- historical period
- all available sources
- compare periods

## 9. Geographic Or Cultural Scope

What context matters?

- global
- United States
- EU
- Global South
- specific country
- city or region
- Indigenous/local communities
- diaspora communities
- cross-cultural comparison

## 10. Representation Goal

What kind of diversity should retrieval seek?

- disciplinary diversity
- viewpoint diversity
- methodological diversity
- geographic diversity
- historical-period diversity
- source-type diversity
- citation-cluster diversity
- stakeholder diversity
- language or terminology diversity

Representation does not mean false balance. A weak or fringe position may be
represented as a claim in the landscape while still being clearly marked as
weakly supported.

## 11. Known Vocabulary

What terms does the researcher already know?

Example:

- climate migration
- displacement
- adaptation

## 12. Vocabulary To Discover

What language should the system help uncover?

Example:

- trapped populations
- managed retreat
- immobility
- livelihood diversification
- environmental mobility
- adaptive capacity

## 13. Known Disagreements

What debates should the system look for?

Example:

- environmental drivers versus political-economic drivers
- migration as adaptation versus migration as failure
- voluntary mobility versus forced displacement
- macro-level modeling versus ethnographic accounts

## 14. Research Lens

Choose one or more retrieval lenses:

- **Broad discovery:** map neighborhoods and terminology.
- **Canonical depth:** prioritize central, high-quality sources.
- **Recent scan:** check whether newer sources change the trail.
- **Controversy map:** retrieve competing claims and disagreements.
- **Mechanism map:** connect pathways, mediators, and causal mechanisms.
- **Historical lens:** compare terminology and evidence across time.
- **Stakeholder lens:** separate expert, community, policy, industry, and public
  perspectives.
- **Overclaim check:** test whether a draft claim outruns the evidence.

## 15. Desired Output

What should the system produce?

- discovery map
- cited answer
- evidence table
- source shortlist
- controversy map
- query plan
- literature review outline
- claim check
- terminology guide
- human review queue

## Prompt Form

```text
You are helping me create a research query plan.

Use this research frame:

Topic:
Subdomain constellation:
Question type:
Claim being tested:
Evidence standards by subdomain:
Source inclusion:
Source caution:
Time frame:
Geographic or cultural scope:
Representation goal:
Known vocabulary:
Vocabulary to discover:
Known disagreements:
Research lens:
Desired output:

Return:
1. Better search queries
2. Field-specific terminology by subdomain
3. Subdomain-specific synonyms and related concepts
4. Sources to prioritize
5. Sources to treat cautiously
6. Expected disagreements
7. Evidence standards and false-balance risks
8. Retrieval risks
9. Questions I should ask next
```

## How To Use This With CANON

After completing the template, run a lens calibration report:

```powershell
python -m canon.eval.research_lens "YOUR QUESTION" --mode YOUR_CORPUS --top-k 10
```

Then inspect:

- whether the retrieved evidence spans the intended subdomains
- whether one discipline, source, or cluster dominates
- whether field phrases match the subdomains you care about
- whether the answer needs a broader lens, deeper lens, or narrower claim
- which terms should be accepted, edited, or rejected as calibration patterns

