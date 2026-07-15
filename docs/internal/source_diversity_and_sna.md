# Source Diversity and Social Network Analysis

CANON should treat repeated sources as a retrieval-quality risk. A cited answer
can still be narrow if most evidence comes from the same source, author group,
venue, graph cluster, or topical neighborhood.

The goal is not to force artificial variety. The goal is to show whether the
evidence base is broad enough for the user's question, and to warn when it is
not.

## Design Claim

Source diversity belongs inside the evidence engine. It should not be left to a
later report-writing layer.

The evidence engine should expose:

- distinct source count
- distinct work count
- dominant source share
- citation/community cluster count
- author or venue concentration when metadata exists
- repeated-source warnings
- useful breadth versus noisy breadth
- graph-neighborhood notes in evidence packets

This follows the broader retrieval and bibliometrics lesson that relevance is
not only lexical or semantic similarity. Citation links, co-citation,
bibliographic coupling, community structure, and diversification all provide
additional evidence about where a result sits in an information ecosystem
[1]-[8].

## Implemented First Pass

The first source-diversity evaluator is:

```powershell
python -m canon.eval.source_diversity --mode social_science_ir_v1_harvest10 --policy rag --top-k 10
```

It writes:

- `reports/source_diversity_<mode>_<policy>_k<k>.json`
- `reports/source_diversity_<mode>_<policy>_k<k>.md`

Per query, it reports:

- evidence count
- distinct sources
- distinct works
- distinct graph clusters
- dominant source
- dominant source share
- source counts
- graph cluster counts
- warnings
- compact top evidence

Warnings:

- `no_evidence`
- `low_source_diversity`
- `dominant_source_concentration`
- `low_graph_neighborhood_diversity`

## How SNA Fits

Social network analysis should be used as an evidence-discovery and
coverage-audit layer.

Recommended graph types:

- **Citation network:** source paper -> referenced paper.
- **Co-citation network:** papers are related when later works cite them
  together.
- **Bibliographic-coupling network:** papers are related when they cite the same
  sources.
- **Co-author network:** author collaboration and concentration.
- **Venue/source network:** recurring journals, platforms, agencies, or
  publishers.
- **Entity co-occurrence network:** countries, companies, policies, outcomes,
  methods.
- **Claim/theme network:** chunks grouped by semantically similar claims.

The retrieval layer can then use graph features to:

- find adjacent but relevant literature
- identify bridge sources between communities
- avoid overusing one source family
- surface contradictory evidence from another community
- warn when top evidence is narrow
- compare baseline retrieval with community-diversified retrieval

## Evidence Packet Impact

Evidence packets should include source-diversity metadata:

```json
{
  "source_diversity": {
    "distinct_sources": 5,
    "distinct_works": 8,
    "dominant_source": "Annual Review of Political Science",
    "dominant_source_share": 0.3,
    "graph_clusters": 3,
    "warnings": []
  },
  "graph_notes": [
    "Evidence spans three citation communities.",
    "One bridge source connects sanctions effectiveness and humanitarian-effects literature."
  ]
}
```

This improves integration with downstream systems. A report-writing or
multi-agent system can use the evidence packet without reading CANON's database
directly.

## Human Review Impact

The human reviewer should inspect source diversity when deciding:

- whether the answer is too narrow
- whether more evidence should be retrieved
- whether repeated sources weaken confidence
- whether graph-diverse additions are useful or noisy

This should become part of the release gate after the first implementation is
stable.

## Acceptance Question Update

The human-review acceptance set should not test only sanctions. It should cover
multiple topical and retrieval behaviors:

- democratic peace
- economic sanctions
- international institutions
- civil war
- human rights treaties
- climate-conflict relevance traps
- source-quality sensitivity
- recency sensitivity
- overclaim checks
- cross-topic vocabulary coaching

This makes the release gate more credible because it tests whether CANON can
retrieve across several evidence neighborhoods instead of repeatedly returning
the same source family.

## References

[1] P. Lewis *et al.*, "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks," 2020. [Online]. Available: https://arxiv.org/abs/2005.11401

[2] K. Guu, K. Lee, Z. Tung, P. Pasupat, and M.-W. Chang, "REALM: Retrieval-Augmented Language Model Pre-Training," 2020. [Online]. Available: https://arxiv.org/abs/2002.08909

[3] N. Thakur *et al.*, "BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models," 2021. [Online]. Available: https://arxiv.org/abs/2104.08663

[4] J. Carbonell and J. Goldstein, "The Use of MMR, Diversity-Based Reranking for Reordering Documents and Producing Summaries," 1998. [Online]. Available: https://doi.org/10.1145/290941.291025

[5] L. Page, S. Brin, R. Motwani, and T. Winograd, "The PageRank Citation Ranking: Bringing Order to the Web," 1999. [Online]. Available: http://ilpubs.stanford.edu:8090/422/1/1999-66.pdf

[6] J. M. Kleinberg, "Authoritative Sources in a Hyperlinked Environment," 1999. [Online]. Available: https://www.cs.cornell.edu/home/kleinber/auth.pdf

[7] M. M. Kessler, "Bibliographic Coupling Between Scientific Papers," 1963. [Online]. Available: https://doi.org/10.1002/asi.5090140103

[8] H. Small, "Co-citation in the Scientific Literature: A New Measure of the Relationship Between Two Documents," 1973. [Online]. Available: https://doi.org/10.1002/asi.4630240406

[9] H. D. White and B. C. Griffith, "Author Cocitation: A Literature Measure of Intellectual Structure," 1981. [Online]. Available: https://doi.org/10.1002/asi.4630320302

[10] M. E. J. Newman, "The Structure and Function of Complex Networks," 2003. [Online]. Available: https://doi.org/10.1137/S003614450342480

[11] V. D. Blondel, J.-L. Guillaume, R. Lambiotte, and E. Lefebvre, "Fast Unfolding of Communities in Large Networks," 2008. [Online]. Available: https://arxiv.org/abs/0803.0476

[12] S. Fortunato, "Community Detection in Graphs," 2010. [Online]. Available: https://doi.org/10.1016/j.physrep.2009.11.002

[13] O. Khattab and M. Zaharia, "ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT," 2020. [Online]. Available: https://arxiv.org/abs/2004.12832

[14] L. Gao *et al.*, "Precise Zero-Shot Dense Retrieval without Relevance Labels," 2022. [Online]. Available: https://arxiv.org/abs/2212.10496

[15] A. Asai *et al.*, "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection," 2023. [Online]. Available: https://arxiv.org/abs/2310.11511

[16] S. Yan *et al.*, "Corrective Retrieval Augmented Generation," 2024. [Online]. Available: https://arxiv.org/abs/2401.15884

[17] J. Saad-Falcon *et al.*, "ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation Systems," 2023. [Online]. Available: https://arxiv.org/abs/2311.09476

[18] D. Edge *et al.*, "From Local to Global: A Graph RAG Approach to Query-Focused Summarization," 2024. [Online]. Available: https://arxiv.org/abs/2404.16130

[19] F. Feng *et al.*, "Language-agnostic BERT Sentence Embedding," 2020. [Online]. Available: https://arxiv.org/abs/2007.01852

[20] J. Zhang *et al.*, "Mr. TyDi: A Multi-lingual Benchmark for Dense Retrieval," 2021. [Online]. Available: https://arxiv.org/abs/2108.08787

[21] A. Asai *et al.*, "XOR QA: Cross-lingual Open-Retrieval Question Answering," 2020. [Online]. Available: https://arxiv.org/abs/2010.11856

[22] N. Muennighoff *et al.*, "MTEB: Massive Text Embedding Benchmark," 2022. [Online]. Available: https://arxiv.org/abs/2210.07316
