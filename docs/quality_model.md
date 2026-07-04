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

## Text-Level Signals

- section role
- claim density
- citation density
- methods/results/theory status
- recency relative to query need

## Required Diagnostics

Every corpus report should include missingness and correlation checks where the
data exists. The system must not silently impute missing quality dimensions.

Author prominence is separated from source quality because it can amplify
existing prestige hierarchies and double-count citation advantage.
