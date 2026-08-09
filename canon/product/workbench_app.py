from __future__ import annotations


def render_app() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CANON Evidence Discovery</title>
  <style>
    :root {
      --bg: #f4f6f2;
      --surface: #ffffff;
      --ink: #18201d;
      --muted: #59645f;
      --line: #d7ded8;
      --accent: #19675a;
      --accent-soft: #dcefeb;
      --amber: #8a5a10;
      --amber-soft: #fff3d8;
      --red: #a33b2f;
      --red-soft: #fde9e4;
      --focus: #2d6cdf;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      color: var(--ink);
      background: var(--bg);
      font-family: Arial, Helvetica, sans-serif;
      font-size: 16px;
      line-height: 1.55;
    }
    button, input, select, textarea { font: inherit; }
    button {
      border: 1px solid transparent;
      background: var(--accent);
      color: #fff;
      min-height: 44px;
      padding: 0 16px;
      border-radius: 6px;
      cursor: pointer;
    }
    button.secondary { background: var(--surface); color: var(--accent); border-color: var(--accent); }
    button.ghost { background: transparent; color: var(--ink); border-color: var(--line); }
    button:disabled { opacity: 0.55; cursor: wait; }
    input, select, textarea {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      color: var(--ink);
      padding: 10px 11px;
      min-height: 44px;
    }
    textarea { min-height: 104px; resize: vertical; }
    label { display: grid; gap: 6px; font-weight: 700; color: var(--ink); }
    small { color: var(--muted); font-weight: 400; }
    .help {
      color: var(--muted);
      font-size: 13px;
      font-weight: 400;
      line-height: 1.35;
    }
    .recommendation {
      color: #10483f;
      font-size: 13px;
      font-weight: 700;
    }
    .guide-list {
      display: grid;
      gap: 8px;
      margin: 0;
      padding-left: 20px;
    }
    .guide-list li { padding-left: 2px; }
    .mini-note {
      color: var(--muted);
      font-size: 13px;
      margin: -4px 0 2px;
    }
    button:focus-visible,
    input:focus-visible,
    select:focus-visible,
    textarea:focus-visible,
    summary:focus-visible {
      outline: 3px solid var(--focus);
      outline-offset: 2px;
    }
    header {
      border-bottom: 1px solid var(--line);
      background: var(--surface);
    }
    .topbar {
      max-width: 1280px;
      margin: 0 auto;
      padding: 14px 20px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 16px;
    }
    .brand { display: flex; gap: 12px; align-items: center; min-width: 240px; }
    .mark {
      width: 34px;
      height: 34px;
      border-radius: 7px;
      background: linear-gradient(135deg, #19675a, #b78b21);
      color: #fff;
      display: grid;
      place-items: center;
      font-weight: 800;
    }
    .brand h1 { font-size: 21px; margin: 0; letter-spacing: 0; }
    .brand p {
      margin: 1px 0 0;
      color: var(--muted);
      font-size: 14px;
      max-width: 760px;
    }
    .pill {
      display: inline-flex;
      align-items: center;
      min-height: 28px;
      padding: 3px 9px;
      border-radius: 999px;
      background: var(--accent-soft);
      color: #10483f;
      border: 1px solid #bdddd6;
      font-size: 13px;
      white-space: nowrap;
    }
    .pill.warn { background: var(--amber-soft); color: var(--amber); border-color: #f0d79d; }
    .pill.danger { background: var(--red-soft); color: var(--red); border-color: #f3beb5; }
    main {
      max-width: 1280px;
      margin: 0 auto;
      padding: 18px 20px 28px;
      display: grid;
      grid-template-columns: minmax(300px, 380px) minmax(0, 1fr);
      gap: 18px;
      align-items: start;
    }
    aside, .workspace {
      display: grid;
      gap: 14px;
    }
    section, .panel {
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
    }
    h2 { margin: 0 0 10px; font-size: 17px; letter-spacing: 0; }
    h3 { margin: 0 0 8px; font-size: 15px; letter-spacing: 0; }
    .formgrid { display: grid; gap: 12px; }
    .advanced {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fbfcfa;
      padding: 0;
    }
    .advanced summary {
      min-height: 46px;
      padding: 10px 12px;
      cursor: pointer;
      font-weight: 800;
      color: var(--ink);
    }
    .advanced-body {
      display: grid;
      gap: 12px;
      padding: 0 12px 12px;
    }
    .advanced > .formgrid,
    .advanced > .list,
    .advanced > .corpus-result {
      margin: 0 12px 12px;
    }
    .row2 { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; align-items: start; }
    .row3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; align-items: end; }
    .toggles { display: grid; gap: 8px; }
    .checkrow { display: flex; gap: 8px; align-items: center; font-weight: 400; color: var(--ink); }
    .checkrow input { width: auto; }
    .buttonrow { display: flex; gap: 8px; flex-wrap: wrap; }
    .metrics {
      display: grid;
      grid-template-columns: repeat(5, minmax(0, 1fr));
      gap: 10px;
    }
    .metric {
      min-height: 76px;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      background: #fbfcfa;
    }
    .metric span { display: block; color: var(--muted); font-size: 12px; }
    .metric strong { display: block; margin-top: 4px; font-size: 20px; }
    .split {
      display: grid;
      grid-template-columns: minmax(0, 1.05fr) minmax(280px, 0.95fr);
      gap: 14px;
      align-items: start;
    }
    .draft {
      min-height: 180px;
      white-space: pre-wrap;
    }
    .plain-summary {
      margin-top: 12px;
      color: var(--muted);
    }
    .evidence-list { display: grid; gap: 10px; }
    .evidence {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      background: #fff;
    }
    .evidence-head {
      display: flex;
      gap: 8px;
      justify-content: space-between;
      align-items: start;
      margin-bottom: 8px;
    }
    .evidence-title { font-weight: 800; }
    .evidence-meta { color: var(--muted); font-size: 13px; margin-top: 2px; }
    .evidence-text { color: #28312d; margin: 8px 0 0; }
    .list { margin: 0; padding-left: 18px; }
    .list li { margin: 4px 0; }
    .empty {
      color: var(--muted);
      border: 1px dashed var(--line);
      border-radius: 8px;
      padding: 18px;
      background: #fbfcfa;
    }
    .banner {
      border: 1px solid var(--line);
      border-left: 4px solid var(--accent);
      border-radius: 8px;
      padding: 12px 14px;
      margin-bottom: 14px;
      background: #fbfcfa;
      color: var(--ink);
    }
    .banner.warn { border-left-color: var(--amber); background: var(--amber-soft); }
    .banner.danger { border-left-color: var(--red); background: var(--red-soft); }
    .diagnosis {
      display: grid;
      gap: 10px;
      margin-bottom: 14px;
    }
    .diagnosis-stage {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      background: #fbfcfa;
    }
    .stage-head {
      display: flex;
      justify-content: space-between;
      gap: 10px;
      align-items: center;
      margin-bottom: 5px;
    }
    .stage-name { font-weight: 800; }
    .stage-status {
      display: inline-flex;
      align-items: center;
      min-height: 24px;
      padding: 2px 8px;
      border-radius: 999px;
      border: 1px solid #bdddd6;
      color: #10483f;
      background: var(--accent-soft);
      font-size: 12px;
      white-space: nowrap;
    }
    .stage-status.warn,
    .stage-status.review_required { color: var(--amber); background: var(--amber-soft); border-color: #f0d79d; }
    .stage-status.fail { color: var(--red); background: var(--red-soft); border-color: #f3beb5; }
    .stage-status.danger { color: var(--red); background: var(--red-soft); border-color: #f3beb5; }
    .signals {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 4px 12px;
      margin-top: 8px;
      color: var(--muted);
      font-size: 13px;
    }
    .signals span { min-width: 0; overflow-wrap: anywhere; }
    .corpus-result {
      max-height: 190px;
      font-size: 12px;
      white-space: pre-wrap;
    }
    .tabs { display: flex; gap: 6px; border-bottom: 1px solid var(--line); margin: -4px -4px 12px; padding: 0 4px; }
    .tab {
      color: var(--muted);
      background: transparent;
      border: 0;
      border-radius: 0;
      min-height: 38px;
      padding: 0 10px;
    }
    .tab.active {
      color: var(--accent);
      border-bottom: 3px solid var(--accent);
    }
    .hidden { display: none; }
    .toast {
      position: fixed;
      right: 18px;
      bottom: 18px;
      max-width: 420px;
      background: #17201d;
      color: #fff;
      padding: 12px 14px;
      border-radius: 8px;
      box-shadow: 0 10px 30px rgba(0, 0, 0, 0.18);
    }
    pre {
      overflow: auto;
      max-height: 360px;
      background: #111816;
      color: #e8f2ef;
      padding: 12px;
      border-radius: 8px;
      font-size: 12px;
    }
    @media (max-width: 980px) {
      main { grid-template-columns: 1fr; }
      .split { grid-template-columns: 1fr; }
      .metrics { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .topbar { align-items: flex-start; flex-direction: column; }
    }
    @media (max-width: 560px) {
      .row2, .row3, .metrics { grid-template-columns: 1fr; }
      main { padding: 12px; }
      .topbar { padding: 12px; }
    }
  </style>
</head>
<body>
  <header>
    <div class="topbar">
      <div class="brand">
        <div class="mark">C</div>
        <div>
          <h1>CANON</h1>
          <p>Search your documents, compare AI drafts, and inspect the evidence before trusting an answer.</p>
        </div>
      </div>
    </div>
  </header>

  <main>
    <aside>
      <details class="advanced">
        <summary>Quick Guide</summary>
        <ol class="guide-list">
          <li>Choose the corpus you want to search.</li>
          <li>Ask one focused question in ordinary language.</li>
          <li>Start with the recommended settings unless the evidence looks thin.</li>
          <li>Read the evidence cards before trusting the draft.</li>
          <li>Use Diagnostics when the note says evidence is weak, missing, or mixed.</li>
        </ol>
      </details>

      <section>
        <h2>Ask</h2>
        <form id="queryForm" class="formgrid">
          <label>
            Question
            <textarea id="query" required>Radioiodine treatment of non-toxic multinodular goitre reduces thyroid volume.</textarea>
            <small class="help">Ask one specific question or claim. Avoid very broad prompts until you know the corpus covers the topic.</small>
            <small class="recommendation">Recommended: one sentence, with the key people, place, method, or outcome named.</small>
          </label>
          <div class="row2">
            <label>
              Corpus
              <select id="mode"></select>
              <small class="help">The document collection CANON will search.</small>
              <small class="recommendation">Recommended: use your own indexed corpus; use beir_scifact_mini only for quick testing.</small>
            </label>
            <label>
              Reading Depth
              <input id="topK" type="number" min="1" max="50" value="12">
              <small class="help">How many evidence items are shown and considered for the note.</small>
              <small class="recommendation">Recommended: 12 for normal work; 5 for quick checks; 20+ for difficult reviews.</small>
            </label>
          </div>
          <div class="buttonrow">
            <button id="runButton" type="submit">Run Search</button>
            <button class="secondary" id="sampleButton" type="button">Use Sample</button>
          </div>
          <details class="advanced">
            <summary>Advanced settings</summary>
            <div class="advanced-body">
              <p class="mini-note">These defaults are tuned for hosted Qdrant retrieval plus two OpenRouter generation calls. Change them only when you are testing a failure mode or comparing models.</p>
              <label>
                Query Mode
                <select id="freedomLevel">
                  <option value="balanced">Balanced - recommended</option>
                  <option value="strict">Strict</option>
                  <option value="exploratory">Exploratory</option>
                </select>
                <small class="help">Controls how tightly CANON interprets the wording of your question.</small>
                <small class="recommendation">Recommended: Balanced. Use Strict for exact claim checking; Exploratory when you are still learning the vocabulary.</small>
              </label>
              <label>
                Retrieval Engine
                <select id="retrievalEngine">
                  <option value="model_candidate_pool">Model candidate pool - recommended</option>
                  <option value="canon_synthesis">CANON synthesis retriever</option>
                </select>
                <small class="help">Chooses the retrieval path before drafting.</small>
                <small class="recommendation">Recommended: Model candidate pool for production-like search. CANON synthesis is useful for simpler local comparisons.</small>
              </label>
              <div class="row2">
                <label>
                  Candidate Scope
                  <select id="candidateScope">
                    <option value="vector_store">Qdrant/ANN index - recommended</option>
                    <option value="lexical_window">Fast full-corpus scan</option>
                    <option value="full_embedding_store">Full semantic index</option>
                  </select>
                  <small class="help">Where candidates come from before reranking.</small>
                  <small class="recommendation">Recommended: Qdrant/ANN index after corpus setup. Use lexical scan when Qdrant is unavailable.</small>
                </label>
                <label>
                  Candidate K
                  <input id="candidateK" type="number" min="1" max="100" value="12">
                  <small class="help">How many candidate chunks are gathered before final evidence selection.</small>
                  <small class="recommendation">Recommended: 12 for interactive use; raise it when key evidence is missing.</small>
                </label>
              </div>
              <label>
                Lexical Window
                <input id="lexicalWindowK" type="number" min="12" max="500" value="96">
                <small class="help">How many lexical matches to inspect when the fallback scan is used.</small>
                <small class="recommendation">Recommended: 96. Increase only for broad corpora or hard-to-match wording.</small>
              </label>
              <div class="row3">
                <label>
                  Embeddings
                  <select id="retrievalProvider">
                    <option value="openrouter">OpenRouter - recommended</option>
                    <option value="sentence-transformers">Local BGE</option>
                    <option value="cohere">Cohere</option>
                    <option value="local">Local hashed</option>
                  </select>
                  <small class="help">The service used to turn text into semantic vectors.</small>
                  <small class="recommendation">Recommended: OpenRouter for hosted alpha. Local hashed is only a fallback/control.</small>
                </label>
                <label>
                  Embedding Model
                  <select id="retrievalModel">
                    <option value="qwen/qwen3-embedding-8b">Qwen3 embedding 8B - recommended</option>
                    <option value="baai/bge-m3">BGE-M3</option>
                    <option value="openai/text-embedding-3-small">text-embedding-3-small via OpenRouter</option>
                    <option value="BAAI/bge-small-en-v1.5">BGE small local</option>
                    <option value="embed-v4.0">Cohere embed v4</option>
                    <option value="hashed-semantic-v1">Hashed lexical fallback</option>
                  </select>
                  <small class="help">The embedding model used for semantic search and Qdrant indexing compatibility.</small>
                  <small class="recommendation">Recommended: Qwen3 embedding 8B. Rebuild the vector index if you change this.</small>
                </label>
                <label>
                  Fusion
                  <select id="fusion">
                    <option value="weighted_bm25_dense">Weighted BM25 + dense - recommended</option>
                    <option value="union">Union</option>
                  </select>
                  <small class="help">Combines keyword and semantic retrieval before reranking.</small>
                  <small class="recommendation">Recommended: Weighted BM25 + dense for mixed technical and plain-language queries.</small>
                </label>
              </div>
              <div class="row2">
                <label>
                  Reranker
                  <select id="rerankerProvider">
                    <option value="cohere">Cohere - recommended</option>
                    <option value="openrouter">OpenRouter Cohere</option>
                    <option value="heuristic">Heuristic</option>
                  </select>
                  <small class="help">Reorders candidate evidence after initial retrieval.</small>
                  <small class="recommendation">Recommended: Cohere for quality. Heuristic is a no-cost fallback, not the best result path.</small>
                </label>
                <label>
                  Reranker Model
                  <select id="rerankerModel">
                    <option value="rerank-v4.0-pro">Cohere rerank v4 pro - recommended</option>
                    <option value="cohere/rerank-v3.5">OpenRouter Cohere rerank v3.5</option>
                    <option value="lexical-overlap-rerank-v1">Lexical overlap</option>
                  </select>
                  <small class="help">The model used for reranking retrieved snippets.</small>
                  <small class="recommendation">Recommended: rerank-v4.0-pro when Cohere is configured.</small>
                </label>
              </div>
              <div class="row2">
                <label>
                  Generator
                  <select id="generatorProvider">
                    <option value="openrouter">OpenRouter - recommended</option>
                    <option value="deterministic">Deterministic</option>
                  </select>
                  <small class="help">The service used to write the draft note from retrieved evidence.</small>
                  <small class="recommendation">Recommended: OpenRouter. Deterministic mode is only for debugging without model calls.</small>
                </label>
                <label>
                  Model
                  <select id="generatorModel">
                    <option value="openai/gpt-4.1-mini">GPT-4.1 mini via OpenRouter - recommended</option>
                    <option value="openai/gpt-4o-mini">GPT-4o mini via OpenRouter</option>
                    <option value="moonshotai/kimi-k3">Kimi K3</option>
                    <option value="moonshotai/kimi-k2.7-code">Kimi K2.7 Code</option>
                    <option value="moonshotai/kimi-k2.6">Kimi K2.6</option>
                    <option value="moonshotai/kimi-k2-thinking">Kimi K2 Thinking</option>
                  </select>
                  <small class="help">The primary model selected for the visible draft when it succeeds.</small>
                  <small class="recommendation">Recommended: GPT-4.1 mini for quick, concise first drafts.</small>
                </label>
              </div>
              <div class="row2">
                <label>
                  Comparison Model
                  <select id="comparisonGeneratorModel">
                    <option value="moonshotai/kimi-k3">Kimi K3 - recommended</option>
                    <option value="openai/gpt-4.1-mini">GPT-4.1 mini via OpenRouter</option>
                    <option value="openai/gpt-4o-mini">GPT-4o mini via OpenRouter</option>
                    <option value="moonshotai/kimi-k2.7-code">Kimi K2.7 Code</option>
                    <option value="moonshotai/kimi-k2.6">Kimi K2.6</option>
                    <option value="moonshotai/kimi-k2-thinking">Kimi K2 Thinking</option>
                  </select>
                  <small class="help">A second model runs on the same evidence so you can compare phrasing, abstention, and latency.</small>
                  <small class="recommendation">Recommended: Kimi K3 as an independent comparison model.</small>
                </label>
                <label>
                  <span class="checkrow"><input id="modelPairEnabled" type="checkbox" checked> Compare two generation models</span>
                  <small class="help">Runs both the primary and comparison model and stores both outputs for analysis.</small>
                  <small class="recommendation">Recommended: on. Turn off only when reducing cost or latency.</small>
                </label>
              </div>
              <div class="toggles">
                <label>
                  <span class="checkrow"><input id="suggestExternal" type="checkbox"> External-search suggestions</span>
                  <small class="help">Suggests online searches when the corpus looks incomplete.</small>
                  <small class="recommendation">Recommended: off for first pass; on when coverage gaps are expected.</small>
                </label>
                <label>
                  <span class="checkrow"><input id="executeExternalSearch" type="checkbox"> OpenAlex online results</span>
                  <small class="help">Adds clearly marked online results alongside corpus evidence.</small>
                  <small class="recommendation">Recommended: off when validating your own corpus; on for discovery work.</small>
                </label>
                <label>
                  <span class="checkrow"><input id="runModelReview" type="checkbox"> Model stance review</span>
                  <small class="help">Asks a model to classify stance and extraction issues over the retrieved evidence.</small>
                  <small class="recommendation">Recommended: off for quick searches; on before sharing or acting on a note.</small>
                </label>
                <label>
                  <span class="checkrow"><input id="writeTelemetry" type="checkbox" checked> Local telemetry</span>
                  <small class="help">Stores local run records, model outputs, timing, and failure types for later analysis.</small>
                  <small class="recommendation">Recommended: on. Turn off only for throwaway experiments.</small>
                </label>
              </div>
              <div class="row2">
                <label>
                  Review Model
                  <select id="modelReviewModel">
                    <option value="openai/gpt-4.1-mini">GPT-4.1 mini via OpenRouter - recommended</option>
                    <option value="openai/gpt-4o-mini">GPT-4o mini via OpenRouter</option>
                    <option value="moonshotai/kimi-k3">Kimi K3</option>
                  </select>
                  <small class="help">The model used only when Model stance review is enabled.</small>
                  <small class="recommendation">Recommended: GPT-4.1 mini for fast review notes.</small>
                </label>
                <label>
                  Vector Backend
                  <select id="vectorBackend">
                    <option value="qdrant">Qdrant - recommended</option>
                  </select>
                  <small class="help">The hosted vector database used for indexed retrieval.</small>
                  <small class="recommendation">Recommended: Qdrant for hosted alpha; the processed corpus remains the source of truth.</small>
                </label>
              </div>
              <label>
                Online Results
                <input id="maxExternalResults" type="number" min="1" max="10" value="5">
                <small class="help">Maximum number of online items to add when OpenAlex online results are enabled.</small>
                <small class="recommendation">Recommended: 5 so online results help without crowding corpus evidence.</small>
              </label>
            </div>
          </details>
        </form>
      </section>

      <details class="advanced">
        <summary>Corpus setup</summary>
        <form id="corpusForm" class="formgrid">
          <label>
            Local Path
            <input id="inputPath" type="text" placeholder="C:\\path\\to\\docs or data/my_docs">
            <small class="help">Folder or file path containing the documents you want to research.</small>
            <small class="recommendation">Recommended: one project folder per research topic.</small>
          </label>
          <div class="row2">
            <label>
              Mode ID
              <input id="corpusMode" type="text" placeholder="my_topic_v1">
              <small class="help">Short name used to run searches against this corpus.</small>
              <small class="recommendation">Recommended: lowercase words with underscores, such as grid_risk_v1.</small>
            </label>
            <label>
              Corpus ID
              <input id="corpusId" type="text" placeholder="my_topic_v1_corpus">
              <small class="help">Internal storage name for the processed document collection.</small>
              <small class="recommendation">Recommended: use the mode ID plus _corpus.</small>
            </label>
          </div>
          <div class="row2">
            <label>
              Domain
              <input id="corpusDomain" type="text" placeholder="optional">
              <small class="help">Optional topic label attached to processed sources.</small>
              <small class="recommendation">Recommended: fill this when the project has a clear domain.</small>
            </label>
            <label>
              Source Name
              <input id="sourceName" type="text" placeholder="optional">
              <small class="help">Optional source label used when documents come from one collection or client folder.</small>
              <small class="recommendation">Recommended: leave blank for mixed folders.</small>
            </label>
          </div>
          <label>
            <span class="checkrow"><input id="profileOnly" type="checkbox"> Profile only</span>
            <small class="help">Checks what CANON can ingest without building or indexing the corpus.</small>
            <small class="recommendation">Recommended: off when you are ready to search; on for a dry run.</small>
          </label>
          <div class="toggles">
            <label>
              <span class="checkrow"><input id="indexVectorStore" type="checkbox" checked> Refresh vector index</span>
              <small class="help">Uploads new/changed chunks to Qdrant after processing documents.</small>
              <small class="recommendation">Recommended: on for normal use.</small>
            </label>
            <label>
              <span class="checkrow"><input id="deleteStaleVectors" type="checkbox" checked> Remove stale vectors</span>
              <small class="help">Removes index entries for files that were deleted or changed.</small>
              <small class="recommendation">Recommended: on to keep Qdrant aligned with your folder.</small>
            </label>
            <label>
              <span class="checkrow"><input id="forceRefresh" type="checkbox"> Force refresh</span>
              <small class="help">Rebuilds even when file hashes look unchanged.</small>
              <small class="recommendation">Recommended: off; turn on after changing embedding model or chunking settings.</small>
            </label>
          </div>
          <div class="row2">
            <label>
              Index Backend
              <select id="indexVectorBackend">
                <option value="qdrant">Qdrant - recommended</option>
              </select>
              <small class="help">Where semantic vectors are stored for search.</small>
              <small class="recommendation">Recommended: Qdrant for hosted alpha.</small>
            </label>
            <label>
              Index Model
              <select id="indexEmbeddingModel">
                <option value="qwen/qwen3-embedding-8b">Qwen3 embedding 8B - recommended</option>
                <option value="baai/bge-m3">BGE-M3</option>
                <option value="openai/text-embedding-3-small">text-embedding-3-small via OpenRouter</option>
              </select>
              <small class="help">Embedding model used when refreshing the vector index.</small>
              <small class="recommendation">Recommended: match the search embedding model, usually Qwen3 embedding 8B.</small>
            </label>
          </div>
          <div class="buttonrow">
            <button class="secondary" id="corpusButton" type="submit">Set Up Corpus</button>
            <button class="ghost" id="refreshCorpusButton" type="button">Refresh Changed Files</button>
          </div>
        </form>
        <pre id="corpusResult" class="corpus-result">{}</pre>
      </details>

      <details class="advanced">
        <summary>Boundary</summary>
        <p class="mini-note">Use this to remind yourself what the system can safely do without human review.</p>
        <ul class="list" id="boundaryList"></ul>
      </details>

      <details class="advanced">
        <summary>Feedback</summary>
        <form id="feedbackForm" class="formgrid">
          <label>
            Rating
            <select id="rating">
              <option value="">No rating</option>
              <option value="5">5 - useful</option>
              <option value="4">4</option>
              <option value="3">3</option>
              <option value="2">2</option>
              <option value="1">1 - not useful</option>
            </select>
            <small class="help">Your usefulness score for the run.</small>
            <small class="recommendation">Recommended: rate after checking the evidence cards.</small>
          </label>
          <label>
            Type
            <select id="feedbackType">
              <option value="useful">Useful</option>
              <option value="irrelevant">Irrelevant</option>
              <option value="missing_source">Missing source</option>
              <option value="citation_issue">Citation issue</option>
              <option value="query_needs_refinement">Query needs refinement</option>
            </select>
            <small class="help">The main category of feedback for this run.</small>
            <small class="recommendation">Recommended: choose the most concrete failure type you noticed.</small>
          </label>
          <label>
            Note
            <textarea id="feedbackComment" placeholder="What worked or failed?"></textarea>
            <small class="help">Short free-text note for later analysis.</small>
            <small class="recommendation">Recommended: mention missing sources, wrong evidence, or useful phrasing.</small>
          </label>
          <button class="secondary" id="feedbackButton" type="submit" disabled>Save Feedback</button>
        </form>
      </details>
    </aside>

    <div class="workspace">
      <section>
        <div class="metrics">
          <div class="metric"><span>Status</span><strong id="metricStatus">Idle</strong></div>
          <div class="metric"><span>Evidence</span><strong id="metricEvidence">0</strong></div>
          <div class="metric"><span>Support</span><strong id="metricSupport">-</strong></div>
          <div class="metric"><span>AI Models</span><strong id="metricModels">-</strong></div>
          <div class="metric"><span>Gaps</span><strong id="metricGaps">0</strong></div>
        </div>
      </section>

      <section>
        <div class="tabs">
          <button class="tab active" data-tab="answer" type="button">Draft</button>
          <button class="tab" data-tab="evidence" type="button">Evidence</button>
          <button class="tab" data-tab="diagnostics" type="button">Diagnostics</button>
          <button class="tab" data-tab="raw" type="button">JSON</button>
        </div>
        <div id="tab-answer" class="tabpane">
          <div id="relevanceBanner" class="banner hidden"></div>
          <div class="split">
            <div>
              <h2>Evidence Note</h2>
              <div id="draft" class="draft empty">Run a search to prepare an evidence-derived note.</div>
              <div id="modelPlainSummary" class="plain-summary empty">AI model status will appear here.</div>
            </div>
            <div>
              <h2>Next Actions</h2>
              <ul class="list" id="actions"></ul>
            </div>
          </div>
        </div>
        <div id="tab-evidence" class="tabpane hidden">
          <h2>Evidence</h2>
          <div id="evidence" class="evidence-list">
            <div class="empty">Evidence will appear here.</div>
          </div>
        </div>
        <div id="tab-diagnostics" class="tabpane hidden">
          <h2>Why This Run</h2>
          <div id="runDiagnosis" class="diagnosis">
            <div class="empty">No run diagnosis yet.</div>
          </div>
          <h2>Model Comparison</h2>
          <div id="modelComparison" class="diagnosis">
            <div class="empty">No model comparison yet.</div>
          </div>
          <h2>Model Review</h2>
          <div id="modelReview" class="diagnosis">
            <div class="empty">No model review yet.</div>
          </div>
          <div class="split">
            <div>
              <h2>Query Lingo</h2>
              <div id="queryLingo" class="empty">No diagnostics yet.</div>
            </div>
            <div>
              <h2>Coverage Gaps</h2>
              <ul class="list" id="gaps"></ul>
            </div>
          </div>
        </div>
        <div id="tab-raw" class="tabpane hidden">
          <h2>Raw Session JSON</h2>
          <pre id="rawJson">{}</pre>
        </div>
      </section>
    </div>
  </main>

  <div id="toast" class="toast hidden"></div>

  <script>
    const state = { status: null, session: null };
    const $ = (id) => document.getElementById(id);

    function esc(value) {
      return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;");
    }

    function showToast(message) {
      $("toast").textContent = message;
      $("toast").classList.remove("hidden");
      window.setTimeout(() => $("toast").classList.add("hidden"), 4200);
    }

    async function api(path, options = {}) {
      const response = await fetch(path, {
        headers: { "Content-Type": "application/json" },
        ...options,
      });
      const text = await response.text();
      let payload = {};
      try { payload = text ? JSON.parse(text) : {}; } catch (error) { payload = { error: text }; }
      if (!response.ok) throw new Error(payload.error || response.statusText);
      return payload;
    }

    function setBusy(isBusy) {
      $("runButton").disabled = isBusy;
      $("runButton").setAttribute("aria-busy", isBusy ? "true" : "false");
      $("runButton").textContent = isBusy ? "Searching..." : "Run Search";
    }

    function setCorpusBusy(isBusy) {
      $("corpusButton").disabled = isBusy;
      $("refreshCorpusButton").disabled = isBusy;
      $("corpusButton").textContent = isBusy ? "Working..." : "Set Up Corpus";
      $("refreshCorpusButton").textContent = isBusy ? "Working..." : "Refresh Changed Files";
    }

    function renderStatus(status) {
      state.status = status;
      const select = $("mode");
      select.innerHTML = (status.available_modes || []).map((mode) =>
        `<option value="${esc(mode)}" ${mode === status.mode ? "selected" : ""}>${esc(mode)}</option>`
      ).join("");
      $("topK").value = status.recommended_defaults?.top_k || 12;
      $("candidateK").value = status.recommended_defaults?.candidate_k || 12;
      $("retrievalEngine").value = status.recommended_defaults?.retrieval_engine || "model_candidate_pool";
      $("candidateScope").value = status.recommended_defaults?.candidate_scope || "vector_store";
      $("vectorBackend").value = status.recommended_defaults?.vector_backend || "qdrant";
      $("lexicalWindowK").value = status.recommended_defaults?.lexical_window_k || 96;
      $("retrievalProvider").value = status.recommended_defaults?.retrieval_provider || "openrouter";
      $("retrievalModel").value = status.recommended_defaults?.retrieval_model || "qwen/qwen3-embedding-8b";
      $("rerankerProvider").value = status.recommended_defaults?.reranker_provider || "cohere";
      $("rerankerModel").value = status.recommended_defaults?.reranker_model || "rerank-v4.0-pro";
      $("generatorProvider").value = status.recommended_defaults?.generator_provider || "openrouter";
      $("generatorModel").value = status.recommended_defaults?.generator_model || "openai/gpt-4.1-mini";
      $("comparisonGeneratorModel").value = status.recommended_defaults?.comparison_generator_model || "moonshotai/kimi-k3";
      $("modelPairEnabled").checked = status.recommended_defaults?.model_pair_enabled !== false;
      $("modelReviewModel").value = status.recommended_defaults?.model_review_model || "openai/gpt-4.1-mini";
      $("boundaryList").innerHTML = (status.shippable_without_human_review || []).map((item) =>
        `<li>${esc(item)}</li>`
      ).join("");
    }

    function renderSession(session) {
      state.session = session;
      const packet = session.evidence_packet || {};
      const gate = session.relevance_gate || {};
      $("metricStatus").textContent = session.status || "-";
      $("metricEvidence").textContent = packet.usable_evidence_count ?? packet.evidence_count ?? 0;
      $("metricSupport").textContent = gate.status || packet.support_level || "-";
      $("metricModels").textContent = modelMetricText(session.model_comparison || {});
      $("metricGaps").textContent = (session.coverage_gaps || []).length;
      $("feedbackButton").disabled = !session.session_id;
      renderRelevanceBanner(gate);

      const draft = session.draft_brief || {};
      $("draft").className = draft.text ? "draft" : "draft empty";
      $("draft").textContent = draft.text || draft.abstention || "No evidence note was produced for this query.";
      $("modelPlainSummary").className = "plain-summary";
      $("modelPlainSummary").innerHTML = renderModelPlainSummary(session.model_comparison || {});
      $("actions").innerHTML = (session.recommended_actions || []).map((item) => `<li>${esc(item)}</li>`).join("");

      const cards = session.evidence_cards || [];
      $("evidence").innerHTML = cards.length ? cards.map(renderEvidenceCard).join("") : `<div class="empty">No evidence was retrieved.</div>`;
      $("runDiagnosis").innerHTML = renderRunDiagnosis(session.run_diagnosis || {});
      $("modelComparison").innerHTML = renderModelComparison(session.model_comparison || {});
      $("modelReview").innerHTML = renderModelReview(session.model_review || {});
      $("queryLingo").innerHTML = renderQueryLingo(session.query_lingo || {});
      $("gaps").innerHTML = (session.coverage_gaps || []).map((gap) => `<li>${esc(gap.gap || gap)}</li>`).join("");
      $("rawJson").textContent = JSON.stringify(session, null, 2);
    }

    function renderRelevanceBanner(gate) {
      const banner = $("relevanceBanner");
      if (!gate.status) {
        banner.className = "banner hidden";
        banner.textContent = "";
        return;
      }
      const tone = gate.severity === "high" ? "danger" : gate.severity === "medium" ? "warn" : "";
      banner.className = `banner ${tone}`.trim();
      const weak = (gate.weak_terms || []).length ? ` Missing: ${(gate.weak_terms || []).join(", ")}.` : "";
      banner.textContent = `${gate.status}: ${gate.message || ""}${weak}`;
    }

    function renderEvidenceCard(item) {
      const relevance = item.relevance || {};
      const relevanceLabel = relevance.relevance_label || "review";
      const isExternal = item.evidence_scope === "external_source";
      const marker = isExternal ? `<span class="pill warn">ONLINE</span>` : `<span class="pill">CORPUS</span>`;
      return `<article class="evidence">
        <div class="evidence-head">
          <div>
            <div class="evidence-title">${esc(item.evidence_id || "Evidence")} ${marker} ${esc(item.title || "")}</div>
            <div class="evidence-meta">${esc(item.source_name || "Unknown source")} ${item.published_at ? " | " + esc(item.published_at) : ""}</div>
          </div>
          <span class="pill ${relevanceLabel === "off_query" ? "danger" : ""}">${esc(relevanceLabel)} | Rank ${esc(item.rank || "")}</span>
        </div>
        <div class="evidence-meta">${esc(item.evidence_scope || "")} ${item.source_type ? " | " + esc(item.source_type) : ""} ${relevance.usage_status ? " | " + esc(relevance.usage_status) : ""}</div>
        <p class="evidence-text">${esc(item.text || "")}</p>
        <div class="buttonrow">
          <button class="ghost" type="button" data-evidence-id="${esc(item.evidence_id || "")}" data-feedback-type="useful">Useful</button>
          <button class="ghost" type="button" data-evidence-id="${esc(item.evidence_id || "")}" data-feedback-type="irrelevant">Not Relevant</button>
          <button class="ghost" type="button" data-evidence-id="${esc(item.evidence_id || "")}" data-feedback-type="citation_issue">Citation Problem</button>
        </div>
      </article>`;
    }

    function renderQueryLingo(lingo) {
      const rows = [
        ["Matched terms", lingo.matched_terms || []],
        ["Weak terms", lingo.weak_terms || []],
        ["Field phrases", lingo.field_phrases || []],
        ["Top similarity", [lingo.semantic_similarity_summary?.top_max ?? "-"]],
        ["Drift risk", [lingo.drift_risk || "unknown"]],
      ];
      return rows.map(([label, values]) =>
        `<p><strong>${esc(label)}:</strong> ${esc((values || []).join(", ") || "-")}</p>`
      ).join("") + `<h3>Variants</h3><ul class="list">${(lingo.query_variants || []).slice(0, 5).map((row) =>
        `<li>${esc(row.query || row)}</li>`
      ).join("")}</ul>`;
    }

    function renderRunDiagnosis(diagnosis) {
      if (!diagnosis.report_id) return `<div class="empty">No run diagnosis yet.</div>`;
      const tone = diagnosis.overall_status === "failed_no_grounded_answer"
        ? "danger"
        : diagnosis.overall_status === "ready_for_user_inspection"
          ? ""
          : "warn";
      const metrics = diagnosis.metrics || {};
      const issues = (diagnosis.issue_categories || []).map((issue) =>
        `<span class="pill ${issue === "human_review_required" ? "warn" : ""}">${esc(issue)}</span>`
      ).join("");
      const stages = (diagnosis.stages || []).map((stage) => {
        const signals = Object.entries(stage.signals || {}).slice(0, 6).map(([key, value]) =>
          `<span><strong>${esc(key)}:</strong> ${esc(renderSignalValue(value))}</span>`
        ).join("");
        return `<div class="diagnosis-stage">
          <div class="stage-head">
            <span class="stage-name">${esc(stage.stage || "")}</span>
            <span class="stage-status ${esc(stage.status || "")}">${esc(stage.status || "")}</span>
          </div>
          <div>${esc(stage.message || "")}</div>
          ${signals ? `<div class="signals">${signals}</div>` : ""}
        </div>`;
      }).join("");
      return `<div class="banner ${tone}">
        <strong>${esc(diagnosis.overall_status || "")}</strong>: ${esc(diagnosis.summary || "")}
      </div>
      <div class="signals">
        <span><strong>failure_class:</strong> ${esc(diagnosis.failure_class || "-")}</span>
        <span><strong>usable_evidence:</strong> ${esc(metrics.usable_evidence_count ?? "-")}</span>
        <span><strong>term_coverage:</strong> ${esc(metrics.query_term_coverage ?? "-")}</span>
        <span><strong>coverage_gaps:</strong> ${esc(metrics.coverage_gap_count ?? 0)}</span>
      </div>
      <div class="buttonrow">${issues}</div>
      <div>${stages}</div>`;
    }

    function renderModelReview(review) {
      if (!review.status || review.status === "not_requested") {
        return `<div class="empty">${esc(review.boundary || "No model review requested.")}</div>`;
      }
      if (review.status === "model_review_failed") {
        return `<div class="banner danger"><strong>${esc(review.status)}</strong>: ${esc(review.error || "")}</div>`;
      }
      const diagnosis = review.disagreement_diagnosis || {};
      const stanceRows = (review.stance_assessments || []).map((row) =>
        `<div class="diagnosis-stage">
          <div class="stage-head">
            <span class="stage-name">${esc(row.evidence_id || "")} ${esc(row.stance || "")}</span>
            <span class="stage-status">${esc(row.confidence ?? "")}</span>
          </div>
          <div>${esc(row.claim || "")}</div>
          <div class="evidence-meta">${esc(row.excerpt || "")}</div>
        </div>`
      ).join("");
      const dimensions = (review.extracted_dimensions || []).slice(0, 8).map((row) =>
        `<span><strong>${esc(row.dimension || "")}:</strong> ${esc(row.value || "")}</span>`
      ).join("");
      return `<div class="banner warn">
        <strong>${esc(review.status)}</strong>: ${esc(diagnosis.axis || "review")} ${diagnosis.summary ? "- " + esc(diagnosis.summary) : ""}
      </div>
      <div class="signals">
        <span><strong>provider:</strong> ${esc(review.provider || "")}</span>
        <span><strong>model:</strong> ${esc(review.model || "")}</span>
        <span><strong>human_review:</strong> ${esc(review.human_review_required ? "required" : "not flagged")}</span>
        ${dimensions}
      </div>
      <div>${stanceRows || `<div class="empty">No usable stance rows returned.</div>`}</div>`;
    }

    function renderModelComparison(comparison) {
      if (!comparison.status || comparison.status === "disabled") {
        return `<div class="empty">${esc(comparison.boundary || "Model comparison disabled.")}</div>`;
      }
      const runs = (comparison.runs || []).map((run) => {
        const generator = run.generator || {};
        const selected = generator.model === comparison.selected_model ? `<span class="pill">selected</span>` : "";
        const tone = String(run.status || "").includes("failed") || String(run.status || "").includes("empty") ? "danger" : "";
        return `<div class="diagnosis-stage">
          <div class="stage-head">
            <span class="stage-name">${esc(run.model_role || "")} ${esc(generator.model || "")} ${selected}</span>
            <span class="stage-status ${tone}">${esc(run.status || "")}</span>
          </div>
          <div class="signals">
            <span><strong>latency_ms:</strong> ${esc(run.elapsed_ms ?? "-")}</span>
            <span><strong>input_tokens:</strong> ${esc(run.input_tokens ?? "-")}</span>
            <span><strong>output_tokens:</strong> ${esc(run.output_tokens ?? "-")}</span>
            <span><strong>citations:</strong> ${esc(run.citation_count ?? 0)}</span>
          </div>
          <div class="evidence-meta">${esc(run.error || run.boundary || "")}</div>
        </div>`;
      }).join("");
      return `<div class="banner warn">
        <strong>${esc(comparison.status)}</strong>: ${esc(comparison.success_count ?? 0)} / ${esc(comparison.call_count ?? 0)} model calls succeeded.
      </div>
      <div class="signals">
        <span><strong>primary:</strong> ${esc(comparison.primary_model || "")}</span>
        <span><strong>comparison:</strong> ${esc(comparison.comparison_model || "")}</span>
        <span><strong>selected:</strong> ${esc(comparison.selected_model || "")}</span>
        <span><strong>stored:</strong> ${esc(comparison.stored_at || "")}</span>
      </div>
      <div>${runs || `<div class="empty">No model runs were recorded.</div>`}</div>`;
    }

    function modelMetricText(comparison) {
      if (!comparison.status || comparison.status === "disabled") return "-";
      return `${comparison.success_count ?? 0}/${comparison.call_count ?? 0}`;
    }

    function renderModelPlainSummary(comparison) {
      if (!comparison.status || comparison.status === "disabled") {
        return esc(comparison.boundary || "No AI model comparison was run.");
      }
      const selected = comparison.selected_model ? ` Selected: ${comparison.selected_model}.` : "";
      const failures = (comparison.summary?.failure_types || []).filter(Boolean);
      const failureText = failures.length ? ` Issue: ${failures.join(", ")}.` : "";
      if (comparison.status === "model_pair_complete") {
        return `<strong>AI models:</strong> both answered.${esc(selected)}`;
      }
      if (comparison.status === "model_pair_partial") {
        return `<strong>AI models:</strong> one answered and one failed.${esc(selected + failureText)}`;
      }
      return `<strong>AI models:</strong> no model answer was returned.${esc(failureText || " Check Diagnostics for details.")}`;
    }

    function renderSignalValue(value) {
      if (Array.isArray(value)) return value.join(", ") || "-";
      if (value && typeof value === "object") return JSON.stringify(value);
      return value ?? "-";
    }

    async function runSearch(event) {
      event.preventDefault();
      setBusy(true);
      try {
        const payload = {
          query: $("query").value,
          mode: $("mode").value,
          top_k: Number($("topK").value || 12),
          candidate_k: Number($("candidateK").value || $("topK").value || 12),
          freedom_level: $("freedomLevel").value,
          retrieval_engine: $("retrievalEngine").value,
          candidate_scope: $("candidateScope").value,
          vector_backend: $("vectorBackend").value,
          lexical_window_k: Number($("lexicalWindowK").value || 96),
          retrieval_provider: $("retrievalProvider").value,
          retrieval_model: $("retrievalModel").value,
          reranker_provider: $("rerankerProvider").value,
          reranker_model: $("rerankerModel").value,
          fusion: $("fusion").value,
          generator_provider: $("generatorProvider").value,
          generator_model: $("generatorProvider").value === "openrouter" ? $("generatorModel").value : "",
          comparison_generator_model: $("generatorProvider").value === "openrouter" ? $("comparisonGeneratorModel").value : "",
          model_pair_enabled: $("modelPairEnabled").checked,
          run_model_review: $("runModelReview").checked,
          model_review_provider: "openrouter",
          model_review_model: $("modelReviewModel").value,
          allow_external_model_review: $("runModelReview").checked,
          suggest_external_expansion: $("suggestExternal").checked,
          execute_external_search: $("executeExternalSearch").checked,
          external_search_provider: "openalex",
          max_external_results: Number($("maxExternalResults").value || 5),
          write_telemetry: $("writeTelemetry").checked,
        };
        const session = await api("/v1/production/evidence-workbench", {
          method: "POST",
          body: JSON.stringify(payload),
        });
        renderSession(session);
        showToast("Evidence search complete.");
      } catch (error) {
        showToast(error.message);
      } finally {
        setBusy(false);
      }
    }

    async function submitFeedback(event) {
      event.preventDefault();
      if (!state.session) return;
      try {
        const response = await api("/v1/production/feedback", {
          method: "POST",
          body: JSON.stringify({
            session_id: state.session.session_id,
            query: state.session.query,
            rating: $("rating").value,
            feedback_type: $("feedbackType").value,
            comment: $("feedbackComment").value,
          }),
        });
        $("feedbackComment").value = "";
        showToast(response.status);
      } catch (error) {
        showToast(error.message);
      }
    }

    async function setupCorpus(event, endpoint = "/v1/production/corpus-setup") {
      event.preventDefault();
      setCorpusBusy(true);
      try {
        const payload = {
          input_path: $("inputPath").value,
          mode: $("corpusMode").value,
          corpus_id: $("corpusId").value,
          domain: $("corpusDomain").value,
          source_name: $("sourceName").value,
          profile_only: $("profileOnly").checked,
          build_corpus: !$("profileOnly").checked,
          index_vector_store: $("indexVectorStore").checked && !$("profileOnly").checked,
          vector_backend: $("indexVectorBackend").value,
          index_embedding_provider: "openrouter",
          index_embedding_model: $("indexEmbeddingModel").value,
          delete_stale_vectors: $("deleteStaleVectors").checked,
          force: $("forceRefresh").checked,
        };
        const result = await api(endpoint, {
          method: "POST",
          body: JSON.stringify(payload),
        });
        $("corpusResult").textContent = JSON.stringify(result, null, 2);
        const selectedMode = result.recommended_mode || result.mode || payload.mode;
        const status = await api(`/v1/production/status?mode=${encodeURIComponent(selectedMode)}`);
        renderStatus(status);
        if (selectedMode) $("mode").value = selectedMode;
        showToast(result.status);
      } catch (error) {
        showToast(error.message);
      } finally {
        setCorpusBusy(false);
      }
    }

    async function markEvidence(evidenceId, feedbackType) {
      if (!state.session) return;
      try {
        await api("/v1/production/feedback", {
          method: "POST",
          body: JSON.stringify({
            session_id: state.session.session_id,
            query: state.session.query,
            evidence_id: evidenceId,
            feedback_type: feedbackType,
          }),
        });
        showToast("Evidence feedback saved.");
      } catch (error) {
        showToast(error.message);
      }
    }

    function useSample() {
      const samples = state.status?.try_queries || [];
      if (samples.length) $("query").value = samples[Math.floor(Math.random() * samples.length)];
    }

    function setupTabs() {
      document.querySelectorAll(".tab").forEach((button) => {
        button.addEventListener("click", () => {
          document.querySelectorAll(".tab").forEach((item) => item.classList.remove("active"));
          document.querySelectorAll(".tabpane").forEach((item) => item.classList.add("hidden"));
          button.classList.add("active");
          $(`tab-${button.dataset.tab}`).classList.remove("hidden");
        });
      });
    }

    function setupEvidenceFeedback() {
      $("evidence").addEventListener("click", (event) => {
        const target = event.target;
        const button = target instanceof Element ? target.closest("button[data-feedback-type]") : null;
        if (!button) return;
        markEvidence(button.dataset.evidenceId || "", button.dataset.feedbackType || "general");
      });
    }

    async function boot() {
      setupTabs();
      setupEvidenceFeedback();
      $("queryForm").addEventListener("submit", runSearch);
      $("corpusForm").addEventListener("submit", setupCorpus);
      $("refreshCorpusButton").addEventListener("click", (event) => setupCorpus(event, "/v1/production/corpus-refresh"));
      $("feedbackForm").addEventListener("submit", submitFeedback);
      $("sampleButton").addEventListener("click", useSample);
      try {
        renderStatus(await api("/v1/production/status"));
      } catch (error) {
        showToast(error.message);
      }
    }

    boot();
  </script>
</body>
</html>
"""
