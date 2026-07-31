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
      font-size: 15px;
      line-height: 1.45;
    }
    button, input, select, textarea { font: inherit; }
    button {
      border: 1px solid transparent;
      background: var(--accent);
      color: #fff;
      min-height: 40px;
      padding: 0 14px;
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
    }
    textarea { min-height: 104px; resize: vertical; }
    label { display: grid; gap: 6px; font-weight: 700; color: var(--ink); }
    small { color: var(--muted); font-weight: 400; }
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
    .brand h1 { font-size: 18px; margin: 0; letter-spacing: 0; }
    .brand p { margin: 0; color: var(--muted); font-size: 13px; }
    .statusline { display: flex; gap: 8px; flex-wrap: wrap; justify-content: flex-end; }
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
    .row2 { display: grid; grid-template-columns: 1fr 110px; gap: 10px; align-items: end; }
    .row3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; align-items: end; }
    .toggles { display: grid; gap: 8px; }
    .checkrow { display: flex; gap: 8px; align-items: center; font-weight: 400; color: var(--ink); }
    .checkrow input { width: auto; }
    .buttonrow { display: flex; gap: 8px; flex-wrap: wrap; }
    .metrics {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
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
      .statusline { justify-content: flex-start; }
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
          <h1>CANON Evidence Discovery</h1>
          <p>Inspection-first research workbench</p>
        </div>
      </div>
      <div class="statusline" id="statusline">
        <span class="pill warn">Loading status</span>
      </div>
    </div>
  </header>

  <main>
    <aside>
      <section>
        <h2>Ask</h2>
        <form id="queryForm" class="formgrid">
          <label>
            Question
            <textarea id="query" required>Radioiodine treatment of non-toxic multinodular goitre reduces thyroid volume.</textarea>
          </label>
          <div class="row3">
            <label>
              Corpus
              <select id="mode"></select>
            </label>
            <label>
              Top K
              <input id="topK" type="number" min="1" max="50" value="12">
            </label>
            <label>
              Candidate K
              <input id="candidateK" type="number" min="1" max="100" value="12">
            </label>
          </div>
          <label>
            Query Mode
            <select id="freedomLevel">
              <option value="balanced">Balanced</option>
              <option value="strict">Strict</option>
              <option value="exploratory">Exploratory</option>
            </select>
          </label>
          <label>
            Retrieval Engine
            <select id="retrievalEngine">
              <option value="model_candidate_pool">Model candidate pool</option>
              <option value="canon_synthesis">CANON synthesis retriever</option>
            </select>
          </label>
          <div class="row2">
            <label>
              Candidate Scope
              <select id="candidateScope">
                <option value="lexical_window">Fast full-corpus scan</option>
                <option value="full_embedding_store">Full semantic index</option>
              </select>
            </label>
            <label>
              Lexical Window
              <input id="lexicalWindowK" type="number" min="12" max="500" value="96">
            </label>
          </div>
          <div class="row3">
            <label>
              Embeddings
              <select id="retrievalProvider">
                <option value="openrouter">OpenRouter</option>
                <option value="sentence-transformers">Local BGE</option>
                <option value="cohere">Cohere</option>
                <option value="local">Local hashed</option>
              </select>
            </label>
            <label>
              Embedding Model
              <select id="retrievalModel">
                <option value="qwen/qwen3-embedding-8b">Qwen3 embedding 8B</option>
                <option value="baai/bge-m3">BGE-M3</option>
                <option value="openai/text-embedding-3-small">OpenAI text-embedding-3-small</option>
                <option value="BAAI/bge-small-en-v1.5">BGE small local</option>
                <option value="embed-v4.0">Cohere embed v4</option>
                <option value="hashed-semantic-v1">Hashed lexical fallback</option>
              </select>
            </label>
            <label>
              Fusion
              <select id="fusion">
                <option value="weighted_bm25_dense">Weighted BM25 + dense</option>
                <option value="union">Union</option>
              </select>
            </label>
          </div>
          <div class="row2">
            <label>
              Reranker
              <select id="rerankerProvider">
                <option value="cohere">Cohere</option>
                <option value="openrouter">OpenRouter Cohere</option>
                <option value="heuristic">Heuristic</option>
              </select>
            </label>
            <label>
              Reranker Model
              <select id="rerankerModel">
                <option value="rerank-v4.0-pro">Cohere rerank v4 pro</option>
                <option value="cohere/rerank-v3.5">OpenRouter Cohere rerank v3.5</option>
                <option value="lexical-overlap-rerank-v1">Lexical overlap</option>
              </select>
            </label>
          </div>
          <div class="row2">
            <label>
              Generator
              <select id="generatorProvider">
                <option value="openrouter">OpenRouter</option>
                <option value="deterministic">Deterministic</option>
              </select>
            </label>
            <label>
              Model
              <select id="generatorModel">
                <option value="openai/gpt-4.1-mini">GPT-4.1 mini</option>
                <option value="openai/gpt-4o-mini">GPT-4o mini</option>
                <option value="google/gemini-2.5-flash">Gemini 2.5 Flash</option>
                <option value="moonshotai/kimi-k3">Kimi K3</option>
                <option value="moonshotai/kimi-k2.7-code">Kimi K2.7 Code</option>
                <option value="moonshotai/kimi-k2.6">Kimi K2.6</option>
                <option value="moonshotai/kimi-k2-thinking">Kimi K2 Thinking</option>
              </select>
            </label>
          </div>
          <div class="toggles">
            <label class="checkrow"><input id="suggestExternal" type="checkbox"> External-search suggestions</label>
            <label class="checkrow"><input id="executeExternalSearch" type="checkbox"> OpenAlex online results</label>
            <label class="checkrow"><input id="writeTelemetry" type="checkbox" checked> Local telemetry</label>
          </div>
          <label>
            Online Results
            <input id="maxExternalResults" type="number" min="1" max="10" value="5">
          </label>
          <div class="buttonrow">
            <button id="runButton" type="submit">Run Evidence Search</button>
            <button class="secondary" id="sampleButton" type="button">Use Sample</button>
          </div>
        </form>
      </section>

      <section>
        <h2>Corpus Setup</h2>
        <form id="corpusForm" class="formgrid">
          <label>
            Local Path
            <input id="inputPath" type="text" placeholder="C:\\path\\to\\docs or data/my_docs">
          </label>
          <div class="row2">
            <label>
              Mode ID
              <input id="corpusMode" type="text" placeholder="my_topic_v1">
            </label>
            <label>
              Corpus ID
              <input id="corpusId" type="text" placeholder="my_topic_v1_corpus">
            </label>
          </div>
          <div class="row2">
            <label>
              Domain
              <input id="corpusDomain" type="text" placeholder="optional">
            </label>
            <label>
              Source Name
              <input id="sourceName" type="text" placeholder="optional">
            </label>
          </div>
          <label class="checkrow"><input id="profileOnly" type="checkbox"> Profile only</label>
          <button class="secondary" id="corpusButton" type="submit">Set Up Corpus</button>
        </form>
        <pre id="corpusResult" class="corpus-result">{}</pre>
      </section>

      <section>
        <h2>Boundary</h2>
        <ul class="list" id="boundaryList"></ul>
      </section>

      <section>
        <h2>Feedback</h2>
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
          </label>
          <label>
            Note
            <textarea id="feedbackComment" placeholder="What worked or failed?"></textarea>
          </label>
          <button class="secondary" id="feedbackButton" type="submit" disabled>Save Feedback</button>
        </form>
      </section>
    </aside>

    <div class="workspace">
      <section>
        <div class="metrics">
          <div class="metric"><span>Status</span><strong id="metricStatus">Idle</strong></div>
          <div class="metric"><span>Evidence</span><strong id="metricEvidence">0</strong></div>
          <div class="metric"><span>Support</span><strong id="metricSupport">-</strong></div>
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
      window.setTimeout(() => $("toast").classList.add("hidden"), 3200);
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
      $("runButton").textContent = isBusy ? "Running..." : "Run Evidence Search";
    }

    function setCorpusBusy(isBusy) {
      $("corpusButton").disabled = isBusy;
      $("corpusButton").textContent = isBusy ? "Working..." : "Set Up Corpus";
    }

    function renderStatus(status) {
      state.status = status;
      const quality = status.quality_state || {};
      const blocked = quality.blocked_checks || [];
      $("statusline").innerHTML = [
        `<span class="pill">${esc(status.status)}</span>`,
        `<span class="pill warn">Gate: ${esc(quality.stage1_gate_status || "unknown")}</span>`,
        `<span class="pill ${blocked.length ? "danger" : ""}">${blocked.length} blocked checks</span>`
      ].join("");
      const select = $("mode");
      select.innerHTML = (status.available_modes || []).map((mode) =>
        `<option value="${esc(mode)}" ${mode === status.mode ? "selected" : ""}>${esc(mode)}</option>`
      ).join("");
      $("topK").value = status.recommended_defaults?.top_k || 12;
      $("candidateK").value = status.recommended_defaults?.candidate_k || 12;
      $("retrievalEngine").value = status.recommended_defaults?.retrieval_engine || "model_candidate_pool";
      $("candidateScope").value = status.recommended_defaults?.candidate_scope || "lexical_window";
      $("lexicalWindowK").value = status.recommended_defaults?.lexical_window_k || 96;
      $("retrievalProvider").value = status.recommended_defaults?.retrieval_provider || "openrouter";
      $("retrievalModel").value = status.recommended_defaults?.retrieval_model || "qwen/qwen3-embedding-8b";
      $("rerankerProvider").value = status.recommended_defaults?.reranker_provider || "cohere";
      $("rerankerModel").value = status.recommended_defaults?.reranker_model || "rerank-v4.0-pro";
      $("generatorProvider").value = status.recommended_defaults?.generator_provider || "openrouter";
      $("generatorModel").value = status.recommended_defaults?.generator_model || "openai/gpt-4.1-mini";
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
      $("metricGaps").textContent = (session.coverage_gaps || []).length;
      $("feedbackButton").disabled = !session.session_id;
      renderRelevanceBanner(gate);

      const draft = session.draft_brief || {};
      $("draft").className = draft.text ? "draft" : "draft empty";
      $("draft").textContent = draft.text || draft.abstention || "No evidence note was produced for this query.";
      $("actions").innerHTML = (session.recommended_actions || []).map((item) => `<li>${esc(item)}</li>`).join("");

      const cards = session.evidence_cards || [];
      $("evidence").innerHTML = cards.length ? cards.map(renderEvidenceCard).join("") : `<div class="empty">No evidence was retrieved.</div>`;
      $("runDiagnosis").innerHTML = renderRunDiagnosis(session.run_diagnosis || {});
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
          <button class="ghost" type="button" data-evidence-id="${esc(item.evidence_id || "")}" data-feedback-type="irrelevant">Irrelevant</button>
          <button class="ghost" type="button" data-evidence-id="${esc(item.evidence_id || "")}" data-feedback-type="citation_issue">Citation Issue</button>
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
          lexical_window_k: Number($("lexicalWindowK").value || 96),
          retrieval_provider: $("retrievalProvider").value,
          retrieval_model: $("retrievalModel").value,
          reranker_provider: $("rerankerProvider").value,
          reranker_model: $("rerankerModel").value,
          fusion: $("fusion").value,
          generator_provider: $("generatorProvider").value,
          generator_model: $("generatorProvider").value === "openrouter" ? $("generatorModel").value : "",
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

    async function setupCorpus(event) {
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
        };
        const result = await api("/v1/production/corpus-setup", {
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
