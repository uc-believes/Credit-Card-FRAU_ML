/**
 * Fraud Shield — Professional Fraud Analytics Dashboard
 * JavaScript Application Logic & Interactive State
 */

let appPresets = {};
let currentModalTxId = null;
let charts = {};

function escapeHtml(str) {
  if (str === null || str === undefined) return "";
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

document.addEventListener("DOMContentLoaded", () => {
  initApp();
});

function initApp() {
  setupNavigation();
  populateAdditionalInputs();
  loadPresets();
  loadOverviewData();
  loadQueueData();
  loadPerformanceData();
  loadExplainabilityData();
  loadMonitoringData();
  loadAdaptiveData();

  // Periodic telemetry refresh every 20 seconds
  setInterval(() => {
    loadMonitoringData(true);
    loadOverviewData(true);
  }, 20000);
}

// ---------------------------------------------------------------------------
// Navigation & Hash Routing
// ---------------------------------------------------------------------------
function setupNavigation() {
  const navItems = document.querySelectorAll(".nav-item");
  navItems.forEach((item) => {
    item.addEventListener("click", () => {
      const tabId = item.getAttribute("data-tab");
      switchTab(tabId);
    });
  });

  // Handle URL hash on initial load
  const hash = window.location.hash.replace("#", "");
  if (hash && document.getElementById(`tab-${hash}`)) {
    switchTab(hash);
  } else {
    switchTab("overview");
  }

  window.addEventListener("hashchange", () => {
    const newHash = window.location.hash.replace("#", "");
    if (newHash && document.getElementById(`tab-${newHash}`)) {
      switchTab(newHash, false);
    }
  });
}

function switchTab(tabId, updateHash = true) {
  // Update nav links
  document.querySelectorAll(".nav-item").forEach((item) => {
    item.classList.toggle("active", item.getAttribute("data-tab") === tabId);
  });

  // Update tab panels
  document.querySelectorAll(".tab-pane").forEach((pane) => {
    pane.classList.toggle("active", pane.id === `tab-${tabId}`);
  });

  // Update title
  const titles = {
    overview: "Executive Overview",
    predict: "Predict Transaction & Risk Assessment",
    queue: "Investigation Queue",
    performance: "Model Performance & Benchmarks",
    explainability: "Explainability & Feature Attribution (SHAP)",
    monitoring: "Model Monitoring & Drift Telemetry",
    adaptive: "Adaptive Intelligence & What-If Simulation",
  };
  const titleEl = document.getElementById("active-tab-title");
  if (titleEl) titleEl.textContent = titles[tabId] || "Dashboard";

  if (tabId === "adaptive") {
    loadAdaptiveData();
  }

  if (updateHash) {
    window.location.hash = `#${tabId}`;
  }
}

// ---------------------------------------------------------------------------
// Inputs Population
// ---------------------------------------------------------------------------
function populateAdditionalInputs() {
  const container = document.getElementById("additional-pca-inputs");
  if (!container) return;

  const highlighted = [14, 12, 10, 4, 11, 17];
  let html = "";
  for (let i = 1; i <= 28; i++) {
    if (!highlighted.includes(i)) {
      html += `
        <div class="form-group">
          <label class="form-label" for="input-v${i}">V${i}</label>
          <input type="number" step="0.001" class="form-control" id="input-v${i}" name="V${i}" value="0.0" />
        </div>
      `;
    }
  }
  container.innerHTML = html;
}

// ---------------------------------------------------------------------------
// Presets & Form Handling
// ---------------------------------------------------------------------------
async function loadPresets() {
  try {
    const res = await fetch("/api/sample-transactions");
    const json = await res.json();
    if (json.status === "success") {
      appPresets = json.presets;
    }
  } catch (e) {
    console.warn("Could not load presets:", e);
  }
}

function loadPreset(key) {
  if (!appPresets[key]) return;
  const p = appPresets[key];
  const data = p.data;

  // Set Amount & Time
  const amtEl = document.getElementById("input-amount");
  const timeEl = document.getElementById("input-time");
  if (amtEl) amtEl.value = data.Amount ?? 50.0;
  if (timeEl) timeEl.value = data.Time ?? 100.0;

  // Set V1..V28
  for (let i = 1; i <= 28; i++) {
    const el = document.getElementById(`input-v${i}`);
    if (el) {
      el.value = data[`V${i}`] !== undefined ? data[`V${i}`] : 0.0;
    }
  }

  // Trigger prediction automatically
  const fakeEvent = { preventDefault: () => {} };
  handlePredictionSubmit(fakeEvent);
}

function resetForm() {
  document.getElementById("prediction-form").reset();
  document.getElementById("prediction-result-content").style.display = "none";
  document.getElementById("prediction-result-placeholder").style.display = "block";
  document.getElementById("res-decision-badge").textContent = "PENDING EVALUATION";
  document.getElementById("res-decision-badge").className = "badge badge-low";
}

async function handlePredictionSubmit(e) {
  e.preventDefault();
  const btn = document.getElementById("btn-run-prediction");
  btn.textContent = "Analyzing Risk Signal...";
  btn.disabled = true;

  // Collect form inputs
  const formData = new FormData(document.getElementById("prediction-form"));
  const payload = {};
  formData.forEach((val, key) => {
    payload[key] = parseFloat(val) || 0.0;
  });

  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const result = await res.json();

    if (result.status === "success") {
      renderPredictionResult(result);
    } else {
      alert("Error running prediction: " + result.message);
    }
  } catch (err) {
    console.error("Prediction error:", err);
    alert("Connection error during prediction.");
  } finally {
    btn.textContent = "Evaluate Transaction Risk";
    btn.disabled = false;
  }
}

function renderPredictionResult(data) {
  document.getElementById("prediction-result-placeholder").style.display = "none";
  document.getElementById("prediction-result-content").style.display = "block";

  // Score Gauge
  const scoreVal = document.getElementById("res-score-value");
  const scoreLabel = document.getElementById("res-score-label");
  const gaugeCircle = document.getElementById("gauge-circle");

  scoreVal.textContent = data.risk_score;
  scoreLabel.textContent = data.risk_category;
  gaugeCircle.style.setProperty("--score-pct", data.risk_score);
  gaugeCircle.style.setProperty("--accent-color", data.color);

  // Decision & Metrics
  const decBadge = document.getElementById("res-decision-badge");
  decBadge.textContent = data.decision;
  decBadge.className = data.decision === "ACCEPT" ? "badge badge-accept" : "badge badge-review";

  const decText = document.getElementById("res-decision-text");
  decText.textContent = `${data.decision} (${data.status_label})`;
  decText.style.color = data.color;

  document.getElementById("res-prob-value").textContent = data.probability.toFixed(6);
  document.getElementById("res-threshold-value").textContent = data.threshold.toFixed(4);
  document.getElementById("res-latency-value").textContent = `${data.inference_latency_ms} ms`;

  // Disclaimer Box
  const disEl = document.getElementById("res-disclaimer-text");
  if (data.decision === "ACCEPT") {
    disEl.innerHTML = `<strong>Operational Clearance Notice:</strong> ${data.legitimacy_disclaimer}`;
    document.getElementById("res-disclaimer-box").className = "notice-box";
  } else {
    disEl.innerHTML = `<strong>Triage Advisory:</strong> ${data.description}`;
    document.getElementById("res-disclaimer-box").className = "notice-box warning";
  }

  // Contributing Factors List
  const listEl = document.getElementById("res-factors-list");
  let factorsHtml = "";
  if (data.top_contributing_factors && data.top_contributing_factors.length > 0) {
    data.top_contributing_factors.forEach((f) => {
      const pct = Math.min(100, Math.max(10, Math.round(f.relative_impact * 100)));
      factorsHtml += `
        <div class="factor-item">
          <div class="factor-name">
            <span style="color: var(--red-high);">&bull;</span>
            ${f.feature_name}
            <span style="font-size: 10px; color: var(--text-muted);">(val: ${f.feature_value})</span>
          </div>
          <div class="factor-bar-container">
            <div class="factor-bar positive" style="width: ${pct}%;"></div>
          </div>
          <div class="factor-value positive">+${f.shap_value.toFixed(3)}</div>
        </div>
      `;
    });
  } else {
    factorsHtml = `<div style="font-size: 11px; color: var(--text-dim);">No features pushed prediction toward fraud.</div>`;
  }
  listEl.innerHTML = factorsHtml;

  // Update button visibility
  const queueBtn = document.getElementById("btn-queue-action");
  if (queueBtn) {
    queueBtn.style.display = data.is_flagged ? "inline-flex" : "none";
  }
}

function sendToInvestigationQueue() {
  alert("Transaction has been logged and queued in the Investigation Queue.");
  switchTab("queue");
  loadQueueData();
}

// ---------------------------------------------------------------------------
// Overview Data
// ---------------------------------------------------------------------------
async function loadOverviewData(silent = false) {
  try {
    const res = await fetch("/api/overview");
    const json = await res.json();
    if (json.status !== "success") return;

    const kpis = json.kpis;
    document.getElementById("kpi-total-processed").textContent = kpis.total_processed.toLocaleString();
    document.getElementById("kpi-suspected-count").textContent = kpis.suspected_fraud_count.toLocaleString();
    document.getElementById("kpi-high-risk-count").textContent = kpis.high_risk_count.toLocaleString();
    document.getElementById("kpi-precision-recall").textContent = `${kpis.model_precision}% / ${kpis.model_recall}%`;

    // Dynamic topbar bindings
    const f2El = document.getElementById("topbar-f2");
    if (f2El && kpis.model_f2) f2El.textContent = kpis.model_f2.toFixed(4);
    const praucEl = document.getElementById("topbar-prauc");
    if (praucEl && kpis.model_pr_auc) praucEl.textContent = kpis.model_pr_auc.toFixed(4);
    const precEl = document.getElementById("topbar-precision");
    if (precEl && kpis.model_precision) precEl.textContent = `${kpis.model_precision}%`;
    const recEl = document.getElementById("topbar-recall");
    if (recEl && kpis.model_recall) recEl.textContent = `${kpis.model_recall}%`;

    // Render Recent Stream in Overview
    const streamContainer = document.getElementById("overview-recent-stream");
    if (streamContainer && json.recent_activity) {
      let streamHtml = "";
      json.recent_activity.slice(0, 5).forEach((item) => {
        const isFraud = item.is_flagged || item.risk_score >= 30;
        const color = isFraud ? "var(--red-high)" : "var(--green-low)";
        const safeId = escapeHtml(item.id);
        const safeCat = escapeHtml(item.risk_category);
        const safeTime = escapeHtml(String(item.timestamp || "").split(" ")[1] || "");
        streamHtml += `
          <div style="display: flex; align-items: center; justify-content: space-between; padding: 8px 12px; background: rgba(0,0,0,0.25); border-radius: 6px; border: 1px solid var(--border-subtle); font-size: 11px;">
            <div style="display: flex; align-items: center; gap: 8px;">
              <span style="font-family: var(--font-mono); font-weight: 700; color: #fff;">${safeId}</span>
              <span class="badge ${item.risk_score >= 71 ? "badge-high" : item.risk_score >= 31 ? "badge-medium" : "badge-low"}">${safeCat}</span>
            </div>
            <div style="display: flex; align-items: center; gap: 14px;">
              <span class="mono">$${parseFloat(item.amount).toFixed(2)}</span>
              <span class="mono" style="color: ${color}; font-weight: 700;">Score: ${parseInt(item.risk_score, 10)}</span>
              <span style="color: var(--text-muted); font-size: 10px;">${safeTime}</span>
            </div>
          </div>
        `;
      });
      streamContainer.innerHTML = streamHtml || "<div style='color: var(--text-dim);'>No recent activity.</div>";
    }
  } catch (e) {
    if (!silent) console.warn("Could not load overview data:", e);
  }
}

// ---------------------------------------------------------------------------
// Investigation Queue Data & Actions
// ---------------------------------------------------------------------------
async function loadQueueData() {
  const statusFilter = document.getElementById("queue-status-filter")?.value || "";
  const sortBy = document.getElementById("queue-sort-by")?.value || "risk_score";

  try {
    const url = `/api/queue?status=${encodeURIComponent(statusFilter)}&sort=${sortBy}&order=desc`;
    const res = await fetch(url);
    const json = await res.json();
    if (json.status !== "success") return;

    // Update queue badge count
    const badge = document.getElementById("badge-queue-count");
    if (badge) badge.textContent = json.total_flagged;

    const tbody = document.getElementById("queue-table-body");
    if (!tbody) return;

    if (json.items.length === 0) {
      tbody.innerHTML = `<tr><td colspan="10" style="text-align: center; padding: 30px; color: var(--text-muted);">No transactions match filter.</td></tr>`;
      return;
    }

    let rowsHtml = "";
    json.items.forEach((item) => {
      const scoreBadge = item.risk_score >= 71 ? "badge-high" : item.risk_score >= 31 ? "badge-medium" : "badge-low";
      const statusClass = item.status === "CONFIRMED_FRAUD" ? "badge-high" : item.status === "DISMISSED_FALSE_ALARM" ? "badge-low" : "badge-medium";
      const priorityTier = escapeHtml(item.priority_tier || (item.risk_score >= 71 ? "P1 - CRITICAL" : item.risk_score >= 31 ? "P2 - HIGH" : "P3 - MEDIUM"));
      const priorityClass = escapeHtml(item.badge_class || (item.risk_score >= 71 ? "priority-p1" : item.risk_score >= 31 ? "priority-p2" : "priority-p3"));
      const safeId = escapeHtml(item.id);
      const safeTime = escapeHtml(item.timestamp);
      const safeCat = escapeHtml(item.risk_category);
      const safeDriver = escapeHtml(item.top_driver || "V14");
      const safeStatus = escapeHtml(String(item.status || "").replace("_", " "));

      rowsHtml += `
        <tr>
          <td class="mono"><strong>${safeId}</strong></td>
          <td style="color: var(--text-dim); font-size: 11px;">${safeTime}</td>
          <td class="mono">$${parseFloat(item.amount).toFixed(2)}</td>
          <td><span class="badge-priority ${priorityClass}">${priorityTier}</span></td>
          <td><span class="badge ${scoreBadge}">${parseInt(item.risk_score, 10)}/100</span></td>
          <td><span style="font-size: 11px; font-weight: 600;">${safeCat}</span></td>
          <td class="mono">${parseFloat(item.probability).toFixed(4)}</td>
          <td><span class="mono" style="color: var(--cyan-accent);">${safeDriver}</span></td>
          <td><span class="badge ${statusClass}">${safeStatus}</span></td>
          <td>
            <button class="btn btn-secondary btn-sm" onclick="openTransactionModal('${safeId}')">
              Review
            </button>
          </td>
        </tr>
      `;
    });
    tbody.innerHTML = rowsHtml;
  } catch (e) {
    console.error("Queue load error:", e);
  }
}

async function openTransactionModal(txId) {
  currentModalTxId = txId;
  const res = await fetch("/api/queue");
  const json = await res.json();
  const tx = (json.items || []).find((i) => i.id === txId);

  if (!tx) {
    alert("Transaction details not found.");
    return;
  }

  const safeId = escapeHtml(tx.id);
  const safeCat = escapeHtml(tx.risk_category);
  const safeStatus = escapeHtml(tx.status);
  const safeTime = escapeHtml(tx.timestamp);

  document.getElementById("modal-tx-title").textContent = `Investigation Review — ${safeId}`;

  const bodyEl = document.getElementById("modal-tx-body");
  let factorsHtml = "";
  if (tx.top_factors && tx.top_factors.length > 0) {
    tx.top_factors.forEach((f) => {
      const featName = escapeHtml(f.feature_name);
      const featVal = parseFloat(f.feature_value).toFixed(3);
      const shapVal = parseFloat(f.shap_value).toFixed(3);
      factorsHtml += `
        <div style="display: flex; justify-content: space-between; font-size: 11px; padding: 4px 8px; background: rgba(0,0,0,0.3); border-radius: 4px; margin-bottom: 4px;">
          <span class="mono" style="color: #fff;">${featName} (val: ${featVal})</span>
          <span class="mono" style="color: var(--red-high); font-weight: 700;">+${shapVal}</span>
        </div>
      `;
    });
  } else {
    factorsHtml = `<span style="color: var(--text-dim); font-size: 11px;">Top factor: ${escapeHtml(tx.top_driver || "V14")}</span>`;
  }

  bodyEl.innerHTML = `
    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; font-size: 12px; background: rgba(0,0,0,0.25); padding: 14px; border-radius: 6px;">
      <div><span style="color: var(--text-dim);">Amount:</span><br/><strong class="mono">$${parseFloat(tx.amount).toFixed(2)}</strong></div>
      <div><span style="color: var(--text-dim);">Risk Score:</span><br/><strong class="mono" style="color: var(--red-high);">${parseInt(tx.risk_score, 10)} / 100</strong></div>
      <div><span style="color: var(--text-dim);">Probability:</span><br/><strong class="mono">${parseFloat(tx.probability).toFixed(5)}</strong></div>
      <div><span style="color: var(--text-dim);">Category:</span><br/><strong>${safeCat}</strong></div>
      <div><span style="color: var(--text-dim);">Triage Status:</span><br/><strong>${safeStatus}</strong></div>
      <div><span style="color: var(--text-dim);">Timestamp:</span><br/><span style="font-size: 11px;">${safeTime}</span></div>
    </div>

    <div>
      <div style="font-size: 12px; font-weight: 700; color: #fff; margin-bottom: 6px;">Top SHAP Risk Drivers</div>
      ${factorsHtml}
    </div>

    <div class="notice-box">
      <strong>Analyst Policy:</strong> Confirming fraud triggers card block and initiates zero-liability dispute protocol. Dismissing clears the transaction under operational acceptance rules.
    </div>
  `;

  document.getElementById("investigation-modal").classList.add("active");
}

function closeModal() {
  document.getElementById("investigation-modal").classList.remove("active");
  currentModalTxId = null;
}

async function handleModalAction(action) {
  if (!currentModalTxId) return;

  try {
    const res = await fetch(`/api/queue/${currentModalTxId}/action`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ status: action }),
    });
    const json = await res.json();
    if (json.status === "success") {
      closeModal();
      loadQueueData();
    } else {
      alert("Error: " + json.message);
    }
  } catch (e) {
    console.error("Action error:", e);
  }
}

// ---------------------------------------------------------------------------
// Model Performance Data & Charts
// ---------------------------------------------------------------------------
async function loadPerformanceData() {
  try {
    const res = await fetch("/api/performance");
    const json = await res.json();
    if (json.status !== "success") return;

    // Render Curves via Chart.js
    if (json.curves && json.curves.precision_curve) {
      renderCurvesChart(json.curves);
    }
  } catch (e) {
    console.warn("Performance data error:", e);
  }
}

function renderCurvesChart(curves) {
  const ctx = document.getElementById("chart-pr-curve")?.getContext("2d");
  if (!ctx) return;

  if (charts.prCurve) charts.prCurve.destroy();

  const labels = curves.recall_curve.map((r) => r.toFixed(2));
  charts.prCurve = new Chart(ctx, {
    type: "line",
    data: {
      labels: labels,
      datasets: [
        {
          label: "Precision-Recall Curve (PR-AUC: 0.8178)",
          data: curves.precision_curve,
          borderColor: "#3b82f6",
          backgroundColor: "rgba(59, 130, 246, 0.1)",
          borderWidth: 2,
          fill: true,
          tension: 0.2,
          pointRadius: 1,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { labels: { color: "#e2e8f0", font: { family: "Inter", size: 10 } } },
      },
      scales: {
        x: {
          title: { display: true, text: "Recall", color: "#94a3b8", font: { size: 10 } },
          ticks: { color: "#64748b", font: { size: 9 }, maxTicksLimit: 8 },
          grid: { color: "rgba(255, 255, 255, 0.05)" },
        },
        y: {
          title: { display: true, text: "Precision", color: "#94a3b8", font: { size: 10 } },
          ticks: { color: "#64748b", font: { size: 9 } },
          grid: { color: "rgba(255, 255, 255, 0.05)" },
          min: 0.5,
          max: 1.0,
        },
      },
    },
  });
}

// ---------------------------------------------------------------------------
// Explainability Data & Charts
// ---------------------------------------------------------------------------
async function loadExplainabilityData() {
  try {
    const res = await fetch("/api/explainability");
    const json = await res.json();
    if (json.status !== "success") return;

    if (json.top_features && json.top_features.length > 0) {
      renderGlobalImportanceChart(json.top_features);
    }
  } catch (e) {
    console.warn("Explainability load error:", e);
  }
}

function renderGlobalImportanceChart(features) {
  const ctx = document.getElementById("chart-global-importance")?.getContext("2d");
  if (!ctx) return;

  if (charts.globalImportance) charts.globalImportance.destroy();

  const topItems = features.slice(0, 8).reverse();
  const labels = topItems.map((f) => f.feature);
  const data = topItems.map((f) => f.mean_abs_shap);

  charts.globalImportance = new Chart(ctx, {
    type: "bar",
    data: {
      labels: labels,
      datasets: [
        {
          label: "Mean |SHAP Value|",
          data: data,
          backgroundColor: "#3b82f6",
          borderRadius: 4,
        },
      ],
    },
    options: {
      indexAxis: "y",
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
      },
      scales: {
        x: {
          ticks: { color: "#64748b", font: { size: 9 } },
          grid: { color: "rgba(255, 255, 255, 0.05)" },
        },
        y: {
          ticks: { color: "#e2e8f0", font: { family: "JetBrains Mono", size: 10 } },
          grid: { display: false },
        },
      },
    },
  });
}

// ---------------------------------------------------------------------------
// Model Monitoring Data & Charts
// ---------------------------------------------------------------------------
async function loadMonitoringData(silent = false) {
  try {
    const res = await fetch("/api/monitoring");
    const json = await res.json();
    if (json.status !== "success") return;

    const t = json.telemetry;
    const psi = t.psi;

    const psiScoreEl = document.getElementById("mon-psi-score");
    const psiStatusEl = document.getElementById("mon-psi-status");
    if (psiScoreEl) psiScoreEl.textContent = psi.psi_score.toFixed(4);
    if (psiStatusEl) {
      psiStatusEl.textContent = `${psi.status} · PSI < 0.10`;
      psiStatusEl.style.color = psi.color;
    }

    const fraudRateEl = document.getElementById("mon-fraud-rate");
    if (fraudRateEl) fraudRateEl.textContent = `${t.fraud_rate_pct}%`;

    const latEl = document.getElementById("mon-latency");
    if (latEl) latEl.textContent = `${t.system_health.mean_latency_ms} ms`;

    // Production health dynamic telemetry bindings
    const driftStatusEl = document.getElementById("health-drift-status");
    const driftPsiEl = document.getElementById("health-drift-psi");
    if (driftStatusEl) {
      driftStatusEl.textContent = psi.status === "STABLE" ? "No Covariate Shift Detected" : "Covariate Drift Warning";
      driftStatusEl.style.color = psi.color;
    }
    if (driftPsiEl) {
      driftPsiEl.textContent = `Current PSI: ${psi.psi_score.toFixed(4)} (Threshold: 0.20)`;
    }

    const sessionTxEl = document.getElementById("health-session-tx");
    const avgLatEl = document.getElementById("health-avg-latency");
    if (sessionTxEl) {
      sessionTxEl.textContent = `${t.total_inferences} Inferences Logged`;
    }
    if (avgLatEl) {
      avgLatEl.textContent = `Mean Latency: ${t.system_health.mean_latency_ms} ms`;
    }

    // Render Histogram & Doughnut
    if (t.probability_histogram) {
      renderHistogramChart(t.probability_histogram);
    }
    renderClassDoughnutChart(t);
  } catch (e) {
    if (!silent) console.warn("Monitoring data error:", e);
  }
}

function renderHistogramChart(hist) {
  const ctx = document.getElementById("chart-prob-histogram")?.getContext("2d");
  if (!ctx) return;

  if (charts.histogram) charts.histogram.destroy();

  charts.histogram = new Chart(ctx, {
    type: "bar",
    data: {
      labels: hist.labels,
      datasets: [
        {
          label: "Inference Volume",
          data: hist.counts,
          backgroundColor: "#06b6d4",
          borderRadius: 3,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: {
          ticks: { color: "#64748b", font: { size: 9 } },
          grid: { display: false },
        },
        y: {
          ticks: { color: "#64748b", font: { size: 9 } },
          grid: { color: "rgba(255, 255, 255, 0.05)" },
        },
      },
    },
  });
}

function renderClassDoughnutChart(t) {
  const ctx = document.getElementById("chart-class-doughnut")?.getContext("2d");
  if (!ctx) return;

  if (charts.doughnut) charts.doughnut.destroy();

  charts.doughnut = new Chart(ctx, {
    type: "doughnut",
    data: {
      labels: ["Low Risk (0-30)", "Medium Risk (31-70)", "High Risk (71-100)"],
      datasets: [
        {
          data: [t.low_risk_count, t.medium_risk_count, t.high_risk_count],
          backgroundColor: ["#10b981", "#f59e0b", "#ef4444"],
          borderWidth: 0,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: "bottom",
          labels: { color: "#e2e8f0", font: { family: "Inter", size: 10 } },
        },
      },
      cutout: "70%",
    },
  });
}

// ---------------------------------------------------------------------------
// SECTION 7: Adaptive Intelligence & What-If Simulation
// ---------------------------------------------------------------------------
let adaptiveTradeoffData = null;

async function loadAdaptiveData() {
  try {
    // 1. Fetch What-If Trade-Off Data
    const tradeoffRes = await fetch("/api/adaptive/what-if");
    const tradeoffJson = await tradeoffRes.json();
    if (tradeoffJson.status === "success") {
      adaptiveTradeoffData = tradeoffJson;
      renderDemoTable(tradeoffJson.demonstration_table || []);
      renderCostCurveChart(tradeoffJson.curve_grid || [], tradeoffJson.cost_optimal_threshold);

      // Update active threshold badge and slider
      const activeT = tradeoffJson.active_threshold || 0.2773;
      const activeBadge = document.getElementById("adaptive-active-badge");
      if (activeBadge) activeBadge.textContent = `ACTIVE THRESHOLD: ${activeT.toFixed(4)}`;

      const slider = document.getElementById("adaptive-threshold-slider");
      if (slider && !slider.dataset.userInteracted) {
        slider.value = activeT;
        onThresholdSliderChange(activeT);
      }
    }

    // 2. Fetch Multi-Feature Data Drift
    const driftRes = await fetch("/api/adaptive/drift");
    const driftJson = await driftRes.json();
    if (driftJson.status === "success") {
      renderMultiDriftTable(driftJson.features || []);
      const statusBadge = document.getElementById("multi-drift-status-badge");
      if (statusBadge) {
        statusBadge.textContent = `DRIFT STATUS: ${driftJson.overall_drift_status}`;
        statusBadge.className = driftJson.overall_drift_status === "STABLE" ? "badge badge-low" : "badge badge-high";
      }
    }

    // 3. Fetch Risk Distribution & Quantiles
    const distRes = await fetch("/api/adaptive/risk-distribution");
    const distJson = await distRes.json();
    if (distJson.status === "success") {
      renderRiskDistributionChart(distJson.histogram);

      // Update quantiles
      const q = distJson.quantiles || {};
      if (document.getElementById("q-p10")) document.getElementById("q-p10").textContent = q.p10 ?? 0;
      if (document.getElementById("q-p50")) document.getElementById("q-p50").textContent = q.p50 ?? 1;
      if (document.getElementById("q-p90")) document.getElementById("q-p90").textContent = q.p90 ?? 14;
      if (document.getElementById("q-p999")) document.getElementById("q-p999").textContent = q.p99_9 ?? 96;

      // Update category percentages
      const cat = distJson.category_summary || {};
      if (cat.low_risk && document.getElementById("cat-low-pct")) {
        document.getElementById("cat-low-pct").textContent = `${cat.low_risk.pct}% (${cat.low_risk.count.toLocaleString()})`;
      }
      if (cat.medium_risk && document.getElementById("cat-med-pct")) {
        document.getElementById("cat-med-pct").textContent = `${cat.medium_risk.pct}% (${cat.medium_risk.count.toLocaleString()})`;
      }
      if (cat.high_risk && document.getElementById("cat-high-pct")) {
        document.getElementById("cat-high-pct").textContent = `${cat.high_risk.pct}% (${cat.high_risk.count.toLocaleString()})`;
      }
    }
  } catch (err) {
    console.error("Adaptive data loading error:", err);
  }
}

function onThresholdSliderChange(val) {
  const t = parseFloat(val);
  const display = document.getElementById("slider-threshold-display");
  if (display) display.textContent = t.toFixed(4);

  const slider = document.getElementById("adaptive-threshold-slider");
  if (slider) slider.dataset.userInteracted = "true";

  if (!adaptiveTradeoffData || !adaptiveTradeoffData.curve_grid) return;

  // Find closest threshold in evaluated curve grid
  const grid = adaptiveTradeoffData.curve_grid;
  const closest = grid.reduce((prev, curr) => {
    return Math.abs(curr.threshold - t) < Math.abs(prev.threshold - t) ? curr : prev;
  });

  if (closest) {
    const recEl = document.getElementById("sim-recall");
    if (recEl) recEl.textContent = `${closest.recall.toFixed(2)}%`;

    const precEl = document.getElementById("sim-precision");
    if (precEl) precEl.textContent = `${closest.precision.toFixed(2)}%`;

    const tpEl = document.getElementById("sim-tp");
    if (tpEl) tpEl.textContent = `${closest.tp} Fraud Caught / ${closest.tp + closest.fn} Total`;

    const fpEl = document.getElementById("sim-fp");
    if (fpEl) fpEl.textContent = `${closest.fp} False Alarms (Analyst Queue)`;

    const fcEl = document.getElementById("sim-fraud-caught");
    if (fcEl) fcEl.textContent = `$${closest.fraud_caught_dollars.toLocaleString()}`;

    const feEl = document.getElementById("sim-fraud-escaped");
    if (feEl) feEl.textContent = `$${closest.fraud_escaped_dollars.toLocaleString()} Escaped Loss`;

    const tcEl = document.getElementById("sim-total-cost");
    if (tcEl) tcEl.textContent = `$${closest.net_financial_cost.toLocaleString()}`;

    const cbEl = document.getElementById("sim-cost-breakdown");
    if (cbEl) cbEl.textContent = `$${closest.alert_review_cost} Review + $${closest.fraud_escaped_dollars} Fraud Loss`;
  }
}

async function applySimulatedThreshold() {
  const slider = document.getElementById("adaptive-threshold-slider");
  if (!slider) return;

  const targetT = parseFloat(slider.value);
  try {
    const res = await fetch("/api/threshold/set", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ threshold: targetT }),
    });
    const json = await res.json();
    if (json.status === "success") {
      alert(`Success! Operating Decision Threshold updated to ${targetT.toFixed(4)}. Live evaluations and queue policies now use this boundary.`);
      
      // Update topbar badge
      const topbarPill = document.getElementById("active-model-pill");
      if (topbarPill) topbarPill.textContent = `XGBoost · Threshold ${targetT.toFixed(4)}`;

      // Update adaptive active badge
      const activeBadge = document.getElementById("adaptive-active-badge");
      if (activeBadge) activeBadge.textContent = `ACTIVE THRESHOLD: ${targetT.toFixed(4)}`;

      loadAdaptiveData();
      loadOverviewData();
      loadQueueData();
    } else {
      alert(`Error updating threshold: ${json.message}`);
    }
  } catch (err) {
    alert(`Failed to set threshold: ${err.message}`);
  }
}

function resetToOptimalThreshold() {
  const slider = document.getElementById("adaptive-threshold-slider");
  if (slider) {
    slider.value = 0.2773;
    slider.dataset.userInteracted = "true";
    onThresholdSliderChange(0.2773);
    applySimulatedThreshold();
  }
}

function renderCostCurveChart(grid, optimalT) {
  const ctx = document.getElementById("chart-cost-curve")?.getContext("2d");
  if (!ctx || !grid || grid.length === 0) return;

  if (charts.costCurve) charts.costCurve.destroy();

  const labels = grid.map((pt) => pt.threshold.toFixed(2));
  const totalCost = grid.map((pt) => pt.net_financial_cost);
  const fraudLoss = grid.map((pt) => pt.fraud_escaped_dollars);
  const reviewCost = grid.map((pt) => pt.alert_review_cost);

  charts.costCurve = new Chart(ctx, {
    type: "line",
    data: {
      labels: labels,
      datasets: [
        {
          label: "Total Net Cost ($)",
          data: totalCost,
          borderColor: "#3b82f6",
          backgroundColor: "rgba(59, 130, 246, 0.1)",
          borderWidth: 2.5,
          tension: 0.3,
          fill: true,
        },
        {
          label: "Escaped Fraud Loss ($)",
          data: fraudLoss,
          borderColor: "#ef4444",
          borderWidth: 1.5,
          borderDash: [4, 4],
          pointRadius: 0,
          tension: 0.3,
        },
        {
          label: "Analyst Review Cost ($)",
          data: reviewCost,
          borderColor: "#f59e0b",
          borderWidth: 1.5,
          borderDash: [4, 4],
          pointRadius: 0,
          tension: 0.3,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: "top",
          labels: { color: "#e2e8f0", font: { family: "Inter", size: 10 } },
        },
        tooltip: {
          callbacks: {
            label: (ctx) => `${ctx.dataset.label}: $${ctx.parsed.y.toLocaleString()}`,
          },
        },
      },
      scales: {
        x: {
          title: { display: true, text: "Decision Threshold (T)", color: "#94a3b8", font: { size: 10 } },
          ticks: { color: "#64748b", font: { size: 9 } },
          grid: { display: false },
        },
        y: {
          title: { display: true, text: "Cost in USD ($)", color: "#94a3b8", font: { size: 10 } },
          ticks: { color: "#64748b", font: { size: 9 } },
          grid: { color: "rgba(255, 255, 255, 0.05)" },
        },
      },
    },
  });
}

function renderDemoTable(rows) {
  const tbody = document.getElementById("demo-tradeoff-body");
  if (!tbody || !rows || rows.length === 0) return;

  let html = "";
  rows.forEach((r) => {
    const isOptimal = Math.abs(r.threshold - 0.2773) < 1e-4;
    const highlightStyle = isOptimal ? "background: rgba(59,130,246,0.1); font-weight: 700;" : "";
    const badgeClass = isOptimal ? "badge-low" : r.threshold <= 0.1 ? "badge-high" : "badge-medium";

    html += `
      <tr style="${highlightStyle}">
        <td class="mono"><strong>${r.threshold.toFixed(4)}</strong></td>
        <td class="mono" style="color: var(--primary);">${r.recall.toFixed(2)}%</td>
        <td class="mono" style="color: var(--green-low);">${r.precision.toFixed(2)}%</td>
        <td class="mono">${r.fp}</td>
        <td class="mono">$${r.net_financial_cost.toLocaleString()}</td>
        <td><span class="badge ${badgeClass}">${r.badge || r.policy_name}</span></td>
      </tr>
    `;
  });
  tbody.innerHTML = html;
}

function renderMultiDriftTable(features) {
  const tbody = document.getElementById("multi-drift-table-body");
  if (!tbody || !features || features.length === 0) return;

  let html = "";
  features.forEach((f) => {
    const badgeClass = f.status === "STABLE" ? "badge-low" : f.status === "MODERATE" ? "badge-med" : "badge-high";

    html += `
      <tr>
        <td class="mono" style="color: var(--cyan-accent);"><strong>${f.feature}</strong></td>
        <td class="mono">${f.ref_mean}</td>
        <td class="mono">${f.stream_mean}</td>
        <td class="mono">${f.ref_std}</td>
        <td class="mono">${f.stream_std}</td>
        <td class="mono">${f.ks_statistic}</td>
        <td class="mono">${f.wasserstein_distance}</td>
        <td class="mono" style="font-weight: 700; color: ${f.color};">${f.psi_score.toFixed(4)}</td>
        <td><span class="badge ${badgeClass}">${f.status}</span></td>
      </tr>
    `;
  });
  tbody.innerHTML = html;
}

function renderRiskDistributionChart(hist) {
  const ctx = document.getElementById("chart-risk-distribution")?.getContext("2d");
  if (!ctx || !hist || !hist.labels) return;

  if (charts.riskDist) charts.riskDist.destroy();

  charts.riskDist = new Chart(ctx, {
    type: "bar",
    data: {
      labels: hist.labels,
      datasets: [
        {
          label: "Transaction Count",
          data: hist.counts,
          backgroundColor: hist.labels.map((_, i) => {
            if (i <= 3) return "#10b981"; // 0-30 Low Risk
            if (i <= 7) return "#f59e0b"; // 31-70 Medium Risk
            return "#ef4444"; // 71-100 High Risk
          }),
          borderRadius: 3,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (ctx) => `${ctx.parsed.y.toLocaleString()} transactions (${hist.percentages[ctx.dataIndex]}%)`,
          },
        },
      },
      scales: {
        x: {
          ticks: { color: "#64748b", font: { size: 9 } },
          grid: { display: false },
        },
        y: {
          ticks: { color: "#64748b", font: { size: 9 } },
          grid: { color: "rgba(255, 255, 255, 0.05)" },
        },
      },
    },
  });
}

