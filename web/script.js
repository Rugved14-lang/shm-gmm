/* =============================================================
   SHM GMM – script.js
   Loads results.json, renders Plotly charts, animates counters,
   waveform hero background, and scroll-triggered fade-ins.
   ============================================================= */

"use strict";

/* ─── Plotly dark theme config ───────────────────────────── */
const DARK_LAYOUT = {
  paper_bgcolor: "rgba(0,0,0,0)",
  plot_bgcolor:  "rgba(0,0,0,0)",
  font: { family: "'Inter', sans-serif", color: "#8fb3d4", size: 12 },
  xaxis: {
    gridcolor:    "rgba(0,229,255,0.07)",
    linecolor:    "rgba(0,229,255,0.15)",
    tickcolor:    "rgba(0,229,255,0.15)",
    zerolinecolor:"rgba(0,229,255,0.12)",
  },
  yaxis: {
    gridcolor:    "rgba(0,229,255,0.07)",
    linecolor:    "rgba(0,229,255,0.15)",
    tickcolor:    "rgba(0,229,255,0.15)",
    zerolinecolor:"rgba(0,229,255,0.12)",
  },
  legend: {
    bgcolor:      "rgba(4,13,26,0.75)",
    bordercolor:  "rgba(0,229,255,0.15)",
    borderwidth:  1,
    font:         { color: "#8fb3d4" },
  },
  margin: { l: 60, r: 24, t: 36, b: 56 },
};

const PLOTLY_CONFIG = {
  displayModeBar: true,
  modeBarButtonsToRemove: ["lasso2d", "select2d", "toImage"],
  displaylogo: false,
  responsive: true,
};

/* ─── Cluster colours matching Plotly qualitative palette ── */
const CLUSTER_COLORS = [
  "#636efa", "#ef553b", "#00cc96", "#ab63fa",
  "#ffa15a", "#19d3f3", "#ff6692", "#b6e880",
];

/* =============================================================
   1. WAVEFORM HERO CANVAS ANIMATION
   ============================================================= */
(function initWaveCanvas() {
  const canvas = document.getElementById("waveCanvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");

  let W, H, animId;

  const waves = [
    { freq: 2.5,  amp: 0.025, speed: 0.6,  phase: 0,    color: "rgba(0,229,255,0.35)",  lw: 1.5 },
    { freq: 7.1,  amp: 0.015, speed: 1.4,  phase: 1.2,  color: "rgba(0,229,255,0.20)",  lw: 1 },
    { freq: 12.4, amp: 0.009, speed: 2.5,  phase: 2.5,  color: "rgba(168,85,247,0.25)", lw: 1 },
    { freq: 1.0,  amp: 0.040, speed: 0.3,  phase: 0.4,  color: "rgba(0,229,255,0.08)",  lw: 2 },
    { freq: 4.0,  amp: 0.012, speed: 0.9,  phase: 3.1,  color: "rgba(168,85,247,0.12)", lw: 1.2 },
  ];

  function resize() {
    W = canvas.width  = canvas.offsetWidth;
    H = canvas.height = canvas.offsetHeight;
  }

  let t = 0;
  function draw() {
    ctx.clearRect(0, 0, W, H);
    const midY = H * 0.52;

    waves.forEach(w => {
      ctx.beginPath();
      ctx.strokeStyle = w.color;
      ctx.lineWidth   = w.lw;

      const pts = 320;
      for (let i = 0; i <= pts; i++) {
        const x = (i / pts) * W;
        const nx = x / W;                 // 0..1
        // multi-frequency damped wave
        const y = midY
          - H * w.amp * Math.sin(2 * Math.PI * w.freq * nx * 2 + t * w.speed + w.phase)
          * Math.exp(-1.5 * nx)           // gentle damping envelope
          - H * 0.008 * Math.sin(2 * Math.PI * w.freq * 3 * nx + t * w.speed * 0.7);
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      }
      ctx.stroke();
    });

    t += 0.012;
    animId = requestAnimationFrame(draw);
  }

  resize();
  draw();
  window.addEventListener("resize", () => { resize(); });
})();

/* =============================================================
   2. SCROLL FADE-IN  (IntersectionObserver)
   ============================================================= */
(function initScrollFade() {
  const els = document.querySelectorAll(".fade-in");
  const obs = new IntersectionObserver(
    entries => {
      entries.forEach(e => {
        if (e.isIntersecting) {
          e.target.classList.add("visible");
          obs.unobserve(e.target);
        }
      });
    },
    { threshold: 0.12 }
  );
  els.forEach(el => obs.observe(el));
})();

/* =============================================================
   3. COUNT-UP ANIMATION
   ============================================================= */
function animateCounter(el, target, duration = 1400) {
  const start = performance.now();
  const isLarge = target >= 1000;
  function step(now) {
    const p = Math.min((now - start) / duration, 1);
    const ease = 1 - Math.pow(1 - p, 3);  // ease-out cubic
    const val = Math.round(ease * target);
    el.textContent = isLarge ? val.toLocaleString() : val;
    if (p < 1) requestAnimationFrame(step);
    else el.textContent = isLarge ? target.toLocaleString() : target;
  }
  requestAnimationFrame(step);
}

function initCounters(data) {
  const meta = data.meta;
  const best_n = data.bic.best_n;
  const nModes = data.cluster_summary.length;

  const map = {
    "stat-modes":   nModes,
    "stat-gmm":     best_n,
    "stat-samples": meta.n_samples,
    "stat-fs":      meta.fs_hz,
  };

  // Use IntersectionObserver to trigger when stat cards come into view
  const section = document.getElementById("stats");
  const obs = new IntersectionObserver(entries => {
    if (entries[0].isIntersecting) {
      Object.entries(map).forEach(([id, val]) => {
        const el = document.getElementById(id);
        if (el) animateCounter(el, val);
      });
      obs.disconnect();
    }
  }, { threshold: 0.3 });
  if (section) obs.observe(section);
}

/* =============================================================
   4. PLOTLY CHARTS
   ============================================================= */

/* ── 4a. PSD + Peaks ────────────────────────────────────── */
function renderPSD(data) {
  const { freqs, psd } = data.psd;
  const peaks = data.peaks;
  const clusters = data.cluster_summary;

  // Assign colours by cluster
  const peakColors  = peaks.map(p => CLUSTER_COLORS[p.cluster % CLUSTER_COLORS.length]);
  const peakLabels  = peaks.map(p => `Mode ${p.cluster + 1}`);
  const peakDamp    = peaks.map(p => (p.damping_ratio * 100).toFixed(3));

  const traces = [
    {
      type:   "scatter",
      x:      freqs,
      y:      psd,
      mode:   "lines",
      name:   "PSD",
      line:   { color: "rgba(0,229,255,0.75)", width: 1.4 },
      hovertemplate: "f = %{x:.3f} Hz<br>PSD = %{y:.3e}<extra>PSD</extra>",
    },
    {
      type:         "scatter",
      x:            peaks.map(p => p.frequency_hz),
      y:            peaks.map(p => p.psd_amplitude),
      mode:         "markers",
      name:         "Detected Peaks",
      marker: {
        color:  peakColors,
        size:   11,
        symbol: "triangle-up",
        line:   { color: "rgba(255,255,255,0.6)", width: 1 },
      },
      text:         peakLabels,
      customdata:   peakDamp,
      hovertemplate:
        "<b>%{text}</b><br>f₀ = %{x:.3f} Hz<br>PSD = %{y:.3e}<br>ζ = %{customdata} %<extra></extra>",
    },
  ];

  const layout = {
    ...DARK_LAYOUT,
    xaxis: { ...DARK_LAYOUT.xaxis, title: "Frequency (Hz)" },
    yaxis: { ...DARK_LAYOUT.yaxis, title: "PSD [(units)²/Hz]", type: "log" },
    height: 380,
    showlegend: true,
  };

  Plotly.newPlot("chart-psd", traces, layout, PLOTLY_CONFIG);
}

/* ── 4b. Cluster Scatter ────────────────────────────────── */
function renderScatter(data) {
  const peaks = data.peaks;
  const clusterMap = {};
  peaks.forEach(p => {
    if (!clusterMap[p.cluster]) clusterMap[p.cluster] = [];
    clusterMap[p.cluster].push(p);
  });

  const traces = Object.keys(clusterMap).sort((a,b) => +a - +b).map(cid => {
    const group  = clusterMap[cid];
    const color  = CLUSTER_COLORS[+cid % CLUSTER_COLORS.length];
    const sizes  = group.map(p => Math.max(10, p.damping_ratio * 5000 + 8));
    return {
      type:         "scatter",
      x:            group.map(p => p.frequency_hz),
      y:            group.map(p => p.log_amplitude),
      mode:         "markers",
      name:         `Mode ${+cid + 1}`,
      marker: {
        color,
        size:     sizes,
        opacity:  0.80,
        line:     { color: "rgba(255,255,255,0.4)", width: 1 },
      },
      customdata:   group.map(p => (p.damping_ratio * 100).toFixed(3)),
      hovertemplate:
        `<b>Mode ${+cid + 1}</b><br>f₀ = %{x:.3f} Hz<br>log₁₀(PSD) = %{y:.3f}<br>ζ = %{customdata} %<extra></extra>`,
    };
  });

  const layout = {
    ...DARK_LAYOUT,
    xaxis: { ...DARK_LAYOUT.xaxis, title: "Frequency (Hz)" },
    yaxis: { ...DARK_LAYOUT.yaxis, title: "log₁₀(PSD Amplitude)" },
    height: 380,
    showlegend: true,
    legend: { ...DARK_LAYOUT.legend, title: { text: "Mode Cluster" } },
  };

  Plotly.newPlot("chart-scatter", traces, layout, PLOTLY_CONFIG);
}

/* ── 4c. BIC Curve ──────────────────────────────────────── */
function renderBIC(data) {
  const { n_range, bic_scores, best_n } = data.bic;

  const traces = [
    {
      type:         "scatter",
      x:            n_range,
      y:            bic_scores,
      mode:         "lines+markers",
      name:         "BIC",
      line:         { color: "#a855f7", width: 2.5 },
      marker:       { size: 8, color: "#a855f7" },
      hovertemplate: "n = %{x}<br>BIC = %{y:.2f}<extra>BIC</extra>",
    },
    {
      type:         "scatter",
      x:            [best_n],
      y:            [bic_scores[best_n - 1]],
      mode:         "markers",
      name:         `Best (n=${best_n})`,
      marker: {
        size:   18,
        color:  "#00e5ff",
        symbol: "star",
        line:   { color: "#ffffff", width: 1.5 },
      },
      hovertemplate: `<b>Best Model</b><br>n = ${best_n}<br>BIC = %{y:.2f}<extra></extra>`,
    },
  ];

  const layout = {
    ...DARK_LAYOUT,
    xaxis: {
      ...DARK_LAYOUT.xaxis,
      title: "Number of GMM Components",
      tickmode: "linear", tick0: 1, dtick: 1,
    },
    yaxis: { ...DARK_LAYOUT.yaxis, title: "BIC Score" },
    height: 320,
    showlegend: true,
    legend: { ...DARK_LAYOUT.legend, orientation: "h", y: 1.12 },
  };

  Plotly.newPlot("chart-bic", traces, layout, PLOTLY_CONFIG);
}

/* =============================================================
   5. MODE TABLE
   ============================================================= */
function renderModeTable(data) {
  const tbody = document.getElementById("mode-table-body");
  if (!tbody) return;

  const gt = {};
  (data.ground_truth_modes || []).forEach((m, i) => { gt[i] = m; });
  const clusters = data.cluster_summary;

  clusters.forEach((c, i) => {
    const color = CLUSTER_COLORS[c.cluster % CLUSTER_COLORS.length];
    const gtEntry = gt[i];
    const gtFreq  = gtEntry ? `${gtEntry.freq_hz.toFixed(1)} Hz` : "—";

    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><span class="mode-dot" style="background:${color}"></span>${c.label}</td>
      <td class="mono-cell">${c.mean_freq_hz.toFixed(3)} Hz</td>
      <td class="mono-cell">${(c.std_freq_hz || 0).toFixed(4)} Hz</td>
      <td class="mono-cell">${c.mean_log_amp.toFixed(3)}</td>
      <td class="mono-cell">${c.mean_damping_pct.toFixed(3)} %</td>
      <td class="mono-cell">${c.count}</td>
      <td class="mono-cell" style="color:var(--cyan)">${gtFreq}</td>
    `;
    tbody.appendChild(tr);
  });
}

/* =============================================================
   6. MAIN — fetch JSON then render everything
   ============================================================= */
async function main() {
  let data;
  try {
    const res = await fetch("data/results.json");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    data = await res.json();
  } catch (err) {
    console.error("Failed to load results.json:", err);
    document.querySelectorAll(".chart-plot").forEach(el => {
      el.innerHTML = `<p style="color:var(--rose);padding:2rem;text-align:center">
        ⚠ Could not load data/results.json.<br>
        Run <code>py generate_results.py</code> then serve via a local web server.</p>`;
    });
    return;
  }

  // Wait until Plotly is ready (deferred script)
  if (typeof Plotly === "undefined") {
    await new Promise(resolve => {
      const check = () => typeof Plotly !== "undefined" ? resolve() : setTimeout(check, 50);
      check();
    });
  }

  renderPSD(data);
  renderScatter(data);
  renderBIC(data);
  renderModeTable(data);
  initCounters(data);

  // Re-render charts on window resize (Plotly responsive handles most, but relayout smooths it)
  let resizeTimer;
  window.addEventListener("resize", () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => {
      ["chart-psd", "chart-scatter", "chart-bic"].forEach(id => {
        const el = document.getElementById(id);
        if (el && el.data) Plotly.relayout(id, { autosize: true });
      });
    }, 200);
  });
}

// Kick off as soon as DOM is ready
if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", main);
} else {
  main();
}
