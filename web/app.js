const seedAuctions = [
  {
    date: "2026-01-07",
    rates: { 91: 15.8, 182: 16.5, 364: 18.47 },
    signal: "Higher rates across the curve"
  },
  {
    date: "2026-03-04",
    rates: { 91: 15.95, 182: 16.65, 364: 16.73 },
    signal: "Long-end demand stayed dominant"
  },
  {
    date: "2026-03-25",
    rates: { 91: 15.95, 182: 16.42, 364: 16.43 },
    signal: "Rates softened on liquidity"
  },
  {
    date: "2026-05-06",
    rates: { 91: 15.949, 182: 16.14, 364: 16.15 },
    signal: "Marginal decline across tenors"
  },
  {
    date: "2026-05-20",
    rates: { 91: 16.5, 182: 17.45, 364: 20.69 },
    signal: "Sharp long-end repricing"
  }
];

const defaults = {
  tenor: "364",
  mpr: 27.5,
  inflation: 23.7,
  offer: 397,
  subscription: 1020,
  liquidity: "balanced",
  fxPressure: "medium"
};

const els = {
  form: document.querySelector("#predictor-form"),
  reset: document.querySelector("#reset-button"),
  tenor: document.querySelector("#tenor"),
  mpr: document.querySelector("#mpr"),
  inflation: document.querySelector("#inflation"),
  offer: document.querySelector("#offer"),
  subscription: document.querySelector("#subscription"),
  liquidity: document.querySelector("#liquidity"),
  fxPressure: document.querySelector("#fxPressure"),
  predictedRate: document.querySelector("#predicted-rate"),
  confidence: document.querySelector("#confidence"),
  latestRate: document.querySelector("#latest-rate"),
  bidCover: document.querySelector("#bid-cover"),
  badge: document.querySelector("#direction-badge"),
  reading: document.querySelector("#model-reading"),
  table: document.querySelector("#auction-table"),
  chart: document.querySelector("#curve-chart")
};

// Guard: warn if any element is missing (catches ID typos early)
{
  const missing = Object.entries(els)
    .filter(([, v]) => v === null)
    .map(([k]) => k);
  if (missing.length) {
    console.error("NTB Predictor — missing DOM elements:", missing.join(", "));
  }
}

// latest is a live reference so it picks up any API-refreshed entry
function getLatest() {
  return seedAuctions[seedAuctions.length - 1];
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function pct(value) {
  return `${value.toFixed(2)}%`;
}

function readInputs() {
  return {
    tenor: els.tenor.value,
    mpr: Number(els.mpr.value),
    inflation: Number(els.inflation.value),
    offer: Math.max(1, Number(els.offer.value)),
    subscription: Math.max(1, Number(els.subscription.value)),
    liquidity: els.liquidity.value,
    fxPressure: els.fxPressure.value
  };
}

// Use mean of consecutive differences instead of first-to-last slope
// so all intermediate data points influence the trend.
function movingTrend(tenor) {
  const recent = seedAuctions.slice(-4).map((row) => row.rates[tenor]);
  const diffs = recent.slice(1).map((v, i) => v - recent[i]);
  return diffs.reduce((a, b) => a + b, 0) / diffs.length;
}

function predictRate(inputs, tenor) {
  const base = getLatest().rates[tenor];
  const bidCover = inputs.subscription / inputs.offer;
  const trend = movingTrend(tenor);
  const policyGap = inputs.mpr - base;
  const realRateGap = base - inputs.inflation;

  const demandAdjustment     = clamp((1.8 - bidCover) * 0.55, -0.9, 0.9);
  const policyAdjustment     = clamp(policyGap * 0.035, -0.55, 0.55);
  const inflationAdjustment  = clamp(-realRateGap * 0.025, -0.35, 0.55);
  const liquidityAdjustment  = { tight: 0.45, balanced: 0, flush: -0.45 }[inputs.liquidity];
  const fxAdjustment         = { low: -0.15, medium: 0, high: 0.35 }[inputs.fxPressure];
  const tenorAdjustment      = { 91: -0.1, 182: 0.05, 364: 0.22 }[tenor];

  const raw =
    base +
    trend * 0.55 +
    demandAdjustment +
    policyAdjustment +
    inflationAdjustment +
    liquidityAdjustment +
    fxAdjustment +
    tenorAdjustment;

  return clamp(raw, 8, 32);
}

// Confidence is based on how far the forecast deviates from the baseline.
// Large adjustments mean the model is extrapolating further — lower confidence.
function confidenceLabel(prediction, base) {
  const deviation = Math.abs(prediction - base);
  if (deviation <= 0.20) return "High confidence";
  if (deviation <= 0.55) return "Medium confidence";
  return "Low to medium — large adjustment from baseline";
}

function drawChart(predictedCurve) {
  const canvas = els.chart;
  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;

  // Size the backing buffer to physical pixels so the chart is sharp on
  // high-DPI screens; all drawing coordinates stay in CSS pixels.
  const cssWidth = canvas.offsetWidth || 760;
  const cssHeight = canvas.offsetHeight || 320;
  canvas.width = cssWidth * dpr;
  canvas.height = cssHeight * dpr;
  ctx.scale(dpr, dpr);

  const width = cssWidth;
  const height = cssHeight;
  const pad = 42;
  const labels = ["91", "182", "364"];
  const latest = getLatest();
  const latestRates = labels.map((tenor) => latest.rates[tenor]);
  const predictedRates = labels.map((tenor) => predictedCurve[tenor]);
  const values = latestRates.concat(predictedRates);
  const min = Math.floor(Math.min(...values) - 1);
  const max = Math.ceil(Math.max(...values) + 1);

  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, width, height);

  ctx.strokeStyle = "#d8dee4";
  ctx.lineWidth = 1;
  ctx.fillStyle = "#62707d";
  ctx.font = "13px Segoe UI, Arial";

  for (let i = 0; i <= 4; i += 1) {
    const y = pad + ((height - pad * 2) / 4) * i;
    const value = max - ((max - min) / 4) * i;
    ctx.beginPath();
    ctx.moveTo(pad, y);
    ctx.lineTo(width - pad, y);
    ctx.stroke();
    ctx.fillText(`${value.toFixed(1)}%`, 8, y + 4);
  }

  function point(index, value) {
    const x = pad + ((width - pad * 2) / (labels.length - 1)) * index;
    const y = height - pad - ((value - min) / (max - min)) * (height - pad * 2);
    return { x, y };
  }

  function line(rates, color) {
    ctx.strokeStyle = color;
    ctx.lineWidth = 4;
    ctx.beginPath();
    rates.forEach((value, index) => {
      const p = point(index, value);
      if (index === 0) ctx.moveTo(p.x, p.y);
      else ctx.lineTo(p.x, p.y);
    });
    ctx.stroke();

    rates.forEach((value, index) => {
      const p = point(index, value);
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(p.x, p.y, 6, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = "#18212a";
      ctx.fillText(pct(value), p.x - 22, p.y - 14);
    });
  }

  line(latestRates, "#b87817");
  line(predictedRates, "#176b4d");

  labels.forEach((label, index) => {
    const p = point(index, min);
    ctx.fillStyle = "#18212a";
    ctx.fillText(`${label}d`, p.x - 14, height - 12);
  });

  ctx.fillStyle = "#b87817";
  ctx.fillText("Latest", width - 138, 24);
  ctx.fillStyle = "#176b4d";
  ctx.fillText("Predicted", width - 82, 24);
}

function renderTable() {
  els.table.innerHTML = seedAuctions
    .map(
      (row) => `
        <tr>
          <td>${row.date}</td>
          <td>${pct(row.rates[91])}</td>
          <td>${pct(row.rates[182])}</td>
          <td>${pct(row.rates[364])}</td>
          <td>${row.signal}</td>
        </tr>
      `
    )
    .join("");
}

function update() {
  const inputs = readInputs();
  const prediction = predictRate(inputs, inputs.tenor);
  const latestForTenor = getLatest().rates[inputs.tenor];
  const bidCover = inputs.subscription / inputs.offer;
  const predictedCurve = {
    91: predictRate(inputs, "91"),
    182: predictRate(inputs, "182"),
    364: predictRate(inputs, "364")
  };

  els.predictedRate.textContent = pct(prediction);
  els.latestRate.textContent = pct(latestForTenor);
  els.bidCover.textContent = `${bidCover.toFixed(2)}x`;
  els.confidence.textContent = confidenceLabel(prediction, latestForTenor);

  const move = prediction - latestForTenor;
  els.badge.className = `badge ${move > 0.15 ? "up" : move < -0.15 ? "down" : ""}`;
  els.badge.textContent =
    move > 0.15 ? "Upward pressure" : move < -0.15 ? "Downward pressure" : "Neutral";

  const demandPhrase =
    bidCover > 2.2
      ? "strong demand should help restrain the stop rate"
      : bidCover < 1.1
        ? "weak cover may require a higher stop rate"
        : "cover is balanced enough to keep the forecast close to recent pricing";
  const liquidityPhrase =
    inputs.liquidity === "flush"
      ? "flush liquidity pulls the forecast lower"
      : inputs.liquidity === "tight"
        ? "tight liquidity adds pressure"
        : "balanced liquidity keeps the model centered";

  els.reading.textContent = `For the ${inputs.tenor}-day bill, the model starts from the latest ${pct(
    latestForTenor
  )} stop rate, then adjusts for trend, bid-cover, policy-rate gap, inflation, liquidity, and FX pressure. With ${bidCover.toFixed(
    2
  )}x bid-cover, ${demandPhrase}; ${liquidityPhrase}.`;

  drawChart(predictedCurve);
}

function reset() {
  Object.entries(defaults).forEach(([key, value]) => {
    els[key].value = value;
  });
  update();
}

// Try to pull the latest market snapshot from the backend API.
// On success, appends a live entry to seedAuctions and pre-fills MPR/inflation.
// Silently falls back to hardcoded seed data if the API is unreachable.
async function tryRefreshFromApi() {
  try {
    const resp = await fetch("/snapshots/latest", {
      signal: AbortSignal.timeout(3000)
    });
    if (!resp.ok) return;

    const data = await resp.json();
    const tenors = [91, 182, 364];
    const rates = {};

    for (const t of tenors) {
      const snap = data[`${t}D`];
      if (!snap || snap.lag1_stop == null) return; // incomplete — skip
      rates[t] = snap.lag1_stop;
    }

    // Use macro from the 364D snapshot (all tenors share the same session values)
    const macro = data["364D"];
    const auctionDate = macro.auction_date ?? new Date().toISOString().split("T")[0];

    // Only append if this date isn't already in the seed list
    const alreadyPresent = seedAuctions.some((a) => a.date === auctionDate);
    if (!alreadyPresent) {
      seedAuctions.push({ date: auctionDate, rates, signal: "Live — API" });
    }

    // Pre-fill shared inputs from the live snapshot
    if (macro.mpr != null) els.mpr.value = macro.mpr;
    if (macro.inflation != null) els.inflation.value = macro.inflation;
    if (macro.system_liquidity != null) {
      // Map numeric liquidity to the select options: <0 = flush, >500 = tight, else balanced
      els.liquidity.value =
        macro.system_liquidity < 0 ? "tight" : macro.system_liquidity > 1500 ? "flush" : "balanced";
    }
  } catch {
    // API unavailable — seed data used as-is
  }
}

// Redraw on window resize so DPR-scaled canvas stays correct
window.addEventListener("resize", () => {
  const inputs = readInputs();
  const predictedCurve = {
    91: predictRate(inputs, "91"),
    182: predictRate(inputs, "182"),
    364: predictRate(inputs, "364")
  };
  drawChart(predictedCurve);
});

renderTable();
update();
els.form.addEventListener("input", update);
els.form.addEventListener("change", update);
els.reset.addEventListener("click", reset);

// Fire-and-forget: refresh from API then re-render if it succeeds
tryRefreshFromApi().then(() => {
  renderTable();
  update();
});
