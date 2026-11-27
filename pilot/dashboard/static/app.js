const state = {
  modules: [],
  current: null,
};

const els = {
  list: document.getElementById("moduleList"),
  title: document.getElementById("title"),
  subtitle: document.getElementById("subtitle"),
  health: document.getElementById("healthChip"),
  overview: document.getElementById("overview"),
  workGrid: document.getElementById("workGrid"),
  workbench: document.getElementById("workbench"),
  stackBody: document.querySelector("#stackTable tbody"),
  formGrid: document.getElementById("formGrid"),
  form: document.getElementById("inferForm"),
  sampleBtn: document.getElementById("sampleBtn"),
  imageRow: document.getElementById("imageRow"),
  leafImage: document.getElementById("leafImage"),
  taskPill: document.getElementById("taskPill"),
  formError: document.getElementById("formError"),
  resultEmpty: document.getElementById("resultEmpty"),
  resultView: document.getElementById("resultView"),
  latency: document.getElementById("latencyChip"),
  metricsRow: document.getElementById("metricsRow"),
  schemaNote: document.getElementById("schemaNote"),
};

function fmt(n) {
  if (n === null || n === undefined) return "—";
  if (typeof n === "number") return Number.isInteger(n) ? String(n) : n.toFixed(4);
  return String(n);
}

function pickMetrics(mod) {
  const m = mod.metrics || {};
  const hold = (mod.schema && mod.schema.metrics_holdout) || m.metrics_holdout || m;
  const rows = [];
  if (hold.macro_f1 != null) rows.push(["Macro F1", hold.macro_f1]);
  if (hold.accuracy != null) rows.push(["Accuracy", hold.accuracy]);
  if (hold.mae != null) rows.push(["MAE", hold.mae]);
  if (hold.persistence && hold.persistence.mae != null) rows.push(["Persistence MAE", hold.persistence.mae]);
  if (hold.ridge && hold.ridge.mae != null) rows.push(["Ridge MAE", hold.ridge.mae]);
  if (m.bajra_mae != null) rows.push(["Bajra MAE", m.bajra_mae]);
  if (m.macro_f1 != null && !rows.find((r) => r[0] === "Macro F1")) rows.push(["Macro F1", m.macro_f1]);
  if (m.mae != null && !rows.find((r) => r[0] === "MAE")) rows.push(["MAE", m.mae]);
  return rows.slice(0, 3);
}

window.openModule = function openModule(id) {
  if (id === "overview") selectOverview();
  else selectModule(id);
};

function renderNav() {
  els.list.innerHTML = "";
  const overviewLi = document.createElement("li");
  overviewLi.innerHTML = `<button type="button" class="module-btn" data-id="overview" aria-current="${state.current ? "false" : "true"}" onclick="openModule('overview')">
    <span class="name">Stack overview</span>
    <span class="meta"><span class="pill">${state.modules.length} modules</span></span>
  </button>`;
  els.list.appendChild(overviewLi);

  for (const mod of state.modules) {
    const li = document.createElement("li");
    const current = state.current === mod.id;
    li.innerHTML = `<button type="button" class="module-btn" data-id="${mod.id}" aria-current="${current ? "true" : "false"}" onclick="openModule('${mod.id}')">
      <span class="name">${mod.short}</span>
      <span class="meta">
        <span class="pill ${mod.ready ? "ready" : "pending"}">${mod.ready ? "ready" : "pending"}</span>
        <span>${mod.original}</span>
      </span>
    </button>`;
    els.list.appendChild(li);
  }
}

function renderOverview() {
  els.stackBody.innerHTML = state.modules
    .map(
      (m) => `<tr>
      <td>${m.short}</td>
      <td>${m.original}</td>
      <td>${m.task}</td>
      <td>${m.version || "—"}</td>
      <td><span class="pill ${m.ready ? "ready" : "pending"}">${m.ready ? "ready" : "pending"}</span></td>
    </tr>`
    )
    .join("");
}

function selectOverview() {
  state.current = null;
  els.overview.classList.add("active");
  els.overview.hidden = false;
  els.workGrid.hidden = true;
  els.title.textContent = "Pilot stack";
  els.subtitle.textContent = "Run each shipped module with the same schema the Android path will use.";
  renderNav();
}

function selectModule(id) {
  const mod = state.modules.find((m) => m.id === id);
  if (!mod) return;
  state.current = id;
  els.overview.classList.remove("active");
  els.overview.hidden = true;
  els.workGrid.hidden = false;
  els.title.textContent = mod.short;
  els.subtitle.textContent = `${mod.id} · ${mod.original} · ${mod.task}`;
  els.taskPill.textContent = mod.task;
  buildForm(mod);
  renderMetrics(mod);
  clearResult();
  renderNav();
}

function buildForm(mod) {
  els.formGrid.innerHTML = "";
  els.formError.hidden = true;
  const isDisease = mod.id === "module_06_pearl_millet_disease";
  els.imageRow.hidden = !isDisease;

  const order = mod.feature_order || [];
  if (!order.length && isDisease) {
    els.formGrid.innerHTML = `<div class="field span-2"><p style="margin:0;color:var(--muted)">Upload a leaf image. Colour features are extracted with the pilot feature code before ONNX runs.</p></div>`;
    return;
  }

  for (const name of order) {
    const wrap = document.createElement("div");
    wrap.className = "field";
    const isCat = ["Soil Type", "Crop Type", "Crop", "Season", "State"].includes(name);
    if (isCat) {
      wrap.innerHTML = `<label for="f_${name}">${name}</label><input id="f_${name}" name="${name}" type="text" autocomplete="off" />`;
    } else {
      wrap.innerHTML = `<label for="f_${name}">${name}</label><input id="f_${name}" name="${name}" type="number" step="any" />`;
    }
    els.formGrid.appendChild(wrap);
  }
  fillSample(mod);
}

function fillSample(mod) {
  const sample = mod.sample || {};
  for (const [k, v] of Object.entries(sample)) {
    const input = document.getElementById(`f_${k}`);
    if (input) input.value = v;
  }
}

function readFeatures(mod) {
  const features = {};
  for (const name of mod.feature_order || []) {
    const input = document.getElementById(`f_${name}`);
    if (!input) continue;
    const raw = input.value;
    if (raw === "") throw new Error(`Missing ${name}`);
    const numeric = !["Soil Type", "Crop Type", "Crop", "Season", "State"].includes(name);
    features[name] = numeric ? Number(raw) : raw;
  }
  return features;
}

function renderMetrics(mod) {
  const rows = pickMetrics(mod);
  els.metricsRow.innerHTML = rows.length
    ? rows.map(([k, v]) => `<div class="metric"><div class="k">${k}</div><div class="v">${fmt(v)}</div></div>`).join("")
    : `<div class="metric"><div class="k">Status</div><div class="v">${mod.ready ? "Artifacts present" : "Pending build"}</div></div>`;

  const honesty = [];
  if (mod.schema?.recommended_policy) honesty.push(`policy: ${mod.schema.recommended_policy}`);
  if (mod.schema?.train?.sources) honesty.push(`sources: ${(mod.schema.train.sources || []).join(", ")}`);
  if (mod.schema?.honesty) honesty.push(mod.schema.honesty);
  if (mod.id === "module_03_crop_recommendation") honesty.push("crop table has no pearl millet class");
  if (mod.id === "module_06_pearl_millet_disease") honesty.push("partial classes; colour features; optimistic image split");
  if (mod.id === "module_05_crop_yield") honesty.push("coconut dominates all-crop MAE; check Bajra slice");
  if (mod.id === "module_04_fertilizer_recommendation") honesty.push("99 rows after dedupe; CV macro F1 is the trust metric");
  els.schemaNote.textContent = honesty.join("\n") || "schema loaded";
}

function clearResult() {
  els.resultEmpty.hidden = false;
  els.resultView.hidden = true;
  els.resultView.innerHTML = "";
  els.latency.textContent = "";
}

function showResult(result, ms) {
  els.resultEmpty.hidden = true;
  els.resultView.hidden = false;
  els.latency.textContent = `${ms.toFixed(0)} ms`;

  const label =
    result.fertility ||
    result.crop ||
    result.fertilizer ||
    result.disease ||
    (result.yield_pred != null ? `${fmt(result.yield_pred)}` : null) ||
    (result.recommended != null ? fmt(result.recommended) : null) ||
    (result.prediction != null ? fmt(result.prediction) : null);

  const conf = result.confidence != null ? `confidence ${fmt(result.confidence)}` : "";
  const version = result.model_version ? `v${result.model_version}` : "";

  let probs = "";
  if (result.probabilities && typeof result.probabilities === "object") {
    const entries = Object.entries(result.probabilities).sort((a, b) => b[1] - a[1]).slice(0, 6);
    probs = `<dl class="kv">${entries.map(([k, v]) => `<dt>${k}</dt><dd>${fmt(v)}</dd>`).join("")}</dl>`;
  }

  els.resultView.innerHTML = `
    <div class="answer">
      <div class="label">Prediction</div>
      <div class="value">${label ?? "see raw"}</div>
      <div class="sub">${[conf, version].filter(Boolean).join(" · ")}</div>
    </div>
    ${probs}
    <pre class="note">${JSON.stringify(result, null, 2)}</pre>
  `;
}

els.sampleBtn.addEventListener("click", () => {
  const mod = state.modules.find((m) => m.id === state.current);
  if (mod) fillSample(mod);
});

els.form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const mod = state.modules.find((m) => m.id === state.current);
  if (!mod) return;
  els.formError.hidden = true;
  if (!mod.ready) {
    els.formError.textContent = "This module is not ready yet (missing artifacts).";
    els.formError.hidden = false;
    return;
  }

  const t0 = performance.now();
  try {
    let data;
    if (mod.id === "module_06_pearl_millet_disease" && els.leafImage.files[0]) {
      const fd = new FormData();
      fd.append("file", els.leafImage.files[0]);
      const res = await fetch(`/api/infer/${mod.id}/image`, { method: "POST", body: fd });
      data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Image inference failed");
    } else {
      const features = readFeatures(mod);
      const res = await fetch(`/api/infer/${mod.id}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ features }),
      });
      data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Inference failed");
    }
    showResult(data.result, performance.now() - t0);
  } catch (err) {
    els.formError.textContent = err.message || String(err);
    els.formError.hidden = false;
    clearResult();
  }
});

async function boot() {
  const [mods, health] = await Promise.all([
    fetch("/api/modules").then((r) => r.json()),
    fetch("/api/health").then((r) => r.json()),
  ]);
  state.modules = mods;
  els.health.textContent = `${health.ready.length} ready · ${health.pending.length} pending`;
  renderNav();
  renderOverview();
  selectOverview();
}

boot().catch((err) => {
  els.health.textContent = "API offline";
  els.formError.hidden = false;
  els.formError.textContent = String(err);
});
