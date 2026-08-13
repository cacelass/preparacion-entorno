/* app.js — motor compartido de la demo {{ project_name }}.
   Carga meta.json y ejecuta el modelo ONNX con onnxruntime-web en el navegador.
   Sin servidor ni build tooling: los .onnx y meta.json viajan en el repo. */
const META_URL = 'models/meta.json';
const BASE = '.';

let ort = null;      // onnxruntime-web
let meta = null;     // meta.json
let sess = null;     // InferenceSession activo
let activeModel = null;

// ── Utilidades ───────────────────────────────────────────
async function fetchJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`No se pudo cargar ${url} (HTTP ${r.status})`);
  return r.json();
}

function el(id) { return document.getElementById(id); }

function setStatus(msg) {
  const s = el('status');
  if (s) s.textContent = msg;
}

function setLoading(msg) {
  const l = el('loading');
  if (l) { l.textContent = msg; l.style.display = msg ? 'block' : 'none'; }
}

async function loadRuntime() {
  if (ort) return;
  setStatus('Cargando onnxruntime-web…');
  // Cargado desde CDN en las páginas; aquí solo confirmamos que existe.
  if (window.ort) { ort = window.ort; return; }
  await new Promise((resolve, reject) => {
    const s = document.createElement('script');
    s.src = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/ort.min.js';
    s.onload = () => { ort = window.ort; resolve(); };
    s.onerror = () => reject(new Error('No se pudo cargar onnxruntime-web desde CDN'));
    document.head.appendChild(s);
  });
  setStatus('');
}

async function initMeta() {
  if (meta) return meta;
  meta = await fetchJSON(`${BASE}/${META_URL}`);
  return meta;
}

// ── Carga del modelo ─────────────────────────────────────
async function loadModel(name) {
  if (sess && activeModel === name) return sess;
  await loadRuntime();
  setLoading(`Cargando modelo ${name}…`);
  const url = `${BASE}/models/${name}.onnx`;
  const arrayBuffer = await (await fetch(url)).arrayBuffer();
  sess = await ort.InferenceSession.create(arrayBuffer, { executionProviders: ['wasm'] });
  activeModel = name;
  setLoading('');
  setStatus(`Modelo ${name} listo`);
  return sess;
}

function modelInfo(name) {
  return meta.models.find(m => m.name === name);
}

// ── Form dinámico desde meta.json ────────────────────────
function featureInputId(idx) { return `feature-${idx}`; }

function renderForm() {
  const form = el('feature-form');
  form.innerHTML = '';
  meta.features.forEach((f, i) => {
    const div = document.createElement('div');
    div.className = 'field';
    const label = document.createElement('label');
    label.htmlFor = featureInputId(i);
    label.textContent = f.name;
    if (f.type === 'categorical') {
      const span = document.createElement('span');
      span.className = 'cat';
      span.textContent = ' · categórica';
      label.appendChild(span);
    }
    div.appendChild(label);
    if (f.type === 'categorical') {
      const sel = document.createElement('select');
      sel.id = featureInputId(i);
      f.classes.forEach(c => {
        const opt = document.createElement('option');
        opt.value = c;
        opt.textContent = c;
        sel.appendChild(opt);
      });
      div.appendChild(sel);
    } else {
      const inp = document.createElement('input');
      inp.type = 'number';
      inp.step = 'any';
      inp.id = featureInputId(i);
      inp.placeholder = f.ref !== undefined ? String(f.ref) : '0.0';
      inp.value = f.ref !== undefined ? String(f.ref) : '';
      div.appendChild(inp);
    }
    form.appendChild(div);
  });
  const actions = document.createElement('div');
  actions.className = 'form-actions';
  const rnd = document.createElement('button');
  rnd.id = 'btn-random';
  rnd.type = 'button';
  rnd.className = 'btn';
  rnd.textContent = 'Rellenar ejemplo';
  rnd.onclick = fillRandom;
  actions.appendChild(rnd);
  const btn = document.createElement('button');
  btn.id = 'btn-predict';
  btn.type = 'submit';
  btn.className = 'btn primary';
  btn.textContent = 'Predecir';
  actions.appendChild(btn);
  form.appendChild(actions);
  form.onsubmit = (e) => { e.preventDefault(); predict(); };
}

function fillRandom() {
  meta.features.forEach((f, i) => {
    const inp = el(featureInputId(i));
    if (!inp) return;
    if (f.type === 'categorical') {
      inp.value = f.classes[Math.floor(Math.random() * f.classes.length)];
    } else if (f.min !== undefined) {
      const v = f.min + Math.random() * (f.max - f.min);
      inp.value = v.toFixed(4);
    }
  });
}

// ── Inferencia ───────────────────────────────────────────
async function predict() {
  const out = el('result');
  out.innerHTML = '';
  if (!sess) { out.innerHTML = '<div class="error">Carga un modelo primero.</div>'; return; }
  try {
    const input = buildInput();
    const feeds = { float_input: input };
    const results = await sess.run(feeds);
    renderResult(results, input);
  } catch (err) {
    out.innerHTML = `<div class="error">Error de inferencia: ${err.message}</div>`;
  }
}

function buildInput() {
  // Categóricas → índice dentro de sus clases (replica LabelEncoder del template).
  const values = [];
  meta.features.forEach((f, i) => {
    const raw = el(featureInputId(i)).value.trim();
    if (f.type === 'categorical') {
      const idx = f.classes.indexOf(raw);
      values.push(idx === -1 ? 0 : idx);
    } else {
      const n = parseFloat(raw);
      values.push(isNaN(n) ? 0.0 : n);
    }
  });
  return new Float32Array(values);  // shape (n_features,) → batch 1
}

function renderResult(results, input) {
  const out = el('result');
  const model = modelInfo(activeModel);
  // Orden de salidas: onnxruntime devuelve [label?, proba?] según kind.
  const keys = Object.keys(results);
  const first = keys[0];
  const data = results[first];

  let html = `<div class="pred">`;

  if (model.kind === 'classification') {
    if (keys.length >= 2 && model.classes && results[keys[0]].data) {
      // zipmap=False → [output_label, output_probability]
      const labelIdx = Number(results[keys[0]].data[0]);
      const proba = Array.from(results[keys[1]].data);
      const cls = model.classes[labelIdx];
      const pct = (proba[labelIdx] * 100).toFixed(1);
      html += `${escapeHtml(cls)} <span style="font-size:16px;color:var(--fg-dim)">(${pct}%)</span>`;
      html += `<div class="sub">Probabilidades por clase:</div>`;
      html += `<div class="sub">${model.classes.map((c, i) =>
        `${escapeHtml(c)}: ${(proba[i]*100).toFixed(1)}%`).join(' · ')}</div>`;
      html += `<div class="prob-bar"><div class="fill" style="width:${pct}%"></div></div>`;
    } else {
      // NN → logits: argmax + softmax en cliente
      const logits = Array.from(data);
      const maxIdx = logits.indexOf(Math.max(...logits));
      const probs = softmax(logits);
      const pct = (probs[maxIdx] * 100).toFixed(1);
      const cls = model.classes ? model.classes[maxIdx] : maxIdx;
      html += `${escapeHtml(String(cls))} <span style="font-size:16px;color:var(--fg-dim)">(${pct}%)</span>`;
      html += `<div class="sub">${(model.classes || logits.map((_, i) => i)).map((c, i) =>
        `${escapeHtml(String(c))}: ${(probs[i]*100).toFixed(1)}%`).join(' · ')}</div>`;
      html += `<div class="prob-bar"><div class="fill" style="width:${pct}%"></div></div>`;
    }
  } else if (model.kind === 'regression') {
    html += `${Number(data[0]).toFixed(4)}`;
    html += `<div class="sub">Salida continua del modelo ${escapeHtml(activeModel)}</div>`;
  } else {
    // clustering
    html += `Cluster ${Number(data[0])}`;
    html += `<div class="sub">Grupo asignado por ${escapeHtml(activeModel)}</div>`;
  }
  html += `</div>`;
  html += `<div class="sub" style="margin-top:8px">Modelo: ${escapeHtml(activeModel)}</div>`;
  out.innerHTML = html;
}

function softmax(logits) {
  const max = Math.max(...logits);
  const exps = logits.map(x => Math.exp(x - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map(x => x / sum);
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g,
    c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

// ── Selector de modelo ───────────────────────────────────
function populateModelSelect() {
  const sel = el('model-select');
  meta.models.forEach(m => {
    const opt = document.createElement('option');
    opt.value = m.name;
    opt.textContent = m.name;
    sel.appendChild(opt);
  });
  if (sel.options.length) {
    sel.value = meta.models[0].name;
    sel.onchange = () => loadModel(sel.value).then(() => { activeModel = sel.value; });
  }
}

async function initTry() {
  try {
    await initMeta();
    populateModelSelect();
    renderForm();
    await loadModel(el('model-select').value);
  } catch (err) {
    setStatus(err.message);
  }
}

// Auto-arranque: si hay form de features, esta es la página try.html.
async function initTryPage() {
  if (!el('feature-form')) return;
  const out = el('result');
  try {
    await initTry();
    if (out) {
      out.style.display = 'block';
      out.innerHTML = '<div class="sub">Modelo cargado. Usá «Rellenar ejemplo» o completá las features y pulsá «Predecir».</div>';
    }
  } catch (err) {
    if (out) {
      out.style.display = 'block';
      out.innerHTML = `<div class="error">${escapeHtml(err.message)}</div>`;
    }
  }
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initTryPage);
} else {
  initTryPage();
}
