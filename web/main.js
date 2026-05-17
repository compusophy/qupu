// QuPU app shell — boots the QPU, wires up the UI, drives the render loop.

import { QuPU, Gates, Presets } from './qupu.js';

const cfg = {
  nQubits: 6,
  nWorkersRequested: Math.min(navigator.hardwareConcurrency || 4, 8),
  stepDelayMs: 90,
};

const els = {
  die: document.getElementById('die'),
  state: document.getElementById('state'),
  circuit: document.getElementById('circuit'),
  ops: document.getElementById('ops'),
  presets: document.getElementById('presets'),
  toolbar: document.getElementById('toolbar'),
  stats: {
    qubits: document.getElementById('stat-qubits'),
    amps: document.getElementById('stat-amps'),
    cores: document.getElementById('stat-cores'),
    gates: document.getElementById('stat-gates'),
    wall: document.getElementById('stat-wall'),
    speedup: document.getElementById('stat-speedup'),
    norm: document.getElementById('stat-norm'),
  },
  targetSel: document.getElementById('target-sel'),
  ctrlSel: document.getElementById('ctrl-sel'),
  thetaInput: document.getElementById('theta-input'),
  qubitsSel: document.getElementById('qubits-sel'),
  runBtn: document.getElementById('run-btn'),
  stepBtn: document.getElementById('step-btn'),
  resetBtn: document.getElementById('reset-btn'),
  clearBtn: document.getElementById('clear-btn'),
  banner: document.getElementById('coherence-banner'),
};

let qpu = null;
let currentOps = [];
let executingIdx = -1;
let coreActivityTs = []; // last-active timestamp per core
let lastFrameTs = 0;

async function boot() {
  // SAB requires cross-origin isolation. Surface a friendly failure if not.
  if (typeof SharedArrayBuffer === 'undefined') {
    document.body.innerHTML = `
      <div style="padding:32px;font-family:monospace;color:#ff7777;background:#0a0e14;height:100vh">
      SharedArrayBuffer is unavailable. Serve this app over the included dev server
      (npm start) so the COOP/COEP isolation headers are applied.
      </div>`;
    return;
  }

  qpu = new QuPU(cfg.nQubits, cfg.nWorkersRequested);
  await qpu.init();
  coreActivityTs = new Array(qpu.nWorkers).fill(0);
  qpu.onCoreActivity = (coreId) => { coreActivityTs[coreId] = performance.now(); };

  await qpu.reset();

  buildDie();
  buildStateGrid();
  buildSelectors();
  bindToolbar();
  bindPresets();
  renderStats();
  requestAnimationFrame(frame);
}

function buildDie() {
  els.die.innerHTML = '';
  // arrange as roughly-square grid
  const k = qpu.nWorkers;
  const cols = Math.ceil(Math.sqrt(k));
  els.die.style.setProperty('--cols', cols);
  for (let i = 0; i < k; i++) {
    const cell = document.createElement('div');
    cell.className = 'core';
    cell.innerHTML = `
      <div class="core-id">C${i.toString(16).toUpperCase()}</div>
      <div class="core-meter"><div class="core-meter-fill" data-core="${i}"></div></div>
      <div class="core-label" data-core-label="${i}">idle</div>
    `;
    els.die.appendChild(cell);
  }
}

function buildStateGrid() {
  els.state.innerHTML = '';
  const n = qpu.nAmps;
  const cols = Math.ceil(Math.sqrt(n));
  els.state.style.setProperty('--cols', cols);
  for (let i = 0; i < n; i++) {
    const cell = document.createElement('div');
    cell.className = 'amp';
    cell.dataset.idx = i;
    cell.innerHTML = `
      <div class="amp-disc" data-disc="${i}"></div>
      <div class="amp-label">|${i.toString(2).padStart(qpu.nQubits, '0')}⟩</div>
    `;
    els.state.appendChild(cell);
  }
}

function buildSelectors() {
  for (const sel of [els.targetSel, els.ctrlSel]) {
    sel.innerHTML = '';
    for (let i = 0; i < qpu.nQubits; i++) {
      const opt = document.createElement('option');
      opt.value = i;
      opt.textContent = `q${i}`;
      sel.appendChild(opt);
    }
  }
  els.ctrlSel.value = 0;
  els.targetSel.value = 1;

  els.qubitsSel.innerHTML = '';
  for (let n = 2; n <= 10; n++) {
    const opt = document.createElement('option');
    opt.value = n;
    opt.textContent = `${n} qubits (${1 << n} amps)`;
    if (n === cfg.nQubits) opt.selected = true;
    els.qubitsSel.appendChild(opt);
  }
  els.qubitsSel.onchange = async (e) => {
    cfg.nQubits = parseInt(e.target.value, 10);
    await reboot();
  };
}

async function reboot() {
  for (const w of qpu.workers) w.terminate();
  qpu = new QuPU(cfg.nQubits, cfg.nWorkersRequested);
  await qpu.init();
  coreActivityTs = new Array(qpu.nWorkers).fill(0);
  qpu.onCoreActivity = (coreId) => { coreActivityTs[coreId] = performance.now(); };
  currentOps = [];
  executingIdx = -1;
  await qpu.reset();
  buildDie();
  buildStateGrid();
  buildSelectors();
  renderCircuit();
  renderStats();
}

function bindToolbar() {
  els.toolbar.querySelectorAll('button[data-gate]').forEach((btn) => {
    btn.onclick = () => {
      const g = btn.dataset.gate;
      const target = parseInt(els.targetSel.value, 10);
      const ctrl = parseInt(els.ctrlSel.value, 10);
      const theta = parseFloat(els.thetaInput.value);
      const op = { gate: g };
      if (g === 'CNOT' || g === 'CZ') {
        if (ctrl === target) { flashBanner('ctrl must differ from target', 'warn'); return; }
        op.ctrl = ctrl; op.target = target;
      } else if (g === 'SWAP') {
        if (ctrl === target) { flashBanner('swap requires distinct qubits', 'warn'); return; }
        op.a = ctrl; op.b = target;
      } else if (g === 'RX' || g === 'RY' || g === 'RZ') {
        op.target = target; op.theta = theta;
      } else {
        op.target = target;
      }
      currentOps.push(op);
      renderCircuit();
    };
  });

  els.runBtn.onclick = runAll;
  els.stepBtn.onclick = stepOne;
  els.resetBtn.onclick = resetState;
  els.clearBtn.onclick = () => {
    currentOps = [];
    executingIdx = -1;
    renderCircuit();
  };
}

function bindPresets() {
  els.presets.querySelectorAll('button[data-preset]').forEach((btn) => {
    btn.onclick = () => {
      const name = btn.dataset.preset;
      const fn = Presets[name];
      if (!fn) return;
      currentOps = fn(qpu.nQubits);
      executingIdx = -1;
      renderCircuit();
    };
  });
}

async function resetState() {
  await qpu.reset();
  qpu.telemetry.gatesApplied = 0;
  qpu.telemetry.totalWallMs = 0;
  qpu.telemetry.perCoreMs = new Array(qpu.nWorkers).fill(0);
  executingIdx = -1;
  renderStats();
}

async function runAll() {
  await resetState();
  for (let i = 0; i < currentOps.length; i++) {
    executingIdx = i;
    renderCircuit();
    await qpu._runOp(currentOps[i]);
    renderStats();
    await sleep(cfg.stepDelayMs);
  }
  executingIdx = -1;
  renderCircuit();
  checkNorm();
}

async function stepOne() {
  if (executingIdx + 1 >= currentOps.length) return;
  executingIdx += 1;
  renderCircuit();
  await qpu._runOp(currentOps[executingIdx]);
  renderStats();
}

async function checkNorm() {
  // Local computation — fast and authoritative.
  let s = 0;
  for (let i = 0; i < qpu.nAmps; i++) {
    const r = qpu.state[2*i]; const im = qpu.state[2*i+1];
    s += r*r + im*im;
  }
  els.stats.norm.textContent = s.toFixed(8);
  if (Math.abs(s - 1) > 1e-6) flashBanner('Norm drift detected', 'warn');
}

function flashBanner(text, kind) {
  els.banner.textContent = `// ${text}`;
  els.banner.dataset.kind = kind || 'info';
  clearTimeout(flashBanner._t);
  flashBanner._t = setTimeout(() => {
    els.banner.textContent = '// coherent';
    els.banner.dataset.kind = 'ok';
  }, 1800);
}

function sleep(ms) { return new Promise((r) => setTimeout(r, ms)); }

// --- Circuit SVG ---

function renderCircuit() {
  const svgNS = 'http://www.w3.org/2000/svg';
  const laneH = 34;
  const colW = 38;
  const padX = 64;
  const padY = 16;
  const w = Math.max(padX + colW * (currentOps.length + 1), 480);
  const h = padY * 2 + laneH * qpu.nQubits;
  els.circuit.setAttribute('width', w);
  els.circuit.setAttribute('height', h);
  els.circuit.setAttribute('viewBox', `0 0 ${w} ${h}`);
  els.circuit.innerHTML = '';

  // Qubit lanes
  for (let i = 0; i < qpu.nQubits; i++) {
    const y = padY + i * laneH + laneH / 2;
    const label = document.createElementNS(svgNS, 'text');
    label.setAttribute('x', 12);
    label.setAttribute('y', y + 4);
    label.setAttribute('class', 'lane-label');
    label.textContent = `q${i}`;
    els.circuit.appendChild(label);
    const line = document.createElementNS(svgNS, 'line');
    line.setAttribute('x1', 40); line.setAttribute('y1', y);
    line.setAttribute('x2', w - 12); line.setAttribute('y2', y);
    line.setAttribute('class', 'lane-wire');
    els.circuit.appendChild(line);
  }

  // Gate columns
  currentOps.forEach((op, idx) => {
    const x = padX + colW * idx + colW / 2;
    const targetQ = op.target ?? op.b ?? 0;
    const yT = padY + targetQ * laneH + laneH / 2;
    const active = idx === executingIdx;
    const cls = active ? 'gate active' : 'gate';
    if (op.gate === 'CNOT' || op.gate === 'CX' || op.gate === 'CZ') {
      const yC = padY + op.ctrl * laneH + laneH / 2;
      const link = document.createElementNS(svgNS, 'line');
      link.setAttribute('x1', x); link.setAttribute('y1', yC);
      link.setAttribute('x2', x); link.setAttribute('y2', yT);
      link.setAttribute('class', active ? 'gate-link active' : 'gate-link');
      els.circuit.appendChild(link);
      const dot = document.createElementNS(svgNS, 'circle');
      dot.setAttribute('cx', x); dot.setAttribute('cy', yC);
      dot.setAttribute('r', 4.5); dot.setAttribute('class', active ? 'ctrl-dot active' : 'ctrl-dot');
      els.circuit.appendChild(dot);
      if (op.gate === 'CZ') {
        const z = document.createElementNS(svgNS, 'text');
        z.setAttribute('x', x); z.setAttribute('y', yT + 4);
        z.setAttribute('text-anchor', 'middle'); z.setAttribute('class', 'gate-text');
        z.textContent = 'Z';
        const tgt = document.createElementNS(svgNS, 'circle');
        tgt.setAttribute('cx', x); tgt.setAttribute('cy', yT);
        tgt.setAttribute('r', 11); tgt.setAttribute('class', cls);
        els.circuit.appendChild(tgt);
        els.circuit.appendChild(z);
      } else {
        // CNOT XOR symbol
        const tgt = document.createElementNS(svgNS, 'circle');
        tgt.setAttribute('cx', x); tgt.setAttribute('cy', yT);
        tgt.setAttribute('r', 11); tgt.setAttribute('class', cls);
        els.circuit.appendChild(tgt);
        const cross1 = document.createElementNS(svgNS, 'line');
        cross1.setAttribute('x1', x - 7); cross1.setAttribute('y1', yT);
        cross1.setAttribute('x2', x + 7); cross1.setAttribute('y2', yT);
        cross1.setAttribute('class', 'gate-cross');
        const cross2 = document.createElementNS(svgNS, 'line');
        cross2.setAttribute('x1', x); cross2.setAttribute('y1', yT - 7);
        cross2.setAttribute('x2', x); cross2.setAttribute('y2', yT + 7);
        cross2.setAttribute('class', 'gate-cross');
        els.circuit.appendChild(cross1);
        els.circuit.appendChild(cross2);
      }
    } else if (op.gate === 'SWAP') {
      const yA = padY + op.a * laneH + laneH / 2;
      const yB = padY + op.b * laneH + laneH / 2;
      const link = document.createElementNS(svgNS, 'line');
      link.setAttribute('x1', x); link.setAttribute('y1', yA);
      link.setAttribute('x2', x); link.setAttribute('y2', yB);
      link.setAttribute('class', active ? 'gate-link active' : 'gate-link');
      els.circuit.appendChild(link);
      for (const yy of [yA, yB]) {
        const x1 = document.createElementNS(svgNS, 'line');
        x1.setAttribute('x1', x - 6); x1.setAttribute('y1', yy - 6);
        x1.setAttribute('x2', x + 6); x1.setAttribute('y2', yy + 6);
        x1.setAttribute('class', 'gate-cross');
        const x2 = document.createElementNS(svgNS, 'line');
        x2.setAttribute('x1', x - 6); x2.setAttribute('y1', yy + 6);
        x2.setAttribute('x2', x + 6); x2.setAttribute('y2', yy - 6);
        x2.setAttribute('class', 'gate-cross');
        els.circuit.appendChild(x1);
        els.circuit.appendChild(x2);
      }
    } else {
      // single-qubit box
      const r = document.createElementNS(svgNS, 'rect');
      r.setAttribute('x', x - 12); r.setAttribute('y', yT - 12);
      r.setAttribute('width', 24); r.setAttribute('height', 24);
      r.setAttribute('rx', 4);
      r.setAttribute('class', cls);
      els.circuit.appendChild(r);
      const t = document.createElementNS(svgNS, 'text');
      t.setAttribute('x', x); t.setAttribute('y', yT + 4);
      t.setAttribute('text-anchor', 'middle');
      t.setAttribute('class', 'gate-text');
      const label = (op.gate === 'RX' || op.gate === 'RY' || op.gate === 'RZ')
        ? op.gate : op.gate;
      t.textContent = label;
      els.circuit.appendChild(t);
      if (op.theta !== undefined) {
        const tt = document.createElementNS(svgNS, 'text');
        tt.setAttribute('x', x); tt.setAttribute('y', yT + 22);
        tt.setAttribute('text-anchor', 'middle');
        tt.setAttribute('class', 'gate-sub');
        tt.textContent = op.theta.toFixed(2);
        els.circuit.appendChild(tt);
      }
    }
  });

  // Op count text
  els.ops.textContent = `${currentOps.length} ops`;
}

// --- Render loop: state vector + die ---

function frame(ts) {
  // throttle to ~30Hz
  if (ts - lastFrameTs > 33) {
    lastFrameTs = ts;
    paintState();
    paintDie(ts);
  }
  requestAnimationFrame(frame);
}

function paintState() {
  const s = qpu.state;
  for (let i = 0; i < qpu.nAmps; i++) {
    const r = s[2*i], im = s[2*i+1];
    const mag = Math.sqrt(r*r + im*im);
    const phase = Math.atan2(im, r); // -π..π
    const hue = ((phase + Math.PI) / (2 * Math.PI)) * 360;
    const disc = document.querySelector(`[data-disc="${i}"]`);
    if (!disc) continue;
    const size = Math.max(2, Math.min(28, mag * 28));
    disc.style.width = `${size}px`;
    disc.style.height = `${size}px`;
    disc.style.background = `hsl(${hue}, 95%, ${30 + mag * 30}%)`;
    disc.style.boxShadow = `0 0 ${mag * 12}px hsl(${hue}, 95%, 60%)`;
    disc.style.opacity = `${0.25 + mag * 0.75}`;
  }
}

function paintDie(ts) {
  for (let i = 0; i < qpu.nWorkers; i++) {
    const last = coreActivityTs[i];
    const sinceMs = ts - last;
    const intensity = last === 0 ? 0 : Math.max(0, 1 - sinceMs / 400);
    const fill = document.querySelector(`[data-core="${i}"]`);
    const label = document.querySelector(`[data-core-label="${i}"]`);
    if (fill) {
      fill.style.transform = `scaleX(${0.05 + intensity * 0.95})`;
      fill.style.background = `linear-gradient(90deg, rgba(0,255,209,${0.3 + intensity*0.6}), rgba(255,45,173,${0.2 + intensity*0.5}))`;
    }
    if (label) {
      label.textContent = intensity > 0.05 ? 'computing' : 'idle';
      label.style.color = intensity > 0.05 ? '#00ffd1' : '#42526b';
    }
  }
}

function renderStats() {
  if (!qpu) return;
  els.stats.qubits.textContent = qpu.nQubits;
  els.stats.amps.textContent = qpu.nAmps;
  els.stats.cores.textContent = qpu.nWorkers;
  els.stats.gates.textContent = qpu.telemetry.gatesApplied;
  els.stats.wall.textContent = qpu.telemetry.totalWallMs.toFixed(2) + ' ms';
  const totalCoreMs = qpu.telemetry.perCoreMs.reduce((a, b) => a + b, 0);
  const speedup = qpu.telemetry.totalWallMs > 0
    ? (totalCoreMs / qpu.telemetry.totalWallMs).toFixed(2) + '×'
    : '—';
  els.stats.speedup.textContent = speedup;
}

boot();
