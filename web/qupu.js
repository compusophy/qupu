// QuPU orchestrator — runs on the main thread.
// Owns the shared state vector (SharedArrayBuffer) and a pool of compute
// workers. Each gate is partitioned into tiles aligned by the target qubit's
// stride; tiles are dispatched in parallel across workers, then a barrier
// waits for completion before the next gate runs.
//
// This mirrors the GPU dispatch model — kernel + workgroup decomposition —
// but with quantum amplitude pairs as the parallel unit of work.

const ROOT2_INV = 1 / Math.SQRT2;

// Gate matrices, row-major flat [a_re, a_im, b_re, b_im, c_re, c_im, d_re, d_im].
export const Gates = {
  H:  () => new Float64Array([ROOT2_INV, 0,  ROOT2_INV, 0,  ROOT2_INV, 0, -ROOT2_INV, 0]),
  X:  () => new Float64Array([0, 0,  1, 0,  1, 0,  0, 0]),
  Y:  () => new Float64Array([0, 0,  0, -1, 0, 1,  0, 0]),
  Z:  () => new Float64Array([1, 0,  0, 0,  0, 0, -1, 0]),
  S:  () => new Float64Array([1, 0,  0, 0,  0, 0,  0, 1]),
  T:  () => new Float64Array([1, 0,  0, 0,  0, 0,  Math.cos(Math.PI/4), Math.sin(Math.PI/4)]),
  RX: (t) => {
    const c = Math.cos(t/2), s = Math.sin(t/2);
    return new Float64Array([c, 0,  0, -s,  0, -s,  c, 0]);
  },
  RY: (t) => {
    const c = Math.cos(t/2), s = Math.sin(t/2);
    return new Float64Array([c, 0,  -s, 0,  s, 0,  c, 0]);
  },
  RZ: (t) => {
    const ch = Math.cos(t/2), sh = Math.sin(t/2);
    return new Float64Array([ch, -sh,  0, 0,  0, 0,  ch, sh]);
  },
};

function highestPow2AtMost(n) {
  let p = 1;
  while (p * 2 <= n) p *= 2;
  return p;
}

export class QuPU {
  constructor(nQubits, nWorkers) {
    this.nQubits = nQubits;
    this.nAmps = 1 << nQubits;
    // Workers must be a power of 2 ≤ nAmps. We clamp to nAmps to avoid idle.
    this.nWorkers = Math.min(highestPow2AtMost(nWorkers), this.nAmps);
    this.sab = new SharedArrayBuffer(this.nAmps * 2 * 8);
    this.state = new Float64Array(this.sab);
    this.workers = [];
    this.nextJobId = 1;
    this.pending = new Map(); // jobId -> {resolve, perCore[]}
    this.telemetry = { gatesApplied: 0, totalWallMs: 0, perCoreMs: [] };
    this.onCoreActivity = null; // (coreId, ms, pairs) callback for viz
  }

  async init() {
    for (let i = 0; i < this.nWorkers; i++) {
      const w = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
      w.onmessage = (e) => this._onWorkerMessage(e.data);
      this.workers.push(w);
      this.telemetry.perCoreMs.push(0);
    }
    await Promise.all(this.workers.map((w, i) => new Promise((res) => {
      const handler = (e) => {
        if (e.data.type === 'ready' && e.data.coreId === i) {
          w.removeEventListener('message', handler);
          // re-attach the main handler
          w.onmessage = (ev) => this._onWorkerMessage(ev.data);
          res();
        }
      };
      w.addEventListener('message', handler);
      w.postMessage({ type: 'init', coreId: i, nQubits: this.nQubits, sab: this.sab });
    })));
  }

  _onWorkerMessage(msg) {
    if (msg.type === 'done') {
      const entry = this.pending.get(msg.jobId);
      if (!entry) return;
      entry.perCore.push({ coreId: msg.coreId, ms: msg.ms, pairs: msg.pairs });
      this.telemetry.perCoreMs[msg.coreId] += msg.ms;
      if (this.onCoreActivity) this.onCoreActivity(msg.coreId, msg.ms, msg.pairs);
      if (entry.perCore.length === entry.expected) {
        this.pending.delete(msg.jobId);
        entry.resolve(entry.perCore);
      }
    }
  }

  _chooseTiles(target) {
    // alignAmps = 2^(target+1). maxTiles is how many independent blocks exist.
    const alignAmps = 1 << (target + 1);
    const maxTiles = this.nAmps / alignAmps; // power of 2
    const n = Math.min(this.nWorkers, maxTiles);
    return { nTiles: n, tileAmps: this.nAmps / n };
  }

  _dispatch(buildJob, nTiles, tileAmps) {
    const jobId = this.nextJobId++;
    const promise = new Promise((resolve) => {
      this.pending.set(jobId, { resolve, perCore: [], expected: nTiles });
    });
    const t0 = performance.now();
    for (let t = 0; t < nTiles; t++) {
      const baseIndex = t * tileAmps;
      const job = buildJob(t, baseIndex, tileAmps);
      job.jobId = jobId;
      this.workers[t].postMessage(job);
    }
    return promise.then((perCore) => {
      this.telemetry.totalWallMs += performance.now() - t0;
      return perCore;
    });
  }

  async reset() {
    const tileAmps = this.nAmps / this.nWorkers;
    const perCore = await this._dispatch((t, baseIndex, nAmps) => ({
      type: 'init_zero', baseIndex, nAmps, containsZero: t === 0,
    }), this.nWorkers, tileAmps);
    return perCore;
  }

  async apply1q(target, matrix) {
    const { nTiles, tileAmps } = this._chooseTiles(target);
    const perCore = await this._dispatch((t, baseIndex, nAmps) => ({
      type: 'apply_1q', target, baseIndex, nAmps, matrix,
    }), nTiles, tileAmps);
    this.telemetry.gatesApplied++;
    return perCore;
  }

  async applyC1q(ctrl, target, matrix) {
    const { nTiles, tileAmps } = this._chooseTiles(target);
    const perCore = await this._dispatch((t, baseIndex, nAmps) => ({
      type: 'apply_c1q', ctrl, target, baseIndex, nAmps, matrix,
    }), nTiles, tileAmps);
    this.telemetry.gatesApplied++;
    return perCore;
  }

  // Convenience: CNOT = controlled-X
  async cnot(ctrl, target) { return this.applyC1q(ctrl, target, Gates.X()); }
  async cz(ctrl, target)   { return this.applyC1q(ctrl, target, Gates.Z()); }
  async swap(a, b) {
    // SWAP via 3 CNOTs
    await this.cnot(a, b);
    await this.cnot(b, a);
    await this.cnot(a, b);
  }

  probabilities() {
    const out = new Float64Array(this.nAmps);
    for (let i = 0; i < this.nAmps; i++) {
      const r = this.state[2*i];
      const im = this.state[2*i + 1];
      out[i] = r*r + im*im;
    }
    return out;
  }

  amplitudes() {
    // Returns live view; consumers should snapshot if needed.
    return this.state;
  }

  // Compile a circuit (list of ops) to a sequence of awaited dispatches.
  // op = {gate: 'H'|'X'|..., target, ctrl?, theta?}
  async runCircuit(ops) {
    for (const op of ops) {
      await this._runOp(op);
    }
  }

  async _runOp(op) {
    const g = op.gate;
    if (g === 'CNOT' || g === 'CX') return this.applyC1q(op.ctrl, op.target, Gates.X());
    if (g === 'CZ')  return this.applyC1q(op.ctrl, op.target, Gates.Z());
    if (g === 'SWAP') return this.swap(op.a, op.b);
    if (g === 'RX')  return this.apply1q(op.target, Gates.RX(op.theta));
    if (g === 'RY')  return this.apply1q(op.target, Gates.RY(op.theta));
    if (g === 'RZ')  return this.apply1q(op.target, Gates.RZ(op.theta));
    if (Gates[g])    return this.apply1q(op.target, Gates[g]());
    throw new Error(`Unknown gate: ${g}`);
  }
}

// Preset circuits keyed by name. Each is a function (nQubits) -> ops[].
export const Presets = {
  Bell: (n) => [
    { gate: 'H', target: 0 },
    { gate: 'CNOT', ctrl: 0, target: 1 },
  ],

  GHZ: (n) => {
    const ops = [{ gate: 'H', target: 0 }];
    for (let i = 1; i < n; i++) ops.push({ gate: 'CNOT', ctrl: 0, target: i });
    return ops;
  },

  // Quantum Fourier Transform on all n qubits (standard textbook form).
  QFT: (n) => {
    const ops = [];
    for (let i = 0; i < n; i++) {
      ops.push({ gate: 'H', target: i });
      for (let j = i + 1; j < n; j++) {
        const theta = Math.PI / (1 << (j - i));
        // Controlled phase: implemented as RZ(theta/2) sandwich; for clarity
        // use the identity CRZ(θ) = RZ_t(θ/2) · CNOT · RZ_t(-θ/2) · CNOT.
        ops.push({ gate: 'RZ', target: j, theta: theta / 2 });
        ops.push({ gate: 'CNOT', ctrl: i, target: j });
        ops.push({ gate: 'RZ', target: j, theta: -theta / 2 });
        ops.push({ gate: 'CNOT', ctrl: i, target: j });
      }
    }
    // Bit reversal via SWAPs
    for (let i = 0; i < Math.floor(n / 2); i++) {
      ops.push({ gate: 'SWAP', a: i, b: n - 1 - i });
    }
    return ops;
  },

  // Grover with 1 iteration over n=3..5 qubits, marking |11..1>.
  Grover: (n) => {
    const ops = [];
    // Uniform superposition
    for (let i = 0; i < n; i++) ops.push({ gate: 'H', target: i });
    // Oracle: phase-flip |11..1> via a multi-controlled Z, approximated for
    // small n with a Z-on-top and CNOT ladders. For n=2 it's just CZ.
    if (n === 2) {
      ops.push({ gate: 'CZ', ctrl: 0, target: 1 });
    } else if (n === 3) {
      // Toffoli-decomposed multi-CZ on |111>
      ops.push({ gate: 'H', target: 2 });
      ops.push({ gate: 'CNOT', ctrl: 1, target: 2 });
      ops.push({ gate: 'T', target: 2 });
      ops.push({ gate: 'CNOT', ctrl: 0, target: 2 });
      ops.push({ gate: 'T', target: 2 });
      ops.push({ gate: 'CNOT', ctrl: 1, target: 2 });
      ops.push({ gate: 'T', target: 2 });
      ops.push({ gate: 'CNOT', ctrl: 0, target: 2 });
      ops.push({ gate: 'H', target: 2 });
    } else {
      // For larger n we just apply nested CZs as a rough proxy.
      for (let i = 0; i < n - 1; i++) ops.push({ gate: 'CZ', ctrl: i, target: n - 1 });
    }
    // Diffuser: H^n, X^n, multi-CZ, X^n, H^n
    for (let i = 0; i < n; i++) ops.push({ gate: 'H', target: i });
    for (let i = 0; i < n; i++) ops.push({ gate: 'X', target: i });
    if (n === 2) {
      ops.push({ gate: 'CZ', ctrl: 0, target: 1 });
    } else {
      for (let i = 0; i < n - 1; i++) ops.push({ gate: 'CZ', ctrl: i, target: n - 1 });
    }
    for (let i = 0; i < n; i++) ops.push({ gate: 'X', target: i });
    for (let i = 0; i < n; i++) ops.push({ gate: 'H', target: i });
    return ops;
  },

  // Simple test pattern — random single-qubit + entangling layers.
  Scramble: (n) => {
    const ops = [];
    const onesie = ['H', 'X', 'Y', 'S', 'T'];
    for (let layer = 0; layer < 3; layer++) {
      for (let i = 0; i < n; i++) {
        ops.push({ gate: onesie[(i + layer) % onesie.length], target: i });
      }
      for (let i = 0; i < n - 1; i += 2) {
        ops.push({ gate: 'CNOT', ctrl: i, target: i + 1 });
      }
      for (let i = 1; i < n - 1; i += 2) {
        ops.push({ gate: 'CNOT', ctrl: i, target: i + 1 });
      }
    }
    return ops;
  },
};
