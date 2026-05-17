// QuPU compute worker — one of K "cores" on the simulated quantum die.
// Each worker holds its own WASM kernel instance, and operates on a tile of
// the global state vector (which lives in a SharedArrayBuffer shared with
// the main thread). Tiles are aligned by the orchestrator so workers never
// touch each other's amplitudes during a gate.

import init, { QKernel } from './pkg/qupu_core.js';

let coreId = -1;
let kernel = null;
let stateView = null; // Float64Array view over the shared state buffer

self.onmessage = async (e) => {
  const msg = e.data;
  switch (msg.type) {
    case 'init': {
      coreId = msg.coreId;
      await init({ module_or_path: new URL('./pkg/qupu_core_bg.wasm', import.meta.url) });
      kernel = new QKernel(msg.nQubits);
      stateView = new Float64Array(msg.sab);
      self.postMessage({ type: 'ready', coreId });
      break;
    }

    case 'init_zero': {
      // Only the worker that owns absolute index 0 sets amp[0] = 1.
      const t0 = performance.now();
      const sub = stateView.subarray(msg.baseIndex * 2, (msg.baseIndex + msg.nAmps) * 2);
      kernel.init_zero(sub, msg.containsZero);
      self.postMessage({
        type: 'done', coreId, jobId: msg.jobId,
        ms: performance.now() - t0, pairs: msg.nAmps / 2,
      });
      break;
    }

    case 'apply_1q': {
      const t0 = performance.now();
      const sub = stateView.subarray(msg.baseIndex * 2, (msg.baseIndex + msg.nAmps) * 2);
      kernel.apply_1q(sub, msg.target, msg.baseIndex, msg.matrix);
      self.postMessage({
        type: 'done', coreId, jobId: msg.jobId,
        ms: performance.now() - t0, pairs: msg.nAmps / 2,
      });
      break;
    }

    case 'apply_c1q': {
      const t0 = performance.now();
      const sub = stateView.subarray(msg.baseIndex * 2, (msg.baseIndex + msg.nAmps) * 2);
      kernel.apply_c1q(sub, msg.ctrl, msg.target, msg.baseIndex, msg.matrix);
      self.postMessage({
        type: 'done', coreId, jobId: msg.jobId,
        ms: performance.now() - t0, pairs: msg.nAmps / 2,
      });
      break;
    }

    case 'norm_sq': {
      const sub = stateView.subarray(msg.baseIndex * 2, (msg.baseIndex + msg.nAmps) * 2);
      const partial = kernel.norm_sq(sub);
      self.postMessage({ type: 'norm_partial', coreId, jobId: msg.jobId, partial });
      break;
    }
  }
};
