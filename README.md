# QuPU

A virtual **quantum processing unit** in the browser. The state-vector kernel is
written in Rust, compiled to WebAssembly, and dispatched across CPU Web Workers
the same way a GPU dispatches a kernel across thread blocks — except the unit of
parallel work is an amplitude pair rather than a pixel.

```
┌───────────── main thread ─────────────┐         ┌──── compute worker C0 ────┐
│  QuPU orchestrator                     │ msg →   │  WASM QKernel instance     │
│  SharedArrayBuffer ◄── state vector ──┼── view ─┤  applies gate to its tile  │
│  partitions each gate into N tiles     │ msg ←   └────────────────────────────┘
│  aligned to 2^(target+1) amplitudes    │         ┌──── compute worker C1 ────┐
│  barrier → next gate                   │ msg ←─→ │   ...                      │
└────────────────────────────────────────┘         └────────────────────────────┘
```

The state vector lives in a `SharedArrayBuffer`. Workers receive a typed-array
view into a slice of it; the alignment rule guarantees that the amplitude pair
`(i, i ^ (1<<target))` always falls inside a single worker's tile, so workers
never race. The top qubit reduces parallelism to one worker — same behavior you
see in distributed state-vector simulators on real hardware.

## Run it

```sh
npm run build   # one-time: cargo + wasm-pack → web/pkg
npm start       # dev server on :8080 with COOP/COEP for SharedArrayBuffer
```

Open <http://localhost:8080>.

## What you can do

- **Compose** circuits from `H, X, Y, Z, S, T, RX, RY, RZ, CNOT, CZ, SWAP`
  with selectable target/control qubits and a θ field for rotations.
- **Presets**: Bell state, GHZ, Quantum Fourier Transform, Grover (1 iteration),
  random Scramble layer.
- Scale the register up to 10 qubits (1024 amplitudes) live from the config
  panel.
- Watch the **die** light up as tiles dispatch, the **state vector** grid
  re-paint (size = magnitude, hue = phase), and **telemetry** track
  gates, wall time, parallel speedup, and norm drift.

## Layout

```
src/lib.rs        Rust kernel: state vector + apply_1q / apply_c1q / probabilities
web/qupu.js       Orchestrator: gate matrices, presets, tile chooser, dispatch
web/worker.js     Compute worker: holds a WASM instance + SAB view
web/main.js       UI: die, state grid, SVG circuit editor, render loop
web/index.html    Page shell
web/style.css     Aesthetic (deep navy / cyan / magenta, monospace)
server.mjs        Tiny dev server with COOP+COEP headers
```

## Verification

`cargo test --lib --release` runs five unit tests against the kernel,
including `tiled_h_matches_unsharded` — proves that splitting a gate across
multiple tiles produces a state vector bit-identical to the single-pass result.

## Why "quantum GPU"

A real GPU exposes a programming model where you write one kernel and the
hardware fans it across thousands of threads. A state-vector quantum simulator
has the same shape: one gate, fanned across amplitude pairs. We don't have
quantum hardware here — we have a Rust kernel and CPU worker threads. But the
dispatch shape, the workgroup-alignment math, and the per-core utilization
telemetry are the same. QuPU is that idea, made visible.

## Notes & limits

- Pure stable Rust 1.94. No nightly, no `-Zbuild-std`, no `wasm-bindgen-rayon`.
  Workers are JS-orchestrated; each holds its own independent WASM instance.
- `wasm-bindgen` marshalls `&mut [f64]` by copy-in / copy-out per call; for the
  tile sizes used (up to ~16 KB at 10 qubits) the overhead is negligible at
  visualization speeds. For a serious simulator you'd want shared-memory WASM
  threads — out of scope here.
- The gate set is universal; `SWAP` is decomposed into 3 CNOTs; multi-controlled
  gates in the Grover preset use a Toffoli decomposition for n=3 and a rougher
  CZ-ladder proxy for larger n.
